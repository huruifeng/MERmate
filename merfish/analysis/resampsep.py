import numpy as np
import numpy.linalg
import scipy

from utils.funcs import *


# Copyright 1993-2002 The MathWorks, Inc.
# $Revision: 1.8 $  $Date: 2002/03/15 15:58:39 $

# Defines the MEX function for separable resamplers.  MATLAB function
# syntax:
#
#   B = resampsep( A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R )
#
# resampsep() is an example of a resample_fcn as decribed in
# makeresampler.m.

# Implementation remarks:
# -----------------------
#
# Iterator example: Path through a 4 x 3 x 2 array
#
#    0   4   8           12  16  20
#    1   5   9           13  17  21
#    2   6  10           14  18  22
#    3   7  11           15  19  23
#
# Computing array offset from permuted and restructured cumulative
# product array
#
#   The basic idea is to perform all permutations, and as many multiplications
#   as possible, up front, before starting the processing loops.  So we
#   translate array subscripts to offsets in an unusual order.  The essence
#   is to construct the cumulative product of the array dimensions, shifted
#   forward with a one inserted at the front, then take its dot product with
#   the subscript array.  However, we partition this dot product into
#   contributions from transform dimensions and contributions from other
#   dimensions, permute the former, and compute the dot products at different
#   times.
#
#   Output array example:
#
#     size_B = [300 400   3   7  10]
#
#     tdims_B = [2 1 5]
#
#   Cumulative product of sizes, with 1 inserted at the front:
#
#     cp = [1   300   300*400  300*400*3   300#400*3*7]
#
#   Transform dimensions: Take the 2nd, 1st, and 5th elements of cp:
#
#     oCon->cpTrans = [300  1  300*400*3*7]
#
#   Other dimensions: Take what's left, in order:
#
#     oCon->cpOther = [300*400   300*400*3]
#
#   Total output offset:
#
#     (sum over k)( oCon->cpTrans[k] * oTransIt->subs[k] )
#
#     + (sum over j)( oCon->cpOther[j] * otherIt->subs[j] )
#
#   The sums are computed in Subs2Offset(), which is very simple and efficient
#   because of all the work done up front.
#
#   In ResampSep(), the outer loop goes over the transform dimensions and
#   the inner loop goes over the other dimensions, so we compute and save
#   the transform dimension contributions before starting the inner loop.
#
#   The input array offsets are handled using the same general idea. But the
#   contribution of the other dimensions to the offset is re-used from the
#   output array computation.  And the transform dimension contributions are
#   computed in the (innermost) convolution loop.


#========================== ResampSep =========================
# Resample using separable, shift-invariant kernels and arbitrary
# numbers of input and output dimensions. (There is one resampling
# kernel for each input transform dimension.)
#
# This function is implemented to include all offset computatations
# (mainly calls to Subs2Offset) and access to the input (A), output (B),
# and fill (F) arrays at the top level (rather than in function
# calls). Therefore, it's the only place where we need to worry
# about storage classes or imaginary parts.
#
# All dynamic memory allocation occurs in the first 11 executable
# lines. (There is no allocation inside the main loop over the
# output transform space.) The first 10 lines allocate working
# objects (which are freed in the 10 lines preceding the return)
# and the 11-th line allocates the return array.

def mxGetNumberOfDimensions(A): #ndims
    if isinstance(A, (int, float)):
        A = np.array([[A]])
    elif isinstance(A, list):
        A = np.array([A])
    return A.ndim

def mxGetDimensions(A): # shape
    if isinstance(A, (int, float)):
        A = np.array([[A]])
    elif isinstance(A, list):
        A = np.array([A])
    return A.shape


def GetCumProd(A):
    n = A.ndim # ndims: int
    d = A.shape # shape: [m,n,...]
    cumprod = [-1] * (n + 1)
    cumprod[0] = 1
    for k in range(n):
        cumprod[k+1] = d[k] * cumprod[k]
    return cumprod

def GetCount(A):
    if isinstance(A,list):
        return len(A)
    elif isinstance(A,(int,float)):
        return 1
    else:
        n = A.ndim
        d = A.shape  ## /* Elements of size(A) */
        count = d[0]
        for i in range(1,n):
            count *= d[i]
        return count

def NewConfig(fsize, tdims ):
    ########
    # Config = {
    #     int  ndims       /* total number of dimensions           */
    #     int  nTrans      /* number of transform dimensions       */
    #     int  nOther      /* number of non-transform dimensions   */
    #     int  tlength     /* total number of transform dimension elements */
    #     int* size        /* size of each dimension, in order     */
    #     int* tdims       /* position of each transform dimension */
    #     int* tsize       /* size of each transform dimension     */
    #     int* osize       /* size of each non-transform dimension */
    #     int* cpTrans     /* cumulative product at the position of each transform dimension */
    #     int* cpOther     /* cumulative product at the position of each non-transform dimension */
    # }
    c = {}

    c["ndims"]  = GetCount(fsize)
    c["nTrans"] = GetCount(tdims)
    c["nOther"] = c["ndims"] - c["nTrans"]

    c["size"]    = [0] * c["ndims"]
    c["tdims"]   = [0] * c["nTrans"]
    c["tsize"]   = [0] * c["nTrans"]
    c["cpTrans"] = [0] * c["nTrans"]
    c["osize"]   = [0] * c["nOther"]
    c["cpOther"] = [0] * c["nTrans"]

    isTrans = [0]*c["ndims"]
    cumprod = [0] * (1 + c["ndims"])
    cumprod[0] = 1
    for k in range(c["ndims"]):
        c["size"][k] = fsize[k]
        cumprod[k+1] = cumprod[k] * c["size"][k]

    c["tlength"] = 1
    for k in range(c["nTrans"]):
        #/* Subtract one to change to zero-based indexing */
        c["tdims"][k]   =  tdims[k]
        c["tsize"][k]   = c["size"][c["tdims"][k]]
        c["cpTrans"][k] = cumprod[c["tdims"][k]]
        isTrans[c["tdims"][k]] = 1
        c["tlength"] *= c["tsize"][k]

     # c->cpTrans contains the cumulative product components corresponding
     # to the transform dimensions, listed in the same order as c->tdims.
     # isTrans is 1 for all transform dimensions and 0 for the others.
     # Now copy the remaining sizes to c->osize and the remaining
     # cumulative product components to c->cpOther.

    j = 0
    for k in range(c["ndims"]):
        if not isTrans[k]:
            c["osize"][j]   = c["size"][k]
            c["cpOther"][j] = cumprod[k]
            j+=1

    return c

def NewKernel(kernelDef):
    k = None
    if kernelDef != None and IsKernelDefArray(kernelDef):
        halfwidth   = kernelDef[0]
        positiveHalf = kernelDef[1]
        k = {}
        k["halfwidth"] = halfwidth
        k["stride"] = np.ceil(2.0 * k["halfwidth"])
        k["nSamplesInPositiveHalf"] = GetCount(positiveHalf)
        k["positiveHalf"] = [0.0] * k["nSamplesInPositiveHalf"]
        k["indexFactor"] = (k["nSamplesInPositiveHalf"] - 1) / k["halfwidth"]
        for i in range(k["nSamplesInPositiveHalf"]):
            k["positiveHalf"][i] = positiveHalf[i]
    elif kernelDef == None or np.size(np.array(kernelDef))!=0:
        error(f"[Error]: NewKernel -> Kernel definition must be either empty or a cell array.")
    return k

def IsKernelDefArray(C ):
    return isinstance(C,(list,np.ndarray)) \
           and (GetCount(C) == 2) \
           and (GetCount(C[0]) == 1) \
           and (C[0] > 0.0) \
           and (GetCount((C[1])) >= 2)

def CreateKernelset(nTrans, K ):
    singleton = (GetCount(K) == 1)
    sharedKernelDef = None
    kernelset = [None] * nTrans

    if (not isinstance(K,(list,np.ndarray))) and (len(K) !=0 ):
        error("K (interpolating kernel array) should be either empty or a cell array.")

    if singleton or len(K)==0 or IsKernelDefArray(K):
        # /* Non-null K0 => All kernels are the same. */
        sharedKernelDef = K[0] if singleton else K

    for j in range(nTrans):
        kernelset[j] =  NewKernel(sharedKernelDef) if sharedKernelDef != None else NewKernel(K[j])

    return kernelset

def NewIterator(ndims,size ):

    i = {}
    i["length"] = 1
    i["size"] = [0] * ndims
    i["subs"] = [0] * ndims

    i["ndims"]  = ndims
    i["offset"] = 0
    for k in range(ndims):
        i["subs"][k] = 0
        i["size"][k] = size[k]
        i["length"] *= size[k]
    return i

def NewConvolver(iCon, K, padmethod ):
    c = {}

    c["padmethod"]  = padmethod
    c["ndims"]      = iCon["nTrans"]
    c["size"]       = [0] * c["ndims"]
    c["tsize"]      = [0] * c["ndims"]
    c["cumsum"]     = [0] * (c["ndims"] + 1)
    c["cumprod"]    = [0] * (c["ndims"] + 1)
    c["weights"]    = [0] * c["ndims"]
    c["tsub"]       = [0] * c["ndims"]
    c["kernelset"]  = CreateKernelset(c["ndims"], K)

    c["cumprod"][0] = 1
    c["cumsum"][0]  = 0
    for k in range(c["ndims"]):
        c["size"][k] = 1 if c["kernelset"][k] == None else c["kernelset"][k]["stride"]
        c["tsize"][k] = iCon["tsize"][k]
        c["cumsum"][k+1]  = c["size"][k] + c["cumsum"][k]
        c["cumprod"][k+1] = c["size"][k] * c["cumprod"][k]

    c["weight_data"] = [0] * int(c["cumsum"][c["ndims"]])
    c["tsub_data"]   = [0] * int(c["cumsum"][c["ndims"]])
    for k in range(c["ndims"]):
        c["weights"][k] = c["weight_data"][int(c["cumsum"][k])]
        c["tsub"][k]    = c["tsub_data"][int(c["cumsum"][k])]

    c["nPoints"]     = int(c["cumprod"][c["ndims"]])
    c["useFill"]     = [0] * c["nPoints"]
    c["subs"]        = [0] * c["nPoints"]
    c["subs_data"]   = [0] * int(c["nPoints"] * c["ndims"])

    for j in range (c["nPoints"]):
        c["subs"][j] = c["subs_data"][j * c["ndims"]]

    c["lo"] = [0] * c["ndims"]
    c["hi"] = [0] * c["ndims"]
    for k in range(c["ndims"]):
        h = c["kernelset"][k]["halfwidth"] if c["kernelset"][k] != None and padmethod == "fill" else 0.5
        if c["tsize"][k] >= 1:
            c["lo"][k] = -h
            c["hi"][k] = (c["tsize"][k] - 1) + h
        else: # /* Never in bounds if tsize is zero. */
            c["lo"][k] =  1
            c["hi"][k] = -1

    c["localIt"]   = NewIterator(c["ndims"], c["size"])
    return c

def CreateCPFill(nOther, F):
    # /*
    #  * Prepare for efficient computation of offsets into the fill
    #  * array F. Create a special cumulative product array for F.
    #  * Its length is nOther (which may be greater than NDIMS(F)) and
    #  * elements corresponding to a value of 1 in SIZE(F) are set to zero.
    #  * The result can be multiplied (dot product via Subs2Offset) with
    #  * a set of non-transform subscripts to produce the correct offset into F.
    #  */

    ndims_F = mxGetNumberOfDimensions(F)
    size_F  = mxGetDimensions(F)
    cpFill  =[0] * nOther

    partialprod = 1

    k = 0
    while(k < ndims_F and k < nOther):
        if (size_F[k] == 1):
            cpFill[k] = 0
        else:
            cpFill[k] = partialprod

        partialprod *= size_F[k]
        k += 1

    k = ndims_F
    while(k < nOther):
        cpFill[k] = 0
        k += 1

    return cpFill

def mxCreateNumericArray(dims, ComplexFlag):
    dtype = np.complex_ if ComplexFlag=="mxCOMPLEX" else np.double
    mxA = np.empty(shape = dims,dtype=dtype)
    mxA[:]=np.nan

    return mxA

def resetIterator(i):
    i["offset"] = 0
    for k in range(i["ndims"]):
        i["subs"][k] = 0
    return i

def incrementIterator(i):
    done = 0
    k = 0
    while ((not done) and (k < i["ndims"])):
        i["subs"][k] += 1
        if (i["subs"][k] < i["size"][k]):
            done = 1
        else:
            i["subs"][k] = 0
            k += 1
    i["offset"] += 1
    return i

def doneIterating(i):
    return i["offset"] >= i["length"]

def Subs2Offset(ndims, cumprod, subs ):
    offset = 0
    assert ndims > 0, "ndims is less or equal to 0!"
    assert subs != None, "subs is None!"
    assert cumprod != None, "cumprod is None!"

    for  k in range(ndims):
        offset += subs[k] * cumprod[k];

    return offset

def Convolve(c, v):
    k = c["ndims"] - 1
    while k >=0 :
        n = int(c["size"][k])
        w = c["weights"][k]

        # /* Use the cumulative size product to iterate over the first k-1 dimensions */
        # /* of v -- all the uncollapsed dimensions except for the highest.           */
        # /* (For each value of q we visit a different point in this subspace.)       */
        s = int(c["cumprod"][k])
        for q in range(s):
            # /* Take the inner product of the k-th weight array and a (1-D) profile    */
            # /* across the highest (i.e., k-th) uncollapsed dimension of v.  Re-use    */
            # /* memory by storing the result back in the first element of the profile. */
            t = 0.0
            for r in range(n):
                t +=  (w[r] * v[q + r * s])
            v[q] = t
        k -= 1

    # /* After repeated iterations, everything is collapsed into the */
    # /* first element of v. It is now zero-dimensional (a scalar).  */
    return v[0]

def AdjustSubscript(subscript, tsize,padMethod ):
    sub = subscript
    if tsize <= 0: return -1  # /* Avoid potential zero-divide with empty input array */

    if padMethod == "fill":
        sub = -1 if sub < 0 else ( -1 if sub >= tsize else sub)
    elif padMethod == "bound" or padMethod == "replicate":
        sub = 0 if sub < 0 else ( tsize-1 if sub >= tsize else sub)
    elif padMethod == "circular":
        sub = sub % tsize if sub >= 0 else (tsize - 1 - ((-sub - 1) % tsize) )
    elif padMethod == "symmetric":
        tsize2 = 2 * tsize
        sub =  sub % tsize2 if sub >= 0 else (tsize2 - 1 - ((-sub - 1) % tsize2) )
        sub = (tsize2 - sub - 1) if sub >= tsize else sub
    return sub

def EvaluateKernel(k, t ):
    result = 0.0

    # /* To do: Decide which side should have equality. (We're doing a convolution.) */
    if  -k["halfwidth"] < t and t <= k["halfwidth"]:
        x = k["indexFactor"] * np.fabs(t)
        index = int(x)   # /* This is equivalent to (int)floor(x) if x>0 */
        if index >= (k["nSamplesInPositiveHalf"] - 1 ):
            result = k["positiveHalf"][k["nSamplesInPositiveHalf"] - 1]
        else:
            # /* WJ Surprisingly, removing this operation, by replacing it with
            #    result = k->positiveHalf[index]; did not provide much of a
            #    speedup */
            w1 = x - index
            w0 = 1.0 - w1
            result = w0 * (k["positiveHalf"][index]) +  w1 * (k["positiveHalf"][index + 1]);
    return result


def initConvolver( c, p):
    for k in range(c["ndims"]):
        # /* Either use the kernel or simply round p[k] for nearest neighbor resampling. */
        c["weights"][k] = []
        c["tsub"][k] = []
        if c["kernelset"][k] != None:
            # /*
            #  * Translate the kernel so that its center is at center, then
            #  * return the lowest integer for which the kernel is defined.
            #  */
            s0 = np.ceil(p[k] - c["kernelset"][k]["halfwidth"])
            for j in range(int(c["size"][k])):
                s = s0 + j
                # /* use the kernel */
                c["weights"][k].append(EvaluateKernel(c["kernelset"][k], p[k] - s))
                c["tsub"][k].append(AdjustSubscript( s, c["tsize"][k], c["padmethod"] ))

        else:
            # /* use MATLAB-Compatible Rounding Function */
	        # /* this used to be (int)floor(p[k]+0.5); */
            s0 =  (int)(p[k]-0.5) if p[k] < 0.0  else (int)(p[k] + 0.5)

            for j in range(c["size"][k]):
                s = s0 + j
                # /* Set the single weight to unity for nearest neighbor resampling. */
                c["weights"][k][j] = 1.0
                c["tsub"][k][j] = AdjustSubscript( s, c["tsize"][k], c["padmethod"])

    # /* Save the outer product set of c->tsub in t->localsubs */
    c["localIt"] = resetIterator(c["localIt"])
    while not doneIterating(c["localIt"]):
        c["useFill"][c["localIt"]["offset"]] = 0
        c["subs"][c["localIt"]["offset"]] = []
        for k in range(c["ndims"]):
            s = c["tsub"][k][c["localIt"]['subs'][k]]
            c["subs"][c["localIt"]["offset"]].append(s)

            # /* Turn on useFill if the adjusted subscript is less than one in any of the dimensions. */
            if s == -1: c["useFill"][c["localIt"]["offset"]] = 1
        c["localIt"] = incrementIterator(c["localIt"])

    return c

def UseConvolution(c, p ):
    if c["padmethod"] == "fill" or c["padmethod"] == "bound":
        for k in range(c["ndims"]):
            if not (c["lo"][k] <= p[k] and p[k] < c["hi"][k]): return 0

    return 1


## example:
#


def do_resampsep(A,M,tdims_A,tdims_B,fsize_A,fsize_B,F,padstr,K):
    # const mxArray*  A,         /* Input array */
    # const mxArray*  M,         /* Inverse mapping from output to input */
    # const mxArray*  tdims_A,   /* List of input transform dimensions */
    # const mxArray*  tdims_B,   /* List of output transform dimensions */
    # const mxArray*  fsize_A,   /* Full size of input array */
    # const mxArray*  fsize_B,   /* Full size of output block */
    # const mxArray*  F,         /* Defines the fill values and dimensions iff non-empty */

    # const mxArray*  padstr,    /* Method for defining values for points that map outside A */
    # const mxArray*  K          /* Interplating kernel cell array */ )

    # Working objects and storage:
    #   cumprodM    Cumulative product array for computing offsets into M
    #   iCon:       Configuration of input dimensions
    #   oCon:       Configuration of output dimensions
    #   convolver:  Weights and subscripts needed for convolution
    #   oTransIt:   Iterator for output transform space
    #   otherIt:    Iterator for non-transform space
    #   cpFill:     Cumulative products for computing offsets into the fill array
    #   p:          Current point in input transform space
    #   vReal:      Input values to be used in convolution (real parts)
    #   vImag:      Input values to be used in convolution (imaginary parts)
    #
    # Return value:
    #   B:          Output array

    cumprodM = GetCumProd(M)
    iCon = NewConfig(fsize_A, tdims_A )
    oCon = NewConfig(fsize_B, tdims_B )

    PadMethod = ["fill", "bound", "replicate", "circular", "symmetric"]
    if padstr not in PadMethod:
        error("[Error]: PadMethod-> padmethod must be 'fill', 'bound', 'replicate', 'circular', or 'symmetric'.")
    convolver = NewConvolver( iCon, K, padstr)
    oTransIt = NewIterator( oCon["nTrans"], oCon["tsize"] )
    otherIt = NewIterator( oCon["nOther"], oCon["osize"] )
    cpFill = CreateCPFill( oCon["nOther"], F )
    p = [0] * iCon["nTrans"]
    vReal = [0] * convolver["nPoints"]
    vImag = [0] * convolver["nPoints"]

    bComplexA = np.iscomplexobj(A)
    bComplexF = np.iscomplexobj(F)

    B = mxCreateNumericArray(oCon["size"], ("mxCOMPLEX" if bComplexA else "mxREAL") )

    ptrMr = M.flatten('F') # Get pointer to M array
    ptrFr = np.array(F).flatten() # Get pointer to F array

    bFempty = (np.size(np.array(F))==0) # For checking if we need to pad

    ptrAr = A.flatten('F')   # Get pointers to real and imaginary parts
    ptrBr = B.flatten('F')

    ptrBi = None
    ptrAi = None
    ptrFi = None

    if bComplexA:
        ptrAi = A.imag
        ptrBi = B.imag

    if bComplexF:
        ptrFi= F.imag

    # Loop over the output transform space
    oTransIt = resetIterator(oTransIt)
    while(not doneIterating(oTransIt)):
        useConvolution = 0

        # Cache transform portion of the output offset
        oTransOffset = Subs2Offset(oCon["nTrans"], oCon["cpTrans"], oTransIt["subs"])

        # Cache output transform portion of the offset into M
        mTransOffset = Subs2Offset(oCon["nTrans"], cumprodM, oTransIt["subs"])

        if( not np.isnan(ptrMr[mTransOffset])):
            # Extract from M the current point in input transform space. (And
            # subtract one to convert the values in M to a zero-based system.)
            for k in range(iCon["nTrans"]):
                mOffset = mTransOffset + k * (oCon["tlength"])
                p[k] =  ptrMr[mOffset]
            useConvolution = UseConvolution(convolver,p)

            # If necessary, construct subscript and weight arrays in the input transform space
            # (Note that subscripts are shifted as necessary to handle out-of-range points.)
            if useConvolution: convolver = initConvolver(convolver, p)

        # Loop over the non-transform output space, assigning one (scalar or complex) value on each pass
        otherIt = resetIterator(otherIt)
        while not doneIterating(otherIt):
            totalOutputOffset = oTransOffset+ Subs2Offset( oCon["nOther"], oCon["cpOther"], otherIt["subs"] )
            fReal = 0.0
            fImag = 0.0
            oReal = 0.0
            oImag = 0.0

            # Get fill values from F now in case they're needed in the convolution loop.
            if not bFempty:
                fOffset = Subs2Offset(oCon["nOther"], cpFill, otherIt["subs"])
                fReal = ptrFr[fOffset]
                if bComplexF: fImag = ptrFi[fOffset]

            if useConvolution:
                iOtherOffset = Subs2Offset(iCon["nOther"], iCon["cpOther"], otherIt["subs"])
                for j in range(convolver["nPoints"]):   # This is the convolution loop.
                    if convolver["useFill"][j]:
                        # Mix fill values with image values near an edge (pad must be 'fill').
                        vReal[j] = fReal
                        if bComplexA: vImag[j] = fImag
                    else:
                        totalInputOffset = int(iOtherOffset + Subs2Offset(iCon["nTrans"], iCon["cpTrans"], convolver["subs"][j]))
                        vReal[j] = ptrAr[totalInputOffset]
                        if bComplexA:vImag[j] = ptrAi[totalInputOffset]

                oReal = Convolve(convolver, vReal)
                if bComplexA: oImag = Convolve(convolver, vImag)
            else:
                oReal = fReal
                oImag = fImag

            ptrBr[totalOutputOffset] = oReal
            if bComplexA: ptrBi[totalOutputOffset] = oImag

            otherIt = incrementIterator(otherIt)

        oTransIt = incrementIterator(oTransIt)

    if B.shape[-1]==1:
        B=ptrBr.reshape(B.shape[:-1]).T
    return B

# def resampsep(A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R):
#     # const mxArray*  A,         /* Input array */
#     # const mxArray*  M,         /* Inverse mapping from output to input */
#     # const mxArray*  tdims_A,   /* List of input transform dimensions */
#     # const mxArray*  tdims_B,   /* List of output transform dimensions */
#     # const mxArray*  fsize_A,   /* Full size of input array */
#     # const mxArray*  fsize_B,   /* Full size of output block */
#     # const mxArray*  F,         /* Defines the fill values and dimensions if non-empty */
#     # const mxArray * R = prhs[7]; / *Resampler * /
#
#     # const mxArray*  padstr,    /* Method for defining values for points that map outside A */
#     # const mxArray*  K          /* Interplating kernel cell array */ )
#
#     if "rdata" not in R:
#         error("[Error]: resampsep - rdata field is missing.")
#     else:
#         rdata = R["rdata"]
#
#     if "padmethod" not in R:
#         error("[Error]: resampsep - padmethod field is missing.")
#     else:
#         padstr = R["padmethod"]
#
#     if "K" not in rdata:
#         error("[Error]: resampsep - rdata['K'] field is missing.")
#     else:
#         K = rdata["K"]
#
#     if isinstance(tdims_A,list):
#         tdims_A = np.array(tdims_A)
#     if isinstance(tdims_B,list):
#         tdims_B = np.array(tdims_B)
#
#
#     R = do_resampsep(A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, padstr, K)
#     return R



# if __name__ == '__main__':
#     # A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R
#     import pickle
#     example = pickle.load(open("example.pkl", "rb"))
#     [A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R] = example
#
#     fsize_A = [128,128,1]
#     fsize_B = [128, 128, 1]
#     B = resampsep(A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R)