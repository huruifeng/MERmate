import gc

import numpy as np
import numpy.linalg
import scipy
import cv2
import matplotlib.pyplot as plt

from merfish.analysis.resampler import makeresampler
from utils.funcs import *
from utils.misc import *

condlimit = 1e9

def maketform(*args):
    # MAKETFORM Create spatial transformation structure (TFORM).  
    #   MAKETFORM is not recommended. Use AFFINE2D, PROJECTIVE2D, AFFINE3D, or
    #   FITGEOTRANS instead.
    #
    #   T = MAKETFORM(TRANSFORMTYPE,...) creates a multidimensional spatial
    #   transformation structure (a 'TFORM struct') that can be used with
    #   TFORMFWD, TFORMINV, FLIPTFORM, IMTRANSFORM, or TFORMARRAY.
    #   TRANSFORMTYPE can be 'affine', 'projective', 'custom', 'box', or
    #   'composite'. Spatial transformations are also called geometric
    #   transformations.
    #
    #   T = MAKETFORM('affine',A) builds a TFORM struct for an N-dimensional
    #   affine transformation.  A is a nonsingular real (N+1)-by-(N+1) or
    #   (N+1)-by-N matrix.  If A is (N+1)-by-(N+1), then the last column
    #   of A must be [zeros(N,1) 1].  Otherwise, A is augmented automatically
    #   such that its last column is [zeros(N,1) 1].  A defines a forward
    #   transformation such that TFORMFWD(U,T), where U is a 1-by-N vector,
    #   returns a 1-by-N vector X such that X = U * A(1:N,1:N) + A(N+1,1:N).
    #   T has both forward and inverse transformations.
    #
    #   T = MAKETFORM('projective',A) builds a TFORM struct for an N-dimensional
    #   projective transformation.  A is a nonsingular real (N+1)-by-(N+1)
    #   matrix.  A(N+1,N+1) cannot be 0.  A defines a forward transformation
    #   such that TFORMFWD(U,T), where U is a 1-by-N vector, returns a 1-by-N
    #   vector X such that X = W(1:N)/W(N+1), where W = [U 1] * A.  T has
    #   both forward and inverse transformations.
    #   
    #   T = MAKETFORM('affine',U,X) builds a TFORM struct for a
    #   two-dimensional affine transformation that maps each row of U
    #   to the corresponding row of X.  U and X are each 3-by-2 and
    #   define the corners of input and output triangles.  The corners
    #   may not be collinear.
    #
    #   T = MAKETFORM('projective',U,X) builds a TFORM struct for a
    #   two-dimensional projective transformation that maps each row of U
    #   to the corresponding row of X.  U and X are each 4-by-2 and
    #   define the corners of input and output quadrilaterals.  No three
    #   corners may be collinear.
    #
    #   T = MAKETFORM('custom',NDIMS_IN,NDIMS_OUT,FORWARD_FCN,INVERSE_FCN,
    #   TDATA) builds a custom TFORM struct based on user-provided function
    #   handles and parameters.  NDIMS_IN and NDIMS_OUT are the numbers of
    #   input and output dimensions.  FORWARD_FCN and INVERSE_FCN are
    #   function handles to forward and inverse functions.  Those functions
    #   must support the syntaxes X = FORWARD_FCN(U,T) and U =
    #   INVERSE_FCN(X,T), where U is a P-by-NDIMS_IN matrix whose rows are
    #   points in the transformation's input space, and X is a
    #   P-by-NDIMS_OUT matrix whose rows are points in the transformation's
    #   output space.  TDATA can be any MATLAB array and is typically used to
    #   store parameters of the custom transformation.  It is accessible to
    #   FORWARD_FCN and INVERSE_FNC via the "tdata" field of T.  Either
    #   FORWARD_FCN or INVERSE_FCN can be empty, although at least
    #   INVERSE_FCN must be defined to use T with TFORMARRAY or IMTRANSFORM.
    #
    #   T = MAKETFORM('composite',T1,T2,...,TL) or T = MAKETFORM('composite',
    #   [T1 T2 ... TL]) builds a TFORM whose forward and inverse functions are
    #   the functional compositions of the forward and inverse functions of the
    #   T1, T2, ..., TL.  Note that the inputs T1, T2, ..., TL are ordered just
    #   as they would be when using the standard notation for function
    #   composition: 
    #
    #      T = T1 o T2 o ... o TL 
    #
    #   and note also that composition is associative, but not commutative.
    #   This means that in order to apply T to the input U, one must apply TL
    #   first and T1 last. Thus if L = 3, for example, then TFORMFWD(U,T) is
    #   the same as TFORMFWD(TFORMFWD(TFORMFWD(U,T3),T2),T1). The components
    #   T1 through TL must be compatible in terms of the numbers of input and
    #   output dimensions.  T has a defined forward transform function only if
    #   all of the component transforms have defined forward transform
    #   functions.  T has a defined inverse transform function only if all of
    #   the component functions have defined inverse transform functions.
    #
    #   T = MAKETFORM('box',TSIZE,LOW,HIGH) or T = MAKETFORM('box',INBOUNDS,
    #   OUTBOUNDS) builds an N-dimensional affine TFORM struct, T.  TSIZE is
    #   an N-element vector of positive integers, and LOW and HIGH are also
    #   N-element vectors.  The transformation maps an input "box" defined
    #   by the opposite corners ONES(1,N) and TSIZE or, alternatively, by
    #   corners INBOUNDS(1,:) and INBOUND(2,:) to an output box defined by
    #   the opposite corners LOW and HIGH or OUTBOUNDS(1,:) and OUTBOUNDS(2,:).
    #   LOW(K) and HIGH(K) must be different unless TSIZE(K) is 1, in which
    #   case the affine scale factor along the K-th dimension is assumed to be
    #   1.0.  Similarly, INBOUNDS(1,K) and INBOUNDS(2,K) must be different
    #   unless OUTBOUNDS(1,K) and OUTBOUNDS(1,K) are the same, and vice versa.
    #   The 'box' TFORM is typically used to register the row and column
    #   subscripts of an image or array to some "world" coordinate system.
    #
    #   Example
    #   -------
    #   Make and apply an affine transformation.
    #
    #       T = maketform('affine',[.5 0 0 .5 2 0 0 0 1])
    #       tformfwd([10 20],T)
    #       I = imread('cameraman.tif')
    #       transformedI = imtransform(I,T)
    #       figure, imshow(I), figure, imshow(transformedI)
    #
    #   See also AFFINE2D, AFFINE3D, FITGEOTRANS, FLIPTFORM, PROJECTIVE2D,
    #            IMTRANSFORM, TFORMARRAY, TFORMFWD, TFORMINV.
    
    #   Copyright 1993-2017 The MathWorks, Inc.
    
    # Testing notes
    # Syntaxes
    #---------
    # T = MAKETFORM( 'affine', A )
    #
    # A:        Numeric, non-singular, real square matrix (no Infs or Nans).
    #           Last column must be zero except for a one in the lower right corner.
    
    if len(args) <=0:
        raise ("[Error]: No input arguments provided for maketform().")

    transform_type =args[0]
    
    if transform_type == 'affine': fcn = affine
    elif transform_type == 'projective': fcn = projective
    elif transform_type == 'composite': fcn = composite
    elif transform_type == 'custom': fcn = custom
    elif transform_type == 'box': fcn = box
    else:
       raise('[Error]:maketform:unknownTransformType' + str(args[0]))
    
    t = fcn(args[1:])

    return t


#--------------------------------------------------------------------------
def assigntform(ndims_in, ndims_out, forward_fcn, inverse_fcn, tdata):
    # Use this function to ensure consistency in the way we assign
    # the fields of each TFORM struct.
    t = {}
    t["ndims_in"]    = ndims_in
    t["ndims_out"]   = ndims_out
    t["forward_fcn"] = forward_fcn
    t["inverse_fcn"] = inverse_fcn
    t["tdata"]       = tdata
    return t

#-------------------------------------------------------------------------
def affine(args):
    # Build an affine TFORM struct.

    if len(args) <= 0 or len(args) >2:
        raise ("[Error]: Invalid number of arguments provided for affine().")
    
    
    if len(args) == 2:
        # Construct a 3-by-3 2-D affine transformation matrix A
        # that maps the three points in X to the three points in U.
        U = args[0]
        X = args[1]
        A = construct_matrix( U, X, 'affine' ) 
        A[:,2] = [0, 0, 1]  # Clean up noise before validating A.
    else:
        A = args[0]

    A = validate_matrix( A, 'affine' )

    N = A.shape[1] - 1
    tdata = {}
    tdata['T']    = A
    tdata['Tinv'] = np.linalg.inv(A)
    
    # In case of numerical noise, coerce the inverse into the proper form.
    tdata['Tinv'][:-1,-1] = 0
    tdata['Tinv'][-1,-1] = 1
    
    t = assigntform(N, N, fwd_affine, inv_affine, tdata)
    return t

#--------------------------------------------------------------------------
def inv_affine( X, t ):
    # INVERSE affine transformation 
    #
    # T is an affine transformation structure. X is the row vector to
    # be transformed, or a matrix with a vector in each row.
    
    U = trans_affine(X, t, 'inverse')
    return U
    
#--------------------------------------------------------------------------
def fwd_affine(U, t):
    # FORWARD affine transformation 
    #
    # T is an affine transformation structure. U is the row vector to
    # be transformed, or a matrix with a vector in each row.
    
    X = trans_affine(U, t, 'forward')
    return X

#--------------------------------------------------------------------------
def trans_affine( X, t, direction ):
    # Forward/inverse affine transformation method
    #
    # T is an affine transformation structure. X is the row vector to
    # be transformed, or a matrix with a vector in each row.
    # DIRECTION is either 'forward' or 'inverse'.
    
    if direction == 'forward':
        M = t["tdata"]["T"]
    elif direction == 'inverse':
        M = t["tdata"]["Tinv"]
    else:
        raise ('[Error]:maketform:invalidDirection')
    
    X1 =np.hstack([X, np.ones((X.shape[0],1))])   # Convert X to homogeneous coordinates
    U1 = X1 @ M                  # Transform in homogeneous coordinates
    U  = U1[:,:-1]           # Convert homogeneous coordinates to U
    return U

#---------------------------------------------------------------
def construct_matrix( U, X, transform_type ):
    # Construct a 3-by-3 2-D transformation matrix A
    # that maps the points in U to the points in X.

    if transform_type == 'affine':
        nPoints = 3
        unitFcn = UnitToTriangle
    elif transform_type == 'projective':
        nPoints = 4
        unitFcn = UnitToQuadrilateral


    if (U.shape[0] not in [nPoints, 2]) or (U.shape[1] not in [nPoints,2]):
        raise('images:maketform:invalidUSize'+str(nPoints))

    if  (X.shape[0] not in [nPoints, 2]) or (X.shape[1] not in [nPoints,2]):
        raise('images:maketform:invalidXSize'+str(nPoints))


    Au = unitFcn(U)
    if np.linalg.cond(Au) > condlimit:
        print('[Warning]: images:maketform:conditionNumberOfUIsHigh')

    Ax = unitFcn(X)
    if np.linalg.cond(Ax) > condlimit:
        print('[Warning]: images:maketform:conditionNumberOfXIsHigh')

    # (unit shape) * Au = U
    # (unit shape) * Ax = X
    #
    # U * inv(Au) * Ax = (unit shape) * Ax = X and U * A = X,
    # so inv(Au) * Ax = A, or Au * A = Ax, or A = Au \ Ax.

    # A = Au \ Ax
    if Au.shape[0] == Au.shape[1]:
        A = np.linalg.solve(Au , Ax)
    else:
        A = np.linalg.lstsq(Au, Ax)

    if np.any(np.isinf(A)):
        raise('[Error]: images:maketform:collinearPointsinUOrX')

    A = A / A[-1,-1]
    return A

#---------------------------------------------------------------

def validate_matrix( A, transform_type ):

    # Make sure A is finite.
    if np.any(np.isinf(A)):
        raise('images:maketform:aContainsInfs')

    # Make sure A is (N + 1)-by-(N + 1).  Append a column if needed for 'affine'.
    N = A.shape[0] - 1
    if transform_type=='affine' and A.shape[1] == N:
        A[:,N] = np.append(np.zeros((N,1)) , [1])

    if N < 1 or (A.shape[1]!= N + 1):
        error('images:maketform:invalidASize')

    if transform_type == 'affine':
          # Validate the final column of A.
          if np.any(np.not_equal(A[:,N], np.append(np.zeros((N,1)),[1]))):
              error('images:maketform:invalidAForAffine')
    elif transform_type == 'projective':
        # Validate lower right corner of A
          if abs(A[N,N]) <= 100 * np.finfo(float).eps * np.linalg.norm(A,2):
            print('[Warning]:images:maketform:lastElementInANearZero')

    if np.linalg.cond(A) > condlimit:
        print('[Warning]: images:maketform:conditionNumberofAIsHigh', np.linalg.cond(A))

    return A

#---------------------------------------------------------------

def UnitToTriangle( X ):

    # Computes the 3-by-3 two-dimensional affine transformation
    # matrix A that maps the unit triangle ([0 0], [1 0], [0 1])
    # to a triangle with corners (X(1,:), X(2,:), X(3,:)).
    # X must be 3-by-2, real-valued, and contain three distinct
    # and non-collinear points. A is a 3-by-3, real-valued matrix.

    A = np.array([[X[1,0]-X[0,0], X[1,1]-X[0,1], 0],
                  [X[2,0]-X[0,0], X[2,1]-X[0,1], 0],
                  [X[0,0],X[0,1],1]])
    return A
#---------------------------------------------------------------

def UnitToQuadrilateral( X ):
    # Computes the 3-by-3 two-dimensional projective transformation
    # matrix A that maps the unit square ([0 0], [1 0], [1 1], [0 1])
    # to a quadrilateral corners (X(1,:), X(2,:), X(3,:), X(4,:)).
    # X must be 4-by-2, real-valued, and contain four distinct
    # and non-collinear points.  A is a 3-by-3, real-valued matrix.
    # If the four points happen to form a parallelogram, then
    # A(1,3) = A(2,3) = 0 and the mapping is affine.
    #
    # The formulas below are derived in
    #   Wolberg, George. "Digital Image Warping," IEEE Computer
    #   Society Press, Los Alamitos, CA, 1990, pp. 54-56,
    # and are based on the derivation in
    #   Heckbert, Paul S., "Fundamentals of Texture Mapping and
    #   Image Warping," Master's Thesis, Department of Electrical
    #   Engineering and Computer Science, University of California,
    #   Berkeley, June 17, 1989, pp. 19-21.

    x = X[:,0]
    y = X[:,1]

    dx1 = x[1] - x[2]
    dx2 = x[3] - x[2]
    dx3 = x[0] - x[1] + x[2] - x[3]

    dy1 = y[1] - y[2]
    dy2 = y[3] - y[2]
    dy3 = y[0] - y[1] + y[2] - y[3]

    if dx3 == 0 and dy3 == 0:
        # Parallelogram: Affine map
        A = np.array([ [x[1] - x[0],    y[1] - y[0],   0],
                       [x[2] - x[1],    y[2] - y[1],   0],
                       [x[0] ,          y[0],          1] ])
    else:
        # General quadrilateral: Projective map
        a13 = [dx3 * dy2 - dx2 * dy3] / [dx1 * dy2 - dx2 * dy1]
        a23 = [dx1 * dy3 - dx3 * dy1] / [dx1 * dy2 - dx2 * dy1]

        A = np.array([[x[1] - x[0] + a13 * x[1],   y[1] - y[0] + a13 * y[1],   a13],
                      [x[3] - x[0] + a23 * x[3],   y[3] - y[0] + a23 * y[3],   a23],
                      [x[0] ,                      y[0],                       1] ])

    return A


#-------------------------------------------------
def projective():
    pass

def box(args):
    if len(args) == 3:
        # Construct an affine TFORM struct that maps a box bounded by 1 and TSIZE(k)
        # in dimension k to a box bounded by LO(k) and HI(k) in dimension k.
        # Construct INBOUNDS and OUTBOUNDS arrays, then call BOX2.

        tsize = np.array(args[0])
        lo    = args[1]
        hi    = args[2]

        if np.any(tsize < 1 ):
            error('images:maketform:tSizeIsNotPositive')

        if not isinstance(lo[0],(int, float)):
            error('images:maketform:invalidLo')

        if not isinstance(hi[0],(int, float)):
            error('images:maketform:invalidHi')

        N = len(tsize)
        if len(lo) != N or len(hi) != N:
            error('images:maketform:unequalLengthsForLoHiAndTSize')

        if np.any((lo == hi) & (~(tsize == 1))):
            error('images:maketform:invalidLoAndHi')

        inbounds  = np.vstack((np.zeros((1,N)), tsize))
        outbounds = np.vstack((lo, hi))
    else:
        inbounds  = args[0]
        outbounds = args[1]

    t = box2(inbounds,outbounds)

    return t

def box2( inBounds, outBounds ):
    # Construct an affine TFORM struct that maps a box bounded by INBOUNDS(1,k)
    # and INBOUNDS(2,k) in dimensions k to a box bounded by OUTBOUNDS(1,k) and
    # OUTBOUNDS(2,k).
    #
    # inBounds:   2-by-N
    # outBounds:  2-by-N

    N = inBounds.shape[1]
    if (np.ndim(inBounds) != 2) or \
            (np.ndim(outBounds)  != 2) or \
            (inBounds.shape[0] != 2) or \
            (outBounds.shape[0] != 2) or \
            (outBounds.shape[1] != N):
        error('images:maketform:inboundsAndOutbounds2ByN')

    qDegenerate  = (inBounds[0,:] == inBounds[1,:])
    if np.any((outBounds[0,:] == outBounds[1,:]) != qDegenerate):
        error('images:maketform:invalidInboundsAndOutbounds')

    num = outBounds[1,:] - outBounds[0,:]
    den =  inBounds[1,:] -  inBounds[0,:]

    # Arbitrarily set the scale to unity for degenerate dimensions.
    num[qDegenerate] = 1
    den[qDegenerate] = 1

    tdata={}
    tdata["scale"] = num / den
    tdata["shift"] = outBounds[0,:] - tdata["scale"] * inBounds[0,:]

    t = assigntform(N, N, fwd_box, inv_box, tdata)
    return t

def inv_box( X, t ):
    # INVERSE box transformation
    #
    # T is an box transformation structure. X is the row vector to
    # be transformed, or a matrix with a vector in each row.

    U = X - t["tdata"]["shift"]
    U = U / t["tdata"]["scale"]
    return U

#--------------------------------------------------------------------------
def fwd_box( U, t):
    # FORWARD box transformation
    #
    # T is an box transformation structure. U is the row vector to
    # be transformed, or a matrix with a vector in each row.

    X = U * t["tdata"]["scale"]
    X = X + t["tdata"]["shift"]
    return X



#---------------------------------------------------------------
def composite(tdata_ls):
    # Construct COMPOSITE transformation structure.

    # Create TDATA as a TFORM structure array.
    if len(tdata_ls) == 0:
        error('[Error]:maketform:tooFewTformStructs')
    elif len(tdata_ls)==1:
        t = tdata_ls[0]
        if isinstance(t,list):
            if len(t) == 1:
                return t[0]
            else:
                tdata_ls = t
        else:
            error("'[Error]:maketform:invalidTformStructArray'")
    else:
        pass

    # Check for consistency of dimensions
    N = len(tdata_ls)
    ndims_in =  [i["ndims_in"] for i in tdata_ls]
    ndims_out = [i["ndims_out"] for i in tdata_ls]

    if not(ndims_in[0:N-1] == ndims_out[1:N]):
        error('[Error]:maketform:tFormsDoNotHaveSameDimension')

    # Check existence of forward and inverse function handles
    forward_fcn = fwd_composite
    for i in tdata_ls:
        if not callable(i["forward_fcn"]):
            forward_fcn = []
            break

    inverse_fcn = inv_composite
    for i in tdata_ls:
        if not callable(i["inverse_fcn"]):
            inverse_fcn = []
            break

    if forward_fcn == [] and inverse_fcn==[]:
        error('[Error]:maketform:invalidForwardOrInverseFunction')

    t = assigntform(tdata_ls[N-1]["ndims_in"], tdata_ls[0]["ndims_out"],
                    forward_fcn, inverse_fcn, tdata_ls)
    
    return t

#---------------------------------------------------------------
def fwd_composite( U, t ):

    # FORWARD composite transformation
    #
    # U is the row vector to be transformed, or
    # a matrix with a vector in each row.

    N = len(t['tdata'])
    Xc = U
    for i in range(N-1,-1,-1):
        Xn = t["tdata"][i]["forward_fcn"](Xc, t["tdata"][i])
        Xc = Xn

    return Xn

#---------------------------------------------------------------
def inv_composite( X, t ):
    # INVERSE composite transformation
    #
    # X is the row vector to be transformed, or
    # a matrix with a vector in each row.

    N = len(t['tdata'])
    Uc = X
    for i in range(N):
        Un = t["tdata"][i]["inverse_fcn"](Uc, t["tdata"][i])
        Uc = Un
    return Un

def custom(args):
    ndims_in, ndims_out, forward_fcn, inverse_fcn, tdata = args
    # Validate sizes and types
    if not isinstance(ndims_in,int):
        error('[Error]:maketform:invalidNDims_In')

    if not isinstance(ndims_out,int):
        error('[Error]:maketform:invalidNDims_Out')

    if ndims_in < 1:
        error('[Error]:maketform:nDimsInIsNotPositive')

    if ndims_out < 1:
        error('[Error]:maketform:nDimsOutIsNotPositive')

    if forward_fcn=="" or (not callable(forward_fcn)):
        error('[Error]:maketform:invalidForwardFcn')

    if inverse_fcn=="" or (not callable(inverse_fcn)):
        error('[Error]:maketform:invalidInverseFcn')

    t = assigntform(ndims_in, ndims_out, forward_fcn, inverse_fcn, tdata)

    return t


def MatchFeducials(image1spots,image2spots,**kwargs):
    # Compute translation/rotation warp that best aligns the points in image1spots
    # and image2spots by maximizing the alignment of the two points that show the
    # most mutually consistent x,y translation.
    #
    #
    # corrPrecision - precision at which to compute the correlation based
    # alignment.  default is 1, no upsampling, which is more robust
    # smaller numbers (e.g. 0.1) will give a finer (subpixel) alignment.
    # maxD - maximum distance after correlation based alignment that objects
    # can be separated.
    # maxTrueSeparation -- maximum distance allowed between matched points

    troubleshoot = False

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['maxD']= 5
    parameters['maxTrueSeparation']=np.inf
    parameters['corrPrecision']=1
    parameters['useCorrAlign']= True
    parameters['fighandle']= []
    parameters['imageSize']= [256, 256]
    parameters['showPlots']= True
    parameters['showCorrPlots']= True
    parameters['verbose']=True

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    if "troubleshoot" in parameters:
        troubleshoot = parameters["troubleshoot"]

    #--------------------------------------------------------------------------
    ## Main Function
    #--------------------------------------------------------------------------

    h = parameters["imageSize"][0]
    w = parameters["imageSize"][1]

    # -------------Step 1: Match by cross correlation -----------------------
    # (This step is optional)
    stp = parameters["corrPrecision"]
    edges1 = [np.arange(0, h + 0.2*stp, stp), np.arange(0, w + 0.2*stp, stp)] ## to include the endpoints:start, stop+0.5*step
    edges2 = [np.arange(0, h + 0.2*stp, stp), np.arange(0, w + 0.2*stp, stp)]
    centers1 = [[e_i - stp*0.5 for e_i in edges1[0]] + [edges1[0][-1]+0.5*stp], [e_i - stp for e_i in edges1[1]] + [edges1[1][-1]+0.5*stp]]
    centers2 = [[e_i - stp*0.5 for e_i in edges2[0]] + [edges2[0][-1]+0.5*stp], [e_i - stp for e_i in edges2[1]] + [edges2[1][-1]+0.5*stp]]
    I1,xE1,yE1 = np.histogram2d(image1spots[1],image1spots[0],bins=edges1)
    I2,xE2,yE2 = np.histogram2d(image2spots[1],image2spots[0],bins=edges2)
    if parameters["useCorrAlign"]:
        [xshift,yshift] = CorrAlign(I1,I2,**parameters)
        xshift = xshift*stp
        yshift = yshift*stp
    else:
        xshift = 0.0
        yshift = 0.0

    # Enforce maximum

    # # figure for troubleshooting correlation alignment
    if troubleshoot:
        fig_x = plt.figure(10,figsize=(8,4))
        ax = fig_x.add_subplot(1,2,1)
        imgX,cmap = Ncolor(np.dstack((I1,I2)))
        ax.imshow((imgX * 255).astype(np.uint8), extent=[0, imgX.shape[0], imgX.shape[1], 0], cmap=cmap)
        ax.plot(image1spots[0]/stp,image1spots[1]/stp,'wo',fillstyle='none',markersize=4,markeredgewidth=0.5)
        ax.plot(image2spots[0]/stp,image2spots[1]/stp,'bo',fillstyle='none',markersize=4,markeredgewidth=0.5)

        ax = fig_x.add_subplot(1,2,2)
        imgX,cmap = Ncolor(np.dstack((I1,I2)))
        ax.imshow((imgX * 255).astype(np.uint8), extent=[0, imgX.shape[0], imgX.shape[1], 0], cmap=cmap)
        ax.plot(image1spots[0]/stp,image1spots[1]/stp,'wo',fillstyle='none',markersize=4,markeredgewidth=0.5)
        ax.plot(image1spots[0]/stp+xshift,image1spots[1]/stp+yshift,'bo',fillstyle='none',markersize=4,markeredgewidth=0.5)

        if not os.path.exists(os.path.join(parameters["savePath"],"savedImages")):
            os.makedirs(os.path.join(parameters["savePath"],"savedImages"), exist_ok=True)
        fig_x_name = os.path.join(parameters["savePath"],
                                  "savedImages/MatchFeducials_Troubleshoot_c"+str(parameters["cellIDs"])+
                                  "_f"+str(parameters["fiducialNum"])+"."+parameters["figFormats"])
        fig_x.savefig(fig_x_name)
        plt.close(fig_x)


    if parameters["verbose"]:
        print('  xshift=',xshift,' yshift=',yshift)

    #-------------- Step 2: Match by warp to nearest neigbhor --------------#
    numFeducials = image2spots.shape[1]
    image2spotsw = image2spots + np.tile([[xshift],[yshift]],(1,numFeducials))
    # Match unique nearest neighbors

    if image1spots.shape[1] >= image2spots.shape[1]:
        [idx1,dist1] = knnsearch2d(image1spots.T,image2spotsw.T, 1) #  indices of image1spots nearest for each point in image2spots
        matches21 = np.array([list(range(image2spots.shape[1])),idx1[0]]).T
        matches21 = matches21[dist1[0] <= parameters["maxD"], :]    # remove distant points

        # for the channel with the smaller number of feducials, remove double hits
        [v,n] = np.unique(idx1[0],return_counts=True)
        multihits1 = v[n>1]
        multihits1_idx = np.logical_not(np.isin(matches21[:,1], multihits1))
        matches21 = matches21[multihits1_idx,:]

        matched1 = matches21[:,1]
        matched2 = matches21[:,0]
    else:
        [idx2, dist2] = knnsearch2d(image2spotsw.T, image1spots.T, 1)  # indices of image1spots nearest for each point in image2spots
        matches12 = np.array([list(range(image1spots.shape[1])), idx2[0]]).T
        matches12 = matches12[dist2[0] <= parameters["maxD"], :]  # remove distant points

        # for the channel with the smaller number of feducials, remove double hits
        [v, n] = np.unique(idx2[0], return_counts=True)
        multihits1 = v[n > 1]
        multihits1_idx = np.logical_not(np.isin(matches12[:, 1], multihits1))
        matches12 = matches12[multihits1_idx, :]


        matched1 = matches12[:,0]
        matched2 = matches12[:,1]

    [idx1, dist1] = knnsearch2d(image1spots.T, image2spotsw.T,1)  # indices of image1spots nearest for each point in image2spots
    matches21 = np.array([list(range(image2spots.shape[1])), idx1[0]]).T
    matches21 = matches21[dist1[0] <= parameters["maxD"], :]  # remove distant points

    # for the channel with the smaller number of feducials, remove double hits
    [v, n] = np.unique(idx1[0], return_counts=True)
    multihits1 = v[n > 1]
    multihits1_idx = np.logical_not(np.isin(matches21[:, 1], multihits1))
    matches21 = matches21[multihits1_idx, :]

    matched1 = matches21[:, 1]
    matched2 = matches21[:, 0]

    #----------------- Plotting ---------
    if parameters["showPlots"]:
        if isinstance(parameters["fighandle"],matplotlib.figure.Figure):
            fig_x = parameters["fighandle"]
        elif parameters["fighandle"] == "" or len(parameters["fighandle"]) == 0:
            fig_x = plt.figure(10, figsize=(6, 6))
            parameters["fighandle"] = fig_x
        else:
            error('[Error]: parameters["fighandle"] setting is wrong.'
                  'parameters["fighandle"] can only be an empty string(''), empty liat([]) or an instance of matplotlib.figure.Figure')
        ax = fig_x.add_subplot()
        ax.plot(image1spots[0],image1spots[1],'k.', markersize = 1)
        ax.plot(image2spots[0],image2spots[1],'bo',fillstyle='none',markersize=4,markeredgewidth=0.5)

        for i in range(image1spots.shape[1]):
            ax.text(image1spots[0,i]+2,image1spots[1,i],str(i+1),size=6)

        for i in range(image2spots.shape[1]):
            ax.text(image2spots[0,i]+2,image2spots[1,i],str(i+1),color='b',size=6)

        for i in range(len(matched1)):
            ax.plot([image1spots[0,matched1[i]],image2spots[0,matched2[i]]],
                     [image1spots[1,matched1[i]],image2spots[1,matched2[i]]],
                     'bo',fillstyle='none',markersize=4,markeredgewidth=0.5)

        if not os.path.exists(os.path.join(parameters["savePath"],"savedImages")):
            os.makedirs(os.path.join(parameters["savePath"],"savedImages"), exist_ok=True)
        fig_x_name = os.path.join(parameters["savePath"],
                                  "savedImages/MatchFeducials_Matched_c"+str(parameters["cellIDs"])+
                                      "_f"+str(parameters["fiducialNum"])+"."+parameters["figFormats"])
        fig_x.savefig(fig_x_name)
        plt.close(fig_x)

    return matched1, matched2, parameters
#-------------------------------------

def CorrAlign(Im1,Im2,**kwargs):
    #  [xshift,yshift,parameters] = CorrAlign(Im1,Im2)
    # Inputs
    # {'region', 'nonnegative', 200} max number of pixels to use
    # {'showplot', 'boolean', false} show image of before and after
    # {'upsample', 'positive', 1} upsample to get subpixel alignment
    #
    # Compute xshift and yshift to align two images based on maximizing
    # cross-correlation.

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['region'] = 200
    parameters['showplot'] = False
    parameters['upsample'] = 1
    parameters['subregion'] = True

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    if "showCorrPlots" in parameters:
        parameters['showplot'] = parameters['showCorrPlots']

    H,W = Im1.shape

    if parameters["subregion"] and parameters["region"] < H:
        hs = np.arange(np.round(H * 0.5)-parameters["region"]*0.5,np.round(H*0.5)+parameters["region"]*0.5)
        ws = np.arange(np.round(W * 0.5)-parameters["region"]*0.5,np.round(W*0.5)+parameters["region"]*0.5)
        Im1 = Im1[hs.astype(np.int),:][:,ws.astype(np.int)]
        Im2 = Im2[hs.astype(np.int),:][:,ws.astype(np.int)]
    if parameters["upsample"] != 1:
        width1 = int(Im1.shape[1] * parameters["upsample"])
        height1 = int(Im1.shape[0] * parameters["upsample"])
        dim1 = (width1, height1)
        Im1 = cv2.resize(Im1,dim1,interpolation = cv2.INTER_CUBIC)

        width2 = int(Im2.shape[1] * parameters["upsample"])
        height2 = int(Im2.shape[0] * parameters["upsample"])
        dim2 = (width2, height2)
        Im2 = cv2.resize(Im2,dim2,interpolation = cv2.INTER_CUBIC)


    H,W = Im1.shape
    corrM = xcorr2_bkp(Im1, Im2)  # The correlation map
    Hc = min(H,parameters["region"]*parameters["upsample"])
    Wc = min(W,parameters["region"]*parameters["upsample"])
    Hc2 = round(Hc*0.5)
    Wc2 = round(Wc*0.5)
    # Just the center of the correlation map
    corrMmini = corrM[H-Hc2:H+Hc2,W-Wc2:W+Wc2]
    parameters["corrPeak"]=  np.max(corrMmini[:])
    indmax=np.argmax(corrMmini)
    [cy,cx] = ind2sub([Hc,Wc],indmax)
    xshift = (cx-Wc2+1)
    yshift = (cy-Hc2+1)

    if parameters["showplot"]:
        fig_x = plt.figure(10,figsize=(12, 4))
        ax = fig_x.add_subplot(1,3,1)
        imgX,cmap = Ncolor(np.dstack((Im1,Im2)))
        ax.imshow((imgX * 255).astype(np.uint8), extent=[0, imgX.shape[0], imgX.shape[1],0],cmap=cmap)

        M = np.float32([[1, 0, xshift], [0, 1, yshift]])
        Im2 = cv2.warpAffine(Im2, M, (Im2.shape[1], Im2.shape[0]))
        ax = fig_x.add_subplot(1,3,2)
        imgX,cmap = Ncolor(np.dstack((Im1,Im2)))
        ax.imshow((imgX * 255).astype(np.uint8), extent=[0, imgX.shape[0], imgX.shape[1],0],cmap=cmap)

        ax = fig_x.add_subplot(1,3,3)
        ax.imshow(corrMmini, extent=[0, corrMmini.shape[0], corrMmini.shape[1],0],cmap = cm.get_cmap("jet",256))

        if not os.path.exists(os.path.join(parameters["savePath"],"savedImages")):
            os.makedirs(os.path.join(parameters["savePath"],"savedImages"), exist_ok=True)
        fig_x_name = os.path.join(parameters["savePath"],
                                  "savedImages/CorrAlign_c"+str(parameters["cellIDs"])+
                                  "_f"+str(parameters["fiducialNum"])+"."+parameters["figFormats"])
        fig_x.savefig(fig_x_name)

        plt.close(fig_x)

    xshift = xshift/parameters["upsample"]
    yshift = yshift/parameters["upsample"]

    return [xshift,yshift]


def fliptform( t ):
    #FLIPTFORM Flip input and output roles of TFORM structure.
    #   TFLIP = FLIPTFORM(T) creates a new spatial transformation structure (a
    #   "TFORM struct") by flipping the roles of the inputs and outputs in an
    #   existing TFORM struct.
    #
    #   Example
    #   -------
    #       T = maketform('affine',[.5 0 0 .5 2 0 0 0 1])
    #       T2 = fliptform(T)
    #
    #   The following are equivalent:
    #       x = tformfwd([-3 7],T)
    #       x = tforminv([-3 7],T2)
    #
    #   See also MAKETFORM, TFORMFWD, TFORMINV.

    #   Copyright 1993-2015 The MathWorks, Inc.

    if not isinstance(t,dict):
        error('[Error]" images:fliptform:tMustBeSingleTformStruct')

    tflip = maketform('custom', t["ndims_out"], t["ndims_in"], t["inverse_fcn"],t["forward_fcn"], t["tdata"])

    return tflip


def imtransform(A, tform, **kwargs):
    '''
    #IMTRANSFORM Apply 2-D spatial transformation to image.
    #   IMTRANSFORM is not recommended. Use IMWARP instead.
    #
    #   B = IMTRANSFORM(A,TFORM) transforms the image A according to the 2-D
    #   spatial transformation defined by TFORM, which is a tform structure
    #   as returned by MAKETFORM or CP2TFORM.  If ndims(A) > 2, such as for
    #   an RGB image, then the same 2-D transformation is automatically
    #   applied to all 2-D planes along the higher dimensions.
    #
    #   When you use this syntax, IMTRANSFORM automatically shifts the origin of
    #   your output image to make as much of the transformed image visible as
    #   possible. If you are using IMTRANSFORM to do image registration, this syntax
    #   is not likely to give you the results you expect you may want to set
    #   'XData' and 'YData' explicitly. See the description below of 'XData' and
    #   'YData' as well as Example 3.
    #
    #   B = IMTRANSFORM(A,TFORM,INTERP) specifies the form of interpolation to
    #   use.  INTERP can be one of the strings 'nearest', 'bilinear', or
    #   'bicubic'.  Alternatively INTERP can be a RESAMPLER struct as returned
    #   by MAKERESAMPLER.  This option allows more control over how resampling
    #   is performed.  The default value for INTERP is 'bilinear'.
    #
    #   [B,XDATA,YDATA] = IMTRANSFORM(...) returns the location of the output
    #   image B in the output X-Y space.  XDATA and YDATA are two-element
    #   vectors.  The elements of XDATA specify the x-coordinates of the first
    #   and last columns of B.  The elements of YDATA specify the y-coordinates
    #   of the first and last rows of B.  Normally, IMTRANSFORM computes XDATA
    #   and YDATA automatically so that B contains the entire transformed image
    #   A.  However, you can override this automatic computation see below.
    #
    #   [B,XDATA,YDATA] = IMTRANSFORM(...,PARAM1,VAL1,PARAM2,VAL2,...)
    #   specifies parameters that control various aspects of the spatial
    #   transformation. Parameter names can be abbreviated, and case does not
    #   matter.
    #
    #   Parameters include:
    #
    #   'UData'      Two-element real vector.
    #   'VData'      Two-element real vector.
    #                'UData' and 'VData' specify the spatial location of the
    #                image A in the 2-D input space U-V.  The two elements of
    #                'UData' give the u-coordinates (horizontal) of the first
    #                and last columns of A, respectively.  The two elements
    #                of 'VData' give the v-coordinates (vertical) of the
    #                first and last rows of A, respectively.
    #
    #                The default values for 'UData' and 'VData' are [1
    #                size(A,2)] and [1 size(A,1)], respectively.
    #
    #   'XData'      Two-element real vector.
    #   'YData'      Two-element real vector.
    #                'XData' and 'YData' specify the spatial location of the
    #                output image B in the 2-D output space X-Y.  The two
    #                elements of 'XData' give the x-coordinates (horizontal)
    #                of the first and last columns of B, respectively.  The
    #                two elements of 'YData' give the y-coordinates
    #                (vertical) of the first and last rows of B,
    #                respectively.
    #
    #                If 'XData' and 'YData' are not specified, then
    #                IMTRANSFORM estimates values for them that will
    #                completely contain the entire transformed output image.
    #
    #   'XYScale'    A one- or two-element real vector.
    #                The first element of 'XYScale' specifies the width of
    #                each output pixel in X-Y space.  The second element (if
    #                present) specifies the height of each output pixel.  If
    #                'XYScale' has only one element, then the same value is
    #                used for both width and height.
    #
    #                If 'XYScale' is not specified but 'Size' is, then
    #                'XYScale' is computed from 'Size', 'XData', and 'YData'.
    #                If neither 'XYScale' nor 'Size' is provided, then
    #                the scale of the input pixels is used for 'XYScale'
    #                except when that would result in an excessively large
    #                output image, in which case the 'XYScale' is increased.
    #
    #   'Size'       A two-element vector of nonnegative integers.
    #                'Size' specifies the number of rows and columns of the
    #                output image B.  For higher dimensions, the size of B is
    #                taken directly from the size of A.  In other words,
    #                size(B,k) equals size(A,k) for k > 2.
    #
    #                If 'Size' is not specified, then it is computed from
    #                'XData', 'YData', and 'XYScale'.
    #
    #   'FillValues' An array containing one or several fill values.
    #                Fill values are used for output pixels when the
    #                corresponding transformed location in the input image is
    #                completely outside the input image boundaries.  If A is
    #                2-D then 'FillValues' must be a scalar.  However, if A's
    #                dimension is greater than two, then 'FillValues' can be
    #                an array whose size satisfies the following constraint:
    #                size(fill_values,k) must either equal size(A,k+2) or 1.
    #                For example, if A is a uint8 RGB image that is
    #                200-by-200-by-3, then possibilities for 'FillValues'
    #                include:
    #
    #                    0                 - fill with black
    #                    [000]           - also fill with black
    #                    255               - fill with white
    #                    [255255255]     - also fill with white
    #                    [00255]         - fill with blue
    #                    [2552550]       - fill with yellow
    #
    #                If A is 4-D with size 200-by-200-by-3-by-10, then
    #                'FillValues' can be a scalar, 1-by-10, 3-by-1, or
    #                3-by-10.
    #
    #   Notes
    #   -----
    #   - When you do not specify the output-space location for B using
    #     'XData' and 'YData', IMTRANSFORM estimates them automatically using
    #     the function FINDBOUNDS.  For some commonly-used transformations,
    #     such as affine or projective, for which a forward-mapping is easily
    #     computable, FINDBOUNDS is fast.  For transformations that do not
    #     have a forward mapping, such as the polynomial ones computed by
    #     CP2TFORM, FINDBOUNDS can take significantly longer.  If you can
    #     specify 'XData' and 'YData' directly for such transformations,
    #     IMTRANSFORM may run noticeably faster.
    #
    #   - The automatic estimate of 'XData' and 'YData' using FINDBOUNDS is
    #     not guaranteed in all cases to completely contain all the pixels of
    #     the transformed input image.
    #
    #   - The output values XDATA and YDATA may not exactly equal the input
    #     'XData and 'YData' parameters.  This can happen either because of
    #     the need for an integer number or rows and columns, or if you
    #     specify values for 'XData', 'YData', 'XYScale', and 'Size' that
    #     are not entirely consistent.  In either case, the first element of
    #     XDATA and YDATA always equals the first element of 'XData' and
    #     'YData', respectively.  Only the second elements of XDATA and YDATA
    #     might be different.
    #
    #   - IMTRANSFORM assumes spatial-coordinate conventions for the
    #     transformation TFORM.  Specifically, the first dimension of the
    #     transformation is the horizontal or x-coordinate, and the second
    #     dimension is the vertical or y-coordinate.  Note that this is the
    #     reverse of MATLAB's array subscripting convention.
    #
    #   - TFORM must be a 2-D transformation to be used with IMTRANSFORM.
    #     For arbitrary-dimensional array transformations, see TFORMARRAY.
    #
    #   Class Support
    #   -------------
    #   A can be of any nonsparse numeric class, real or complex.  It can also be
    #   logical.  The class of B is the same as the class of A.
    #
    #   Example 1
    #   ---------
    #   Apply a horizontal shear to an intensity image.
    #
    #       I = imread('cameraman.tif')
    #       tform = maketform('affine',[1 0 0 .5 1 0 0 0 1])
    #       J = imtransform(I,tform)
    #       figure, imshow(I), figure, imshow(J)
    #
    #   Example 2
    #   ---------
    #   A projective transformation can map a square to a quadrilateral.  In
    #   this example, set up an input coordinate system so that the input
    #   image fills the unit square and then transform the image from the
    #   quadrilateral with vertices (0 0), (1 0), (1 1), (0 1) to the
    #   quadrilateral with vertices (-4 2), (-8 -3), (-3 -5), and (6 3).  Fill
    #   with gray and use bicubic interpolation.  Make the output size the
    #   same as the input size.
    #
    #       I = imread('cameraman.tif')
    #       udata = [0 1]  vdata = [0 1]  # input coordinate system
    #       tform = maketform('projective',[ 0 0  1  0  1  1 0 1],...
    #                                      [-4 2 -8 -3 -3 -5 6 3])
    #       [B,xdata,ydata] = imtransform(I,tform,'bicubic','udata',udata,...
    #                                                       'vdata',vdata,...
    #                                                       'size',size(I),...
    #                                                       'fill',128)
    #       imshow(I,'XData',udata,'YData',vdata), axis on
    #       figure, imshow(B,'XData',xdata,'YData',ydata), axis on
    #
    #   Example 3
    #   ---------
    #   Register an aerial photo to an orthophoto.
    #
    #       unregistered = imread('westconcordaerial.png')
    #       figure, imshow(unregistered)
    #       figure, imshow('westconcordorthophoto.png')
    #       load westconcordpoints # load some points that were already picked
    #       t_concord = cp2tform(movingPoints,fixedPoints,'projective')
    #       info = imfinfo('westconcordorthophoto.png')
    #       registered = imtransform(unregistered,t_concord,...
    #                                'XData',[1 info.Width], 'YData',[1 info.Height])
    #       figure, imshow(registered)
    #
    #   See also CHECKERBOARD, CP2TFORM, IMRESIZE, IMROTATE, IMWARP, MAKETFORM, MAKERESAMPLER, TFORMARRAY.

    #   Copyright 1993-2017 The MathWorks, Inc.

    # Input argument details
    # ----------------------
    # A              numeric nonsparse array, any dimension, real or complex
    #                may be logical
    #                may be empty
    #                NaN's and Inf's OK
    #                required
    #
    # TFORM          valid TFORM struct as returned by MAKETFORM
    #                checked using private/istform
    #                required
    #
    # INTERP         one of these strings: 'nearest', 'linear', 'cubic'
    #                case-insensitive match
    #                nonambiguous abbreviations allowed
    #
    #                OR a resampler structure as returned by makeresampler
    #                checked using private/isresample
    #
    #                optional defaults to 'linear'
    #
    # 'FillValues'   double real matrix
    #                Inf's and NaN's allowed
    #                may be []
    #                optional defaults to 0
    #
    # 'UData'        2-element real double vector
    #                No Inf's or NaN's
    #                The two elements must be different unless A has only
    #                    one column
    #                optional defaults to [1 size(A,2)]
    #
    # 'VData'        2-element real double vector
    #                No Inf's or NaN's
    #                The two elements must be different unless A has only
    #                    one row
    #                optional defaults to [1 size(A,1)]
    #
    # 'XData'        2-element real double vector
    #                No Inf's or NaN's
    #                optional if not provided, computed using findbounds
    #
    # 'YData'        2-element real double vector
    #                No Inf's or NaN's
    #                optional if not provided, computed using findbounds
    #
    # 'XYScale'      1-by-2 real double vector
    #                elements must be positive
    #                optional default is the horizontal and vertical
    #                  scale of the input pixels.
    #
    # 'Size'         real double row-vector
    #                elements must be nonnegative integers
    #                Can be 1-by-2 or 1-by-numdims(A).  If it is
    #                  1-by-numdims(A), sizeB(3:end) must equal
    #                  sizeA(3:end).
    #                optional default computation:
    #                  num_rows = ceil(abs(ydata(2) - ydata(1)) ./ xyscale(2)) + 1
    #                  num_cols = ceil(abs(xdata(2) - xdata(1)) ./ xyscale(1)) + 1
    #
    # If Size is provided and XYScale is not provided, then the output xdata
    # and ydata will be the same as the input XData and YData.  Otherwise,
    # the output xdata and ydata must be modified to account for the fact
    # that the output size is constrained to contain only integer values.
    # The first elements xdata and ydata is left alone, but the second
    # values may be altered to make them consistent with Size and XYScale.
    '''

    args = {}
    args["resampler"] = []
    args["fill_values"] = 0
    args["udata"] = []
    args["vdata"] = []
    args["xdata"] = []
    args["ydata"] = []
    args["size"] = []
    args["xyscale"] = []
    args["pure_translation"] = False

    args["A"] = A
    args["tform"] = tform

    if "interp" in kwargs:
        if kwargs["interp"] not in ['nearest', 'linear', 'cubic']:
            error("[Error]: The value of <interp> should be one in ['nearest', 'linear', 'cubic']")
        else:
            args["resampler"] = makeresampler(kwargs['interp'], 'fill')
    else:
        args["resampler"] = makeresampler('linear', 'fill')

    for k_i in kwargs:
        if k_i == "interp":
            pass
        elif k_i == "fillvalues":
            args[k_i] = kwargs[k_i]
        elif k_i == "udata":
            args[k_i] = kwargs[k_i]
        elif k_i == "vdata":
            args[k_i] = kwargs[k_i]
        elif k_i == "xdata":
            args[k_i] = kwargs[k_i]
        elif k_i == "ydata":
            args[k_i] = kwargs[k_i]
        elif k_i == "size":
            args[k_i] = kwargs[k_i]
        elif k_i == "xyscale":
            args[k_i] = kwargs[k_i]
        else:
            error('[Error]:imtransform:internalError - unknown key:', k_i)

    if len(args["udata"])==0:
        args["udata"] = [0,args["A"].shape[1]]
    if len(args["vdata"])==0:
        args["vdata"] = [0,args["A"].shape[0]]
    if len(args["xdata"])==0 or len(args["ydata"]) == 0:
        error('[Error]:imtransform:missingXData / YData')

    if len(args["xdata"])==0:
        # Output bounds not provided - estimate them.
        input_bounds = [[args["udata"][0], args["vdata"][0]],
                        [args["udata"][1], args["vdata"][1]]]
        try:
            output_bounds = findbounds(args["tform"], input_bounds)
            args["xdata"] = [output_bounds[0,0], output_bounds(1,0)]
            args["ydata"] = [output_bounds[0,1], output_bounds(1,1)]
        except Exception as e:
            error('[Error]: imtransform:unestimableOutputBounds')

        # Need to check for pure translation here on the tform that comes in as input
        # before it gets wrapped in a composite tform in the main function.
        if pureTranslation(args["tform"]):
            args["pure_translation"] = True
    if args["size"] != []:
        # Output size was provided.
        if args["xyscale"] !=[]:
            # xy_scale was provided recompute bounds.
            args["xdata"],args["ydata"] = recompute_output_bounds(**args)
        else:
            # Do nothing.  Scale was not provided but it is not needed.
            pass

    else:
        # Output size was not provided.
        if args["xyscale"] ==[] or args["xyscale"]=="":
            # Output scale was not provided.  Use the scale of the input pixels.
            args["xyscale"] = compute_xyscale(**args)

        # Compute output size.
        num_rows = np.ceil(np.abs(args["ydata"][1] - args["ydata"][0]) / args["xyscale"][1])
        num_cols = np.ceil(np.abs(args["xdata"][1] - args["xdata"][0]) / args["xyscale"][0])
        args["size"] = [num_rows, num_cols]

        args["xdata"],args["ydata"] = recompute_output_bounds(**args)

    args["tform"] = make_composite_tform(**args)

    # imtransform uses x-y convention for ordering dimensions.
    tdims_a = [1,0]
    tdims_b = [1,0]

    tsize_b = [int(args["size"][1]),int(args["size"][0])]
    tmap_b = []

    B = tformarray(args["A"], args["tform"], args["resampler"],
                   tdims_a, tdims_b, tsize_b, tmap_b,args["fill_values"])

    xdata = args["xdata"]
    ydata = args["ydata"]

    if args["pure_translation"]:
        print('[Warning]:imtransform:warnForPureTranslation')

    return [B,xdata,ydata]

def findbounds(tform, inbounds):

    if tform["ndims_in"] != tform["ndims_out"]:
        error('[Error]:findbounds:inOutDimsOfTformMustBeSame')

    if inbounds.shape[0] !=2:
        error('[Error]:findbounds:inboundsMustBe2byN')

    num_dims = inbounds.shape[1]
    if num_dims != tform["ndims_in"]:
        error('[Error]:findbounds:secondDimOfInbundsMustEqualTformNdimsIn')


    if tform["forward_fcn"]=="":
        outbounds = find_bounds_using_search(tform, inbounds)
    else:
        outbounds = find_bounds_using_forward_fcn(tform, inbounds)

    return outbounds


def find_bounds_using_search(tform, in_bounds):
    if tform["inverse_fcn"] == "":
        error('[Error]:findbounds:fwdAndInvFieldsCantBothBeEmpty')

    in_vertices = bounds_to_vertices(in_bounds)
    in_points = add_in_between_points(in_vertices)
    out_points = np.zeros(in_points.shapa)
    success = 1
    for k in range(in_points.shape[0]):
        [x,fval,exitflag] = scipy.optimize.fmin(objective_function, in_points[k,:], args=(tform, in_points[k,:]))
        if exitflag <= 0:
            success = 0
            break
        else:
            out_points[k,:] = x

    if success:
        out_bounds = points_to_bounds(out_points)
    else:
        # Optimization failed the fallback strategy is to make the output
        # bounds the same as the input bounds.  However, if the input
        # transform dimensionality is not the same as the output transform
        # dimensionality, there doesn't seem to be anything reasonable to
        # do.
        if tform["ndims_in"] == tform["ndims_out"]:
            print('[Warning]: images:findbounds:searchFailed')
            out_bounds = in_bounds
        else:
             error('[Error]:findbounds:mixedDimensionalityTFORM')
    return out_bounds

def objective_function(x, tform, u0):
    # This is the function to be minimized by FMINSEARCH.
    s = np.norm(u0 - tforminv(x, tform))
    return s

def bounds_to_vertices(bounds):
    num_dims = bounds.shape[1]
    num_vertices = 2**num_dims

    binary = np.repmat('0',num_vertices,num_dims)
    for k in range(num_vertices):
        binary[k,:] = de2bi(k-1,num_dims)

    mask = binary != '0'

    low = np.repmat(bounds[1,:],num_vertices,1)
    high = np.repmat(bounds[2,:],num_vertices, 1)
    vertices = low
    vertices[mask] = high[mask]

    return vertices

def add_in_between_points(vertices):
    num_vertices = vertices.shape[0]
    ndx = nchoosek(list(range(num_vertices)), 2)
    new_points = (vertices[ndx[:, 0],:] + vertices[ndx[:, 2],:]) / 2
    new_points = np.unique(new_points, axis=1)

    points = [vertices, new_points]

    return points

def points_to_bounds(points):
    bounds = [np.min(points, [], 1),  np.max(points, [], 1)]
    return bounds

def find_bounds_using_forward_fcn(tform, in_bounds):
    in_vertices = bounds_to_vertices(in_bounds)
    in_points = add_in_between_points(in_vertices)
    out_points = tformfwd(in_points, tform)
    out_bounds = points_to_bounds(out_points)

def recompute_output_bounds(**args):
    out_size = np.maximum(args["size"],[1,1])

    xdata = args["xdata"]
    ydata = args["ydata"]

    xdata[1] = args["xdata"][0] + out_size[1] * args["xyscale"][0] * np.sign(args["xdata"][1] - args["xdata"][0])
    ydata[1] = args["ydata"][0] + out_size[0] * args["xyscale"][1] * np.sign(args["ydata"][1] - args["ydata"][0])

    return xdata,ydata

#--------------------------------------------------
def compute_xyscale(**args):
    size_A = args["A"].shape
    xscale = compute_scale(args["udata"], size_A[1])
    yscale = compute_scale(args["vdata"], size_A[0])

    # If the output size would otherwise be twice the input size
    # (in both dimensions), then multiply the output scale by a
    # factor greater than 1.  This makes the output pixels larger
    # (as measured in the output transform coordinates) and limits
    # the size of the output image.

    minscale = np.min([np.abs(args["xdata"][1] - args["xdata"][0]) / (2*size_A[1]),
                      np.abs(args["ydata"][1] - args["ydata"][0]) / (2*size_A[0])])

    if xscale > minscale and yscale > minscale:
      xyscale = [xscale,yscale]
    else:
      xyscale = np.array([xscale,yscale]) * minscale / max(xscale,yscale)
      print('[Error]:imtransform:warnForAutomaticScaleChange')

    return xyscale

#--------------------------------------------------
def compute_scale(udata, N):
    scale_numerator = udata[1] - udata[0]
    scale_denominator = max(N - 1, 0)
    if scale_denominator == 0:
        if scale_numerator == 0:
            scale = 1
        else:
            error('[Error]:imtransform:unclearSpatialLocation')
    else:
        scale = scale_numerator / scale_denominator
    return scale

#--------------------------------------------------
def make_composite_tform(**args):
    reg_b = maketform('box', args["size"][::-1],
                      [args["xdata"][0], args["ydata"][0]],
                      [args["xdata"][1], args["ydata"][1]])

    in_size = list(args["A"].shape)
    in_size = in_size[0:2]

    reg_a = maketform('box',in_size[::-1],
                      [args["udata"][0], args["vdata"][0]],
                      [args["udata"][1], args["vdata"][1]])

    new_tform = maketform('composite', fliptform(reg_b), args["tform"], reg_a)
    return new_tform
    

f_affine_fwd = ""
f_affine_inv = ""
f_proj_fwd = ""
f_proj_inv= ""
def pureTranslation(tform):
    global f_affine_fwd
    global f_affine_inv
    global f_proj_fwd
    global f_proj_inv
    if f_affine_fwd=="":
        # If we haven't found them yet, get function names for reference affine and
        # projective transformations and cache for subsequent use.
        affine_tform = maketform('affine',np.eye(3))
        projective_tform = maketform('projective',np.eye(3))

        f_affine_fwd = getFunctionName(affine_tform["forward_fcn"])
        f_affine_inv = getFunctionName(affine_tform["inverse_fcn"])
        f_proj_fwd = getFunctionName(projective_tform["forward_fcn"])
        f_proj_inv = getFunctionName(projective_tform["inverse_fcn"])

    f_tform_fwd = getFunctionName(tform["forward_fcn"])
    f_tform_inv = getFunctionName(tform["inverse_fcn"])

    is_affine = (f_affine_fwd==f_tform_fwd) and (f_affine_inv==f_tform_inv)
    is_projective = (f_proj_fwd==f_tform_fwd) and (f_proj_inv==f_tform_inv)

    TF = (is_affine or is_projective) and \
         np.equal(tform["tdata"]["T"][0:1,0:1],np.eye(2)) and \
         np.not_equal(tform["tdata"]["T"][2,0:1],[0,0])

    return TF

def getFunctionName(funhandle):
    if callable(funhandle):
        f = funhandle.__name__
    else:
        f = ''
    return f

def tformarray( A, T, R, tdims_A, tdims_B, tsize_B, tmap_B, F ):
    # Construct a new tsize_B if tmap_B is non-empty.
    tsize_B = CheckDimsAndSizes( tdims_A, tdims_B, tsize_B, tmap_B, T )

    # Get the 'full sizes' of A and B and their non-transform sizes (osize).
    [fsize_A, fsize_B, osize] = fullsizes(A.shape, tsize_B, tdims_A, tdims_B)

    # Finish checking the inputs.
    if R["ndims"] !=np.inf and R["ndims"] != len(tdims_A):
        error('images:tformarray:resamplerDimsMismatch')

    F = CheckFillArray( F, osize )

    # Determine blocking, if any.
    if tmap_B!=[]:
        nBlocks = 1  # Must process in a single block if tmap_B is supplied.
    else:
        blockingfactors = GetBlocking(tsize_B,max(len(tdims_A),len(tdims_B)))
        nBlocks = np.prod(blockingfactors,axis=0)

    # If there is no tmap_B, process large arrays in multiple blocks to conserve
    # the memory required by the output grid G and its mapping to the input
    # space M.  Otherwise, do the resampling in one large block.
    if nBlocks == 1:
        # If not already supplied in tmap_B, construct a grid G mapping B's
        # transform subscripts to T's output space or (if T = []) directly to
        # A's transform subscripts.
        if tmap_B!=[]:
            G = tmap_B
        else:
            hi = tsize_B
            lo = np.zeros((len(hi),))
            G = ConstructGrid(lo,hi)

        # If there is a tform, use it to extend the map in G to A's
        # transform subscripts.
        if len(T) > 0:
            M = tforminv(T,G)
        else:
            M = G
        del G # Free memory used by G (which is no longer needed).
        gc.collect()

        # Resample A using the map M and resampler R.
        B = resample(A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R)
    else:
        # Pre-allocate B with size fsize_B and class(B) = class(A).
        B = np.empty()
        B[np.prod(fsize_B,axis=0)] = A[0]
        B = B.reshape(fsize_B)

        # Loop over blocks in output transform space...
        [lo, hi] = GetBlockLimits(tsize_B, blockingfactors)
        for i in range(nBlocks):
            # Construct the geometric map for the current block.
            G = ConstructGrid(lo[i,:],hi[i,:])
            if T!=[]:
                M = tforminv(G,T)
            else:
                M = G

            del G  # Free memory used by G (which is no longer needed).
            gc.collect()

            # Construct size and subscript arrays for the block, then resample.
            [bsize_B, S] = ConstructSubscripts(fsize_B, tdims_B, lo[i,:], hi[i,:])
            B[S[:]] = resample(A, M, tdims_A, tdims_B, fsize_A, bsize_B, F, R)
    return B

def resample( A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R ):
    # Evaluates the resampling function defined in the resampler R at
    # each point in the subscript map M and replicated across all the
    # non-transform dimensions of A, producing a warped version of A in B.

    B = R["resamp_fcn"](A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R)
    return B

def CheckDimsAndSizes( tdims_A, tdims_B, tsize_B, tmap_B, T ):
    
    P = len(tdims_A)
    N = len(tdims_B)

    if isinstance(tmap_B,list) and tmap_B == []:
        tmap_B = np.array([tmap_B])

    if len(np.unique(tdims_A)) != P:
        error('images:tformarray:nonUniqueTDIMS_A')
    
    if len(np.unique(tdims_B)) != N:
        error('images:tformarray:nonUniqueTDIMS_B')
    
    # If tmap_B is supplied, ignore the input value of tsize_B and
    # construct tsize_B from tmap_B instead. Allow for the possibility
    # that the last dimension of tmap_B is a singleton by copying no
    # more than N values from size(tmap_B).
    
    if tmap_B.size==0:
        L = N
    else:
        if tsize_B!=[]:
            print('images:tformarray:ignoringTSIZE_B')

        try:
            L = tmap_B.shape[N]
        except:
            L = 1
        
        if len(T)==0 and L != P:
            error('images:tformarray:invalidSizeTMAP_B')
        
        if np.ndim(tmap_B) != PredictNDimsG(N,L):
            error('images:tformarray:invalidDimsTMAP_B')
        size_G = tmap_B.shape
        tsize_B[0:N] = size_G[0:N]

    # Check T
    if isinstance(T,dict) and len(T)> 0:
        if T["ndims_in"] != P:
            error('images:tformarray:dimsMismatchA')

        if T["ndims_out"] != L:
            error('images:tformarray:dimsMismatchB')
    else:
        error('images:tformarray:Invalid T structure')

    # tsize_B and tdims_B must have the same length.
    if len(tsize_B) != N:
        error('images:tformarray:lengthMismatchB')

    return tsize_B

def PredictNDimsG(N, L):
    if N == 1:
        K = 2
    elif N > 1 and L == 1:
        K = N
    elif N > 1 and L > 1:
        K = N + 1
    else:
        error('images:tformarray:NLMustBePositive')
    return K

def fullsizes(size_A, tsize_B, tdims_A, tdims_B):
    # Constructs row vectors indicating the full sizes of A and B (FSIZE_A and
    # FSIZE_B), including trailing singleton dimensions. Also constructs a row
    # vector (OSIZE) listing (in order) the sizes of the non-transform
    # dimensions of both A and B.
    #
    # There are two ways for trailing singletons to arise: (1) values in
    # TDIMS_A exceed the length of SIZE_A or values in TDIMS_B exceed the
    # length of TSIZE_B plus the number on non-transform dimensions of A and
    # (2) all dimensions of A are transform dimensions (e.g., A is a grayscale
    # image) -- in this case a trailing singleton is added to A and then
    # transferred to B.
    #
    # Example:
    #
    #   [fsize_A, fsize_B] = ...
    #      fullsizes( [7 512 512 512 3 20 ], [200 300], [2 3 4], [1 2] )
    #
    #   returns fsize_A = [  7   512   512   512     3    20]
    #   and     fsize_B = [200   300     7     3    20]
    #   and     osize   = [  7     3    20].

    # Actual dimensionality of input array
    ndims_A = len(size_A)

    # 'Full' dimensionality of input array --
    # Increase ndims(A) as needed, to allow for the largest
    # transform dimensions as specified in tdims_A, then
    # make sure there is at least one non-transform dimension.
    fdims_A = max([ndims_A, max(tdims_A)])
    if fdims_A == len(tdims_A):
        fdims_A = fdims_A + 1

    # 'Full' size of input array --
    # Pad size_A with ones (as needed) to allow for values
    # in tdims_A that are higher than ndims(A):
    fsize_A = np.array(list(size_A)+list(np.ones((fdims_A - ndims_A,),dtype=int)))

    # The non-transform sizes of A and B:
    osize = fsize_A[~np.isin(list(range(fdims_A)),tdims_A)]

    # The minimum ndims of B:
    ndims_B = len(osize) + len(tdims_B)

    # Increase ndims_B as needed, to allow for the largest
    # transform dimensions as specified in tdims_B:
    fdims_B = max(ndims_B, max(tdims_B))

    # The full size of B, including possible padding:
    isT_B = np.isin(list(range(fdims_B)), tdims_B)
    padding = np.ones((fdims_B - (len(osize) + len(tsize_B)),))
    index = np.argsort(tdims_B)
    fsize_B = np.zeros(isT_B.shape)
    fsize_B[isT_B] = np.array(tsize_B)[index]
    fsize_B[~isT_B] = np.hstack((osize,padding))
    fsize_B = fsize_B.astype(int)

    return fsize_A, fsize_B, osize

def CheckFillArray( F, osize ):
    if F==[] or F==0:
        F=0
    else:
        size_F = F.shape()
        last = max([1] + list(np.argwhere(size_F != 1)))
        size_F = size_F[0:last]

        # SIZE_F can't be longer than OSIZE.
        N = len(osize)
        q = len(size_F) <= N
        if q:
            # Add (back) enough singletons to make size_F the same length as OSIZE.
            size_F = [size_F,np.ones((1,N-len(size_F)))]

            # Each value in SIZE_F must be unity (or zero), or must match OSIZE.
            q = np.all(size_F == 1 | size_F == osize)
        if not q:
            error('images:tformarray:sizeMismatchFA')
    return F

def GetBlocking(tsize, P):
    # With large input arrays, the memory used by the grid and/or map
    # (G and M) in the main function can be substantial and may exceed
    # the memory used by the input and output images -- depending on the
    # number of input transform dimensions, the number of non-transform
    # dimensions, and the storage class of the input and output arrays.
    # So the main function may compute the map block-by-block, resample
    # the input for just that block, and assign the result to a pre-allocated
    # output array.  We define the blocks by slicing the output transform space
    # along lines, planes, or hyperplanes normal to each of the subscript
    # axes.  We call the number of regions created by the slicing along a
    # given axis the 'blocking factor' for that dimension.  (If no slices
    # are made, the blocking factor is 1.  If all the blocking factors are
    # 1 (which can be tested by checking if prod(blockingfactors) == 1),
    # then the output array is small enough to process as a single block.)
    # The blockingfactors array created by this function has the same
    # size as TSIZE, and its dimensions are ordered as in TSIZE.
    #
    # This particular implementation has the following properties:
    # - All blocking factors are powers of 2
    # - After excluding very narrow dimensions, it slices across the
    #   remaining dimensions to defined blocks whose dimensions tend
    #   toward equality. (Of course, for large aspect ratios and relatively
    #   few blocks, the result may not actually get very close to equality.)

    # Control parameters
    Nt = 20  # Defines target block size
    Nm =  2  # Defines cut-off to avoid blocking on narrow dimensions
    # The largest block will be on the order of 2^Nt doubles (but may be
    # somewhat larger).  Any dimensions with size less than 2^Nm (= 4)
    # will have blocking factors of 1.  They are tagged by zeros in the
    # qDivisible array below.

    # Separate the dimensions that are too narrow to divide into blocks.
    if isinstance(tsize,list):
        tsize = np.array(tsize)
    qDivisible = tsize > 2**(Nm+1)
    L = np.sum(qDivisible)

    # The number of blocks will be 2^N.  As a goal, each block will
    # contain on the order of 2^Nt values, but larger blocks may be
    # needed to avoid subdividing narrow dimensions too finely.
    #
    # If all dimensions are large, then we'd like something like
    #
    #   (1) N = floor(log2(prod(tsize)/(2^Nt/P)))
    #
    # That's because we'd like prod(size(M)) / 2^N to be on the order of 2^Nt,
    # and prod(size(M)) = prod(tsize) * P.  The following slightly more complex
    # formula is equivalent to (1), but divides out the non-divisible dimensions
    #
    #   (2) N = floor(log2(prod(tsize(qDivisible)) / ...
    #                             (2^Nt/(P*prod(tsize(~qDivisible))))))
    #
    # However, it is possible that we might not be able to block an image as
    # finely as we'd like if its size is due to having many small dimensions
    # rather than a few large ones.  That's why the actual formula for N in
    # the next line of code replaces the numerator in (2) with 2^((Nm+1)*L)
    # if this quantity is larger. In such a case, we'd have
    #
    #   (3) N = floor(log2(prod(tsize(qDivisible)) / 2^((Nm+1)*L)))
    #
    # and would have to make do with fewer blocks in order to ensure a minimum
    # block size greater than or equal to 2^Nm.  The fact that our block sizes
    # always satisfy this constraint is proved in the comments at the end of
    # this function.

    N = np.floor(np.log2(np.prod(tsize[qDivisible],axis=0) /
                         max( (2**Nt)/(P*np.prod(tsize[~qDivisible],axis=0)), 2**((Nm+1)*L) )))
    N = int(N)

    # Initialize the blocking factor for the each divisible dimensions
    # as unity.  Iterate N times, each time multiplying one of the
    # blocking factors by 2.  The choice of which to multiply is driven
    # by the goal of making the final set of average divisible block
    # dimensions (stored in D at the end of the iteration) as uniform
    # as possible.
    B = np.ones((L,))
    D = tsize[qDivisible]
    blockingfactors = np.zeros(tsize.shape)

    for i in range(N):
        k = np.where(D == max(D))[0]
        k = k[0]  # Take the first if there is more than one maximum
        B[k] = B[k] * 2
        D[k] = D[k] * 0.5
    blockingfactors[qDivisible] = B
    blockingfactors[~qDivisible] = 1

    # Assertion: After the loop is complete, all(D >= 2^Nm).
    if np.any(D < 2**Nm):
        error('images:tformarray:blockDimsTooSmall')

    # Let Dmin = min(D) after the loop is complete.  Dmin is the smallest
    # average block size among the 'divisible' dimensions.  The following
    # is a proof that Dmin >= 2^Nm.
    #
    # There are two possibilities: Either the blocking factor corresponding
    # to Dmin is 1 or it is not.  If the blocking factor is 1, the proof
    # is trival: Dmin > 2^(Nm+1) > 2^Nm (otherwise this dimension would
    # not be included by virtue of qDivisible being false). Otherwise,
    # because the algorithm continually divides the largest
    # element of D by 2, it is clear that when the loop is complete
    #
    #   (1)  Dmin >= (1/2) * max(D).
    #
    # max(D) must equal or exceed the geometric mean of D,
    #
    #   (2)  max(D) >= (prod(D))^(1/L).
    #
    # From the formula for N,
    #
    #   (3)  N <= log2(prod(tsize(qDivisible)) / 2^((Nm+1)*L)).
    #
    # Because exactly N divisions by 2 occur in the loop,
    #
    #   (4)  prod(tsize(qDivisible) = (2^N) * prod(D)
    #
    # Combining (3) and (4),
    #
    #   (5)  prod(D) >= 2^((Nm+1)*L).
    #
    # Combining (1), (2), and (5) completes the proof,
    #
    #   (6)  Dmin >= (1/2) * max(D)
    #             >= (1/2)*(prod(D))^(1/L)
    #             >= (1/2)*(2^((Nm+1)*L))^(1/L) = 2^Nm.

    return blockingfactors

#--------------------------------------------------------------------------
def ConstructGrid( lo, hi ):

    # Constructs a regular grid from the range of subscripts defined by
    # vectors LO and HI, and packs the result into a single array, G.
    # G is D(1) x D(2) x ... x D(N) x N where D(k) is the length of
    # lo(k):hi(k) and N is the length of LO and HI.

    N = len(hi)
    E = [[]]*N
    for i in range(N):
        E[i] = list(range(int(lo[i]),int(hi[i])))  # The subscript range for each dimension
    G = CombinedGrid(E)
    return G

#--------------------------------------------------------------------------
def CombinedGrid(args):
    # If N >= 2, G = COMBINEDGRID(x1,x2,...,xN) is a memory-efficient
    # equivalent to:
    #
    #   G = cell(N,1)
    #   [G{:}] = ndgrid(x1,x2,...,xN)
    #   G = cat(N + 1, G{:})
    #
    # (where x{i} = varargin{i} and D(i) is the number of
    # elements in x{}).
    #
    # If N == 1, COMBINEDGRID returns x1 as a column vector. (The code
    # with NDGRID replicates x1 across N columns, returning an N-by-N
    # matrix -- which is inappropriate for our application.)
    #
    # N == 0 is not allowed.

    N = len(args)
    if N == 0:
        error('images:tformarray:combinegridCalledNoArgs')

    D = np.zeros((N,),dtype=int)
    for i in range(N):
        D[i] = len(args[i])

    if N == 1:
        # Special handling required to avoid calling RESHAPE
        # with a single-element size vector.
        G = args[0]
    else:
        # Pre-allocate the output array.
        G = np.zeros(np.append(D,N))

        for i in range(N):
            # Extract the i-th vector, reshape it to run along the
            # i-th dimension (adding singleton dimensions as needed),
            # replicate it across all the other dimensions, and
            # copy it to G(:,:,...,:,i) -- with N colons.
            xi = np.array(args[i],dtype=int)
            xi = xi.reshape(np.hstack((np.ones((i,),dtype=int), D[i], np.ones((N-i-1,),dtype=int))))
            xi = np.tile(xi,np.hstack((D[0:i], 1 , D[i+1:N])))
            G[:,:,i] = xi
    return G

def GetBlockLimits( tsize_B, blockingfactors ):

    # Construct matrices LO and HI containing the lower and upper limits for
    # each block, making sure to enlarge the upper limit of the right-most
    # block in each dimension as needed to reach the edge of the image.
    # LO and HI are each nBlocks-by-N.

    N = len(tsize_B)
    blocksizes = np.floor(tsize_B / blockingfactors)
    blocksubs  = [0, np.ones((1,N-1))]
    delta      = [1, np.zeros((1,N-1))]
    nBlocks = np.prod(blockingfactors)
    lo = np.zeros(nBlocks,N)
    hi = np.zeros(nBlocks,N)
    for i in range(nBlocks):
        blocksubs = blocksubs + delta
        while np.any(blocksubs > blockingfactors):
            # 'Carry' to higher dimensions as required.
            k = np.where(blocksubs > blockingfactors)
            blocksubs[k] = 1
            blocksubs[k+1] = blocksubs[k+1] + 1
        lo[i,:] = 1 + (blocksubs - 1) * blocksizes
        hi[i,:] = blocksubs * blocksizes
        qUpperLim = blocksubs == blockingfactors
        hi[i,qUpperLim] = tsize_B[qUpperLim]
    if np.any(blocksubs != blockingfactors):
        error('images:tformarray:failedToIterate')

    return lo, hi

def ConstructSubscripts( fsize_B, tdims_B, lo, hi ):
    # Determines the size of the block of array B bounded by the values in
    # vectors LO and HI, and constructs an array of subscripts for assigning
    # an array into that block.

    bsize_B = fsize_B
    bsize_B[tdims_B] = hi - lo + 1

    fdims_B = len(fsize_B)
    S = np.repmat([':'],1,fdims_B)
    for i in range(len(tdims_B)):
        S[tdims_B[i]] = list(range(lo[i], hi[i]))
    return S



def tforminv(*args):
    return tform('inv', *args)

def tformfwd(*args):
    return tform('fwd', *args)

def tform_x(direction, *args):
    f = {}
    if direction == 'fwd':
        f["name"] = 'TFORMFWD'
        f["ndims_in"] = 'ndims_in'
        f["ndims_out"] = 'ndims_out'
        f["fwd_fcn"] = 'forward_fcn'
        f["argname"] = 'U'
        f["arglist"] = 'U1,U2,U3'
    else:
        f["name"] = 'TFORMINV'
        f["ndims_in"] = 'ndims_out'
        f["ndims_out"] = 'ndims_in'
        f["fwd_fcn"] = 'inverse_fcn'
        f["argname"] = 'X'
        f["arglist"] = 'X1,X2,X3'

    t = args[0] ## Get the TFORM struct
    A = args[1] ## others

    P = t[f["ndims_in"]]  # Dimensionality of input space.
    L = t[f["ndims_out"]]  # Dimensionality of output space.

    U = A
    D = A.shape[0]

    X = t[f["fwd_fcn"]](U, t)

    return X

def tform(direction, *args):
    f = {}
    if direction == 'fwd':
        f["name"] = 'TFORMFWD'
        f["ndims_in"] = 'ndims_in'
        f["ndims_out"] = 'ndims_out'
        f["fwd_fcn"] = 'forward_fcn'
        f["argname"] = 'U'
        f["arglist"] = 'U1,U2,U3'
    else:
        f["name"] = 'TFORMINV'
        f["ndims_in"] = 'ndims_out'
        f["ndims_out"] = 'ndims_in'
        f["fwd_fcn"] = 'inverse_fcn'
        f["argname"] = 'X'
        f["arglist"] = 'X1,X2,X3'

    t = args[0] # Get the TFORM struct
    A = args[1]

    P = t[f["ndims_in"]]  # Dimensionality of input space.
    L = t[f["ndims_out"]]  # Dimensionality of output space.

    if isinstance(A,np.ndarray):
        [U, D] = checkCoordinates(f, P, A)
        U = U.reshape((np.prod(D), P))
        X = t[f["fwd_fcn"]](U, t)
        X = X.reshape(D + (L,))
    elif isinstance(A,list): # A=[U1,U2,U3.....]
        [U, D] = concatenateCoordinates(f, P, A)
        X = t[f["fwd_fcn"]](U, t)
        X = X.reshape((D,L))

    return X


def checkCoordinates(f,P,U):
    M = np.ndim(U)
    S = U.shape

    if S[M-1] > 1:
        if P == 1:
            N = M         # PROD(S) points in 1-D
        elif P == S[M-1]:
            N = M - 1     # PROD(S(1:M-1)) points in P-D
        else:
            error('images:tform:ArraySizeTformMismatch:'+ " ".join([f["name"], f["argname"],f["ndims_in"]]))
        D = S[0:N]
    else: # S == [S(1) 1]
        if P == 1:
            D = S[0]      # S(1) points in 1-D
        elif P == S[0]:
            D = 1        # 1 point in P-D
        else:
            error('images:tform:ArraySizeTformMismatch:'+ " ".join([f["name"], f["argname"],f["ndims_in"]]))
    return [U,D]

def concatenateCoordinates(f, P, A):
    if len(A) != P:
        error('images:tform:InputCountTformMismatch:'+ f["name"] + " " + f["ndims_in"])

    A  = np.array(A).T
    # Check argument class, properties, consistency.
    size1 = A[:,0].shape

    # Determine the size vector, D.
    D = size1
    if len(D) == 1:
        # U1,U2,... are column vectors.  They must be specified as
        # 1-D because MATLAB does not support explicit 1-D arrays.
        D = D[0]

    U = A

    return [U, D]





