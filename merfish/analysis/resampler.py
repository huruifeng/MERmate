import numpy as np
import numpy.linalg
import scipy

from utils.funcs import *
from merfish.analysis.resampsep import do_resampsep


def makeresampler(*args):
    #MAKERESAMPLER Create resampling structure.
    #   R = MAKERESAMPLER(INTERPOLANT,PADMETHOD) creates a separable
    #   resampler structure for use with TFORMARRAY and IMTRANSFORM.
    #   In its simplest form, INTERPOLANT can be one of these strings
    #   or character vectors: 'nearest', 'linear', or 'cubic'.
    #   INTERPOLANT specifies the interpolating kernel that the
    #   separable resampler uses.  PADMETHOD can be one of these:
    #   'replicate', 'symmetric', 'circular','fill', or 'bound'.
    #   PADMETHOD controls how the resampler to interpolates or
    #   assigns values to output elements that map close to or
    #   outside the edge of input array.
    #
    #   PADMETHOD options
    #   -----------------
    #   In the case of 'fill', 'replicate', 'circular', or 'symmetric',
    #   the resampling performed by TFORMARRAY or IMTRANSFORM occurs in
    #   two logical steps: (1) pad A infinitely to fill the entire input
    #   transform space, then (2) evaluate the convolution of the padded
    #   A with the resampling kernel at the output points specified by
    #   the geometric map.  Each non-transform dimension is handled
    #   separately.  The padding is virtual, (accomplished by remapping
    #   array subscripts) for performance and memory efficiency.
    #
    #   'circular', 'replicate', and 'symmetric' have the same meanings as
    #   in PADARRAY as applied to the transform dimensions of A:
    #
    #     'replicate' -- Repeats the outermost elements
    #     'circular'  -- Repeats A circularly
    #     'symmetric' -- Mirrors A repeatedly.
    #
    #   'fill' generates an output array with smooth-looking edges (except
    #   when using nearest neighbor interpolation) because for output points
    #   that map near the edge of the input array (either inside or outside),
    #   it combines input image and fill values .
    #
    #   'bound' is like 'fill', but avoids mixing fill values and input image
    #   values.  Points that map outside are assigned values from the fill
    #   value array.  Points that map inside are treated as with 'replicate'.
    #   'bound' and 'fill' produce identical results when INTERPOLANT is
    #   'nearest'.
    #
    #   It is up to the user to implement these behaviors in the case of a
    #   custom resampler.
    #
    #   Advanced options for INTERPOLANT
    #   --------------------------------
    #   In general, INTERPOLANT can have one of these forms:
    #
    #       1. One of these : 'nearest', 'linear', 'cubic'
    #
    #       2. A cell array: {HALF_WIDTH, POSITIVE_HALF}
    #          HALF_WIDTH is a positive scalar designating the half width of
    #          a symmetric interpolating kernel.  POSITIVE_HALF is a vector
    #          of values regularly sampling the kernel on the closed interval
    #          [0 POSITIVE_HALF].
    #
    #       3. A cell array: {HALF_WIDTH, INTERP_FCN}
    #          INTERP_FCN is a function handle that returns interpolating
    #          kernel values given an array of input values in the interval
    #          [0 POSITIVE_HALF].
    #
    #       4. A cell array whose elements are one of the three forms above.
    #
    #   Forms 2 and 3 are used to interpolate with a custom interpolating
    #   kernel.  Form 4 is used to specify the interpolation method
    #   independently along each dimension.  The number of elements in the
    #   cell array for form 4 must equal the number of transform dimensions.
    #   For example, if INTERPOLANT is {'nearest', 'linear', {2
    #   KERNEL_TABLE}}, then the resampler will use nearest-neighbor
    #   interpolation along the first transform dimension, linear
    #   interpolation along the second, and a custom table-based
    #   interpolation along the third.
    #
    #   Custom resamplers
    #   -----------------
    #   The syntaxes described above construct a resampler structure that
    #   uses the separable resampler function that ships with the Image
    #   Processing Toolbox.  It is also possible to create a resampler
    #   structure that uses a user-written resampler by using this syntax:
    #   R = MAKERESAMPLER(PropertyName,PropertyValue,...).  PropertyName can
    #   be 'Type', 'PadMethod', 'Interpolant', 'NDims', 'ResampleFcn', or
    #   'CustomData'.
    #
    #   'Type' can be either 'separable' or 'custom' and must always be
    #   supplied.  If 'Type' is 'separable', the only other properties that can
    #   be specified are 'Interpolant' and 'PadMethod', and the result is
    #   equivalent to using the MAKERESAMPLER(INTERPOLANT,PADMETHOD) syntax.
    #   If 'Type' is 'custom', then 'NDims' and 'ResampleFcn' are required
    #   properties, and 'CustomData' is optional.  'NDims' is a positive
    #   integer and indicates what dimensionality the custom resampler can
    #   handle.  Use a value of Inf to indicate that the custom resampler can
    #   handle any dimension.  The value of 'CustomData' is unconstrained.
    #
    #   'ResampleFcn' is a handle to a function that performs the resampling.
    #   The function will be called with the following interface:
    #
    #       B = RESAMPLE_FCN(A,M,TDIMS_A,TDIMS_B,FSIZE_A,FSIZE_B,F,R)
    #
    #   See the help for TFORMARRAY for information on the inputs A, TDIMS_A,
    #   TDIMS_B, and F.
    #
    #   M is an array that maps the transform subscript space of B to the
    #   transform subscript space of A.  If A has N transform dimensions (N =
    #   length(TDIMS_A)) and B has P transform dimensions (P = length(TDIMS_B)),
    #   then NDIMS(M) = P + 1 if N > 1 and P if N == 1, and SIZE(M, P + 1) =
    #   N.  The first P dimensions of M correspond to the output transform
    #   space, permuted according to the order in which the output transform
    #   dimensions are listed in TDIMS_B.  (In general TDIMS_A and TDIMS_B need
    #   not be sorted in ascending order, although such a limitation may be
    #   imposed by specific resamplers.)  Thus the first P elements of SIZE(M)
    #   determine the sizes of the transform dimensions of B.  The input
    #   transform coordinates to which each point is mapped are arrayed across
    #   the final dimension of M, following the order given in TDIMS_A.  M must
    #   be double.
    #
    #   FSIZE_A and FSIZE_B are the full sizes of A and B, padded with 1s as
    #   necessary to be consistent with TDIMS_A, TDIMS_B, and SIZE(A).
    #
    #   Example
    #   -------
    #   Stretch an image in the y-direction using separable resampler that
    #   applies in cubic interpolation in the y-direction and nearest
    #   neighbor interpolation in the x-direction. (This is equivalent to,
    #   but faster than, applying bicubic interpolation.)
    #
    #       A = imread('moon.tif')
    #       resamp = makeresampler({'nearest','cubic'},'fill')
    #       stretch = maketform('affine',[1 0 0 1.3 0 0])
    #       B = imtransform(A,stretch,resamp)
    #
    #   See also IMTRANSFORM, TFORMARRAY.

    #   Copyright 1993-2017 The MathWorks, Inc.


    if len(args) % 2 != 0:
        error('images:makeresampler:invalidNumInputs')

    npairs = len(args)/2

    property_strings = ['type','padmethod','interpolant','ndims','resamplefcn','customdata']

    # Check for the shorthand syntax for separable resamplers.
    if npairs == 1:
        if isinstance(args[0],str):
            if FindValue('type', property_strings, args)==[]:
                r = MakeSeparable( args[0], args[1] )
                return r
        else:
            r = MakeSeparable( args[0], args[1] )
            return r

    # Parse property name/property value syntax.
    type = FindValue('type', property_strings, args)
    if type==[]:
        error('images:makeresampler:missingTYPE')
    canonical_str= GetCanonicalString(type,Type=['separable','custom'])
    if canonical_str == 'separable':
        [interpolant, padmethod] = ParseSeparable(property_strings, args)
        r = MakeSeparable(interpolant, padmethod )
    elif canonical_str == 'custom':
        r = MakeCustom( property_strings, args)
    else:
        error('images:makeresampler:internalError')
    return r

def FindValue( property_name, property_strings, args ):
    value = []
    nargs = len(args)
    property_strings = np.array(property_strings)
    for i in range(int(nargs/2)):
        current_name = args[2*i]
        if isinstance(current_name,str):
            imatch = np.argwhere(property_strings==current_name.lower()).ravel()
            nmatch = len(imatch)
            if nmatch > 1:
                error('images:makeresampler:ambiguousPropertyName', current_name)
            if nmatch == 1:
                canonical_name = property_strings[imatch[0]]
                if canonical_name==property_name:
                    if value == []:
                        if args[2*i+1]=="" or args[2*i+1]==[]:
                            error('images:makeresampler:emptyPropertyName - ' + property_name)
                        value = args[2*i+1]
                    else:
                        error('images:makeresampler:redundantPropertyName-'+ property_name)

    return value


def MakeSeparable( interpolant, padmethod ):

    standardFrequency = 1000  # Standard number of samples per unit
    n_dimensions = np.inf
    varargin = [interpolant,padmethod]

    rdata = {}
    if isinstance(interpolant,list):
        if HasCustomTable(interpolant):
            rdata["K"] = interpolant
        elif HasCustomFunction(interpolant):
            rdata["K"] = CustomKernel(interpolant, standardFrequency)
        else:
            n_dimensions = len(interpolant)
            rdata["K"] = MultipleKernels(interpolant, standardFrequency)
    else:
        rdata["K"] = StandardKernel(interpolant, standardFrequency)

    padmethod = GetCanonicalString( padmethod, 'PadMethod', ['fill','bound','replicate','circular','symmetric'])
    resampsep_fcn = resampsep
    r = AssignResampler(n_dimensions, padmethod,resampsep_fcn,rdata)
    return r


def GetCanonicalString(input_string, property_name, canonical_strings):
    if not isinstance(input_string,str):
        error('images:makeresampler:invalidPropertyName - '+property_name)

    canonical_strings = np.array(canonical_strings)
    imatch = np.argwhere(canonical_strings==input_string.lower()).ravel()
    nmatch = len(imatch)

    if nmatch == 0:
        error('images:makeresampler:unknownPropertyName-'+property_name+":"+input_string)

    if nmatch > 1:
        error('images:makeresampler:ambiguousPropertyValue - '+ property_name+":"+input_string)

    canonical_string = canonical_strings[imatch[0]]
    return canonical_string

def ParseSeparable(property_strings, args):
    interpolant = FindValue('interpolant', property_strings, args)
    if interpolant==[]:
        interpolant = 'linear'

    padmethod = FindValue('padmethod', property_strings, args)
    if padmethod==[]:
        padmethod = 'replicate'

    return interpolant, padmethod

def MakeCustom( property_strings, args):

    padmethod = FindValue('padmethod', property_strings, args)
    if padmethod==[]:
        padmethod = 'replicate'
    padmethod = GetCanonicalString(padmethod, 'PadMethod', ['fill','bound','replicate','circular','symmetric'])
    
    n_dimensions = FindValue('ndims', property_strings, args)
    if n_dimensions==[]:
        error('images:makeresampler:unspecifiedNDims')

    if len(n_dimensions) != 1:
        error('images:makeresampler:invalidNDimsNotScalar')

    if n_dimensions != np.floor(n_dimensions) or n_dimensions < 1:
        error('images:makeresampler:invalidNDimsNotPositive')
    
    resample_fcn = FindValue('resamplefcn', property_strings,args)
    if resample_fcn==[]:
        error('images:makeresampler:resampleFcnNotSpecified')
    
    if len(resample_fcn) != 1:
        error('images:makeresampler:invalidResampleFcnScalar')

    if not callable(resample_fcn):
        error('images:makeresampler:invalidResampleFcnHandle')
    
    rdata = FindValue('customdata', property_strings,args)
    
    r = AssignResampler(n_dimensions, padmethod, resample_fcn, rdata)
    return r

def HasCustomTable(interpolant):
    q = isinstance(interpolant,list)

    if q:
        q = len(interpolant) == 2

    if q:
        q = isinstance(interpolant[0],float) & (len(interpolant[0]) == 1) & \
            isinstance(interpolant[1],float)

    if q:
        q = interpolant[0] > 0

    return q

def HasCustomFunction(interpolant):
    q = isinstance(interpolant,list)

    if q:
        q = len(interpolant) == 2

    if q:
        q = isinstance(interpolant[0],float) & (len(interpolant[0]) == 1) & \
            callable(interpolant[1])

    if q:
        q = interpolant[0] > 0

    return q

def CustomKernel( interpolant, frequency ):
    halfwidth  = interpolant[0]
    kernel_fcn = interpolant[1]
    positiveHalf = SampleKernel(kernel_fcn, halfwidth, frequency)
    K = [halfwidth, positiveHalf]
    return K

def MultipleKernels( interpolant, frequency ):
    K = [[]]*len(interpolant)

    for i in range(len(interpolant)):
        if HasCustomTable(interpolant[i]):
            K[i] = interpolant[i]
        elif HasCustomFunction(interpolant[i]):
            K[i] = CustomKernel(interpolant[i], frequency)
        else:
            K[i] = StandardKernel(interpolant[i], frequency)
    return K

def StandardKernel( interpolant, frequency ):

    interpolant = GetCanonicalString( interpolant, 'Interpolant', ['nearest','linear','cubic'])
    if interpolant=='nearest':
        K = []
    elif interpolant == 'linear':
        halfwidth = 1.0
        positiveHalf = SampleKernel(LinearKernel, halfwidth, frequency)
        K = [halfwidth, positiveHalf]
    elif interpolant=='cubic':
        halfwidth = 2.0
        positiveHalf = SampleKernel(CubicKernel, halfwidth, frequency)
        K = [halfwidth, positiveHalf]
    else:
        error('images:makeresampler:invalidInterpolant')

    return K

#--------------------------------------------------------------------------
def SampleKernel( kernel, halfwidth, frequency ):

    if not callable(kernel):
       error('images:makeresampler:invalidKernel')

    n = np.floor(halfwidth * frequency)
    positiveHalf = kernel((halfwidth / n) * np.arange(0,n+1))
    return positiveHalf

#--------------------------------------------------------------------------
def LinearKernel( x ):
    y = np.zeros((len(x),))
    y = y.reshape(x.shape)
    x[x < 0] = -x[x < 0]
    q = x <= 1
    y[q] = 1 - x[q]
    return y

#--------------------------------------------------------------------------
def CubicKernel( x ):

    # There is a whole family of "cubic" interpolation kernels. The
    # particular kernel used here is described in the article Keys,
    # "Cubic Convolution Interpolation for Digital Image Processing,"
    # IEEE Transactions on Acoustics, Speech, and Signal Processing,
    # Vol. ASSP-29, No. 6, December 1981, p. 1155.

    y = np.zeros((1,len(x)))
    y = y.reshape(x.shape())
    x[x < 0] = -x[x < 0]

    q = [x <= 1]            # Coefficients: 1.5, -2.5, 0.0, 1.0
    y[q] = ((1.5 * x[q] - 2.5) * x[q]) * x[q] + 1.0

    q = [(1 < x) & (x <= 2)]   # Coefficients: -0.5, 2.5, -4.0, 2.0
    y[q] = ((-0.5 * x[q] + 2.5) * x[q] - 4.0) * x[q] + 2.0

    return y

#--------------------------------------------------------------------------

def AssignResampler(n_dimensions, padmethod, resamp_fcn, rdata):

    # Use this function to ensure consistency in the way we assign
    # the fields of each resampling struct. Note that r.ndims = Inf
    # is used to denote that the resampler supports an arbitrary
    # number of dimensions.

    r ={}
    r["ndims"]      = n_dimensions
    r["padmethod"]  = padmethod
    r["resamp_fcn"] = resamp_fcn
    r["rdata"]      = rdata

    return r

def resampsep(A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, R):
    # const mxArray*  A,         /* Input array */
    # const mxArray*  M,         /* Inverse mapping from output to input */
    # const mxArray*  tdims_A,   /* List of input transform dimensions */
    # const mxArray*  tdims_B,   /* List of output transform dimensions */
    # const mxArray*  fsize_A,   /* Full size of input array */
    # const mxArray*  fsize_B,   /* Full size of output block */
    # const mxArray*  F,         /* Defines the fill values and dimensions if non-empty */
    # const mxArray * R = prhs[7]; / *Resampler * /

    # const mxArray*  padstr,    /* Method for defining values for points that map outside A */
    # const mxArray*  K          /* Interplating kernel cell array */ )

    if "rdata" not in R:
        error("[Error]: resampsep - rdata field is missing.")
    else:
        rdata = R["rdata"]

    if "padmethod" not in R:
        error("[Error]: resampsep - padmethod field is missing.")
    else:
        padstr = R["padmethod"]

    if "K" not in rdata:
        error("[Error]: resampsep - rdata['K'] field is missing.")
    else:
        K = rdata["K"]

    if isinstance(tdims_A,list):
        tdims_A = np.array(tdims_A)
    if isinstance(tdims_B,list):
        tdims_B = np.array(tdims_B)


    R = do_resampsep(A, M, tdims_A, tdims_B, fsize_A, fsize_B, F, padstr, K)
    return R





