import os
import numpy as np
from numpy.linalg import inv, norm, lstsq, solve
from numpy.linalg import matrix_rank as rank

from merfish.analysis.image_utils import maketform
from utils.funcs import *

## reference https://github.com/clcarwin/sphereface_pytorch/blob/master/matlab_cp2tform.py

def WarpPoints(uv, xy, method, **kwargs):
    # Infer spatial transformation from control point pairs.
    #   Adapted from CP2TFORM
    #
    #   CP2TFORM takes pairs of control points and uses them to infer a
    #   spatial transformation.
    #
    #   TFORM = CP2TFORM(MOVINGPOINTS,FIXEDPOINTS,TRANSFORMTYPE) returns a TFORM
    #   structure containing a spatial transformation. MOVINGPOINTS is an
    #   M-by-2 double matrix containing the X and Y coordinates of control
    #   points in the moving image you want to transform. FIXEDPOINTS is an
    #   M-by-2 double matrix containing the X and Y coordinates of control
    #   points in the fixed image. TRANSFORMTYPE can be 'nonreflective
    #   similarity', 'similarity', 'affine', 'projective', 'polynomial',
    #   'piecewise linear' or 'lwm'. See the reference page for CP2TFORM for
    #   information about choosing TRANSFORMTYPE.
    #
    #   TFORM = CP2TFORM(CPSTRUCT,TRANSFORMTYPE) works on a CPSTRUCT structure
    #   that contains the control point matrices for the input and base
    #   images. The Control Point Selection Tool, CPSELECT, creates the
    #   CPSTRUCT.
    #
    #   [TFORM,MOVINGPOINTS,FIXEDPOINTS] = CP2TFORM(CPSTRUCT,...) returns the
    #   control points that were actually used in MOVINGPOINTS, and
    #   FIXEDPOINTS. Unmatched and predicted points are not used. See
    #   CPSTRUCT2PAIRS.
    #
    #   TFORM = CP2TFORM(MOVINGPOINTS,FIXEDPOINTS,'polynomial',ORDER)
    #   ORDER specifies the order of polynomials to use. ORDER can be 2, 3, 4.
    #   If you omit ORDER, it defaults to 3.
    #
    #   TFORM = CP2TFORM(CPSTRUCT,'polynomial',ORDER) works on a CPSTRUCT
    #   structure.
    #
    #   TFORM = CP2TFORM(MOVINGPOINTS,FIXEDPOINTS,'piecewise linear') Creates a
    #   delaunay triangulation of the base control points, and maps
    #   corresponding input control points to the base control points. The
    #   mapping is linear (affine) for each triangle, and continuous across the
    #   control points, but not continuously differentiable as each triangle has
    #   its own mapping.
    #
    #   TFORM = CP2TFORM(CPSTRUCT,'piecewise linear') works on a CPSTRUCT
    #   structure.
    #
    #   TFORM = CP2TFORM(MOVINGPOINTS,FIXEDPOINTS,'lwm',N) The local weighted
    #   mean (lwm) method creates a mapping, by inferring a polynomial at each
    #   control point using neighboring control points. The mapping at any
    #   location depends on a weighted average of these polynomials.  You can
    #   optionally specify the number of points, N, used to infer each
    #   polynomial. The N closest points are used to infer a polynomial of order
    #   2 for each control point pair. If you omit N, it defaults to 12. N can
    #   be as small as 6, BUT making N small risks generating ill-conditioned
    #   polynomials.
    #
    #   TFORM = CP2TFORM(CPSTRUCT,'lwm',N) works on a CPSTRUCT structure.
    #
    #   [TFORM,MOVINGPOINTS,FIXEDPOINTS,MOVINGPOINTS_BAD,FIXEDPOINTS_BAD] = ...
    #        CP2TFORM(MOVINGPOINTS,FIXEDPOINTS,'piecewise linear')
    #   returns the control points that were actually used in MOVINGPOINTS and
    #   FIXEDPOINTS, and the control points that were eliminated because they
    #   were middle vertices of degenerate fold-over triangles in
    #   MOVINGPOINTS_BAD and FIXEDPOINTS_BAD.
    #
    #   [TFORM,MOVINGPOINTS,FIXEDPOINTS,MOVINGPOINTS_BAD,FIXEDPOINTS_BAD] = ...
    #        CP2TFORM(CPSTRUCT,'piecewise linear') works on a CPSTRUCT structure.
    #
    #   TRANSFORMTYPE
    #   -------------
    #   CP2TFORM requires a minimum number of control point pairs to infer a
    #   TFORM structure of each TRANSFORMTYPE:
    #
    #       TRANSFORMTYPE         MINIMUM NUMBER OF PAIRS
    #       -------------         -----------------------
    #       'translation rotation'           2
    #       'nonreflective similarity'       2
    #       'similarity'                     3
    #       'affine'                         3
    #       'projective'                     4
    #       'polynomial' (ORDER=2)           6
    #       'polynomial' (ORDER=3)          10
    #       'polynomial' (ORDER=4)          15
    #       'piecewise linear'               4
    #       'lwm'                            6
    #
    #   When TRANSFORMTYPE is 'nonreflective similarity', 'similarity', 'affine',
    #   'projective', or 'polynomial', and MOVINGPOINTS and FIXEDPOINTS (or
    #   CPSTRUCT) have the minimum number of control points needed for a particular
    #   transformation, the coefficients are found exactly. If MOVINGPOINTS and
    #   FIXEDPOINTS have more than the minimum, a least squares solution is
    #   found. See MLDIVIDE.
    #
    #   Note
    #   ----
    #   When either MOVINGPOINTS or FIXEDPOINTS has a large offset with
    #   respect to their origin (relative to range of values that it spans), the
    #   points are shifted to center their bounding box on the origin before
    #   fitting a TFORM structure.  This enhances numerical stability and
    #   is handled transparently by wrapping the origin-centered TFORM within a
    #   custom TFORM that automatically applies and undoes the coordinate shift
    #   as needed. This means that fields(T) may give different results for
    #   different coordinate inputs, even for the same TRANSFORMTYPE.
    #
    #   Example
    #   -------
    #   I = checkerboard
    #   J = imrotate(I,30)
    #   FIXEDPOINTS = [11 11 41 71]
    #   MOVINGPOINTS = [14 44 70 81]
    #   cpselect(J,I,MOVINGPOINTS,FIXEDPOINTS)
    #
    #   t = cp2tform(MOVINGPOINTS,FIXEDPOINTS,'nonreflective similarity')
    #
    #   # Recover angle and scale by checking how a unit vector
    #   # parallel to the x-axis is rotated and stretched.
    #   u = [0 1]
    #   v = [0 0]
    #   [x, y] = tformfwd(t, u, v)
    #   dx = x(2) - x(1)
    #   dy = y(2) - y(1)
    #   angle = (180/pi) * atan2(dy, dx)
    #   scale = 1 / sqrt(dx^2 + dy^2)
    #
    #  See also CPSELECT, CPCORR, CPSTRUCT2PAIRS, FITGEOTRANS,
    #           IMTRANSFORM, TFORMFWD, TFORMINV.

    #   Copyright 1993-2013 The MathWorks, Inc.

    # Note: 'linear conformal' was deprecated in R2008a in favor of
    # 'nonreflective similarity'. Both will still work and give the same
    # result as each other.

    # initialize deviation matrices
    xy_dev = []
    uv_dev = []
    options = {'order':3, "K":[]}

    # Assign function according to method and
    # set K = number of control point pairs needed.
    if method == 'translation rotation':
        findT_fcn = findTranslationRotation
        options["K"] = 2
    elif method == 'nonreflective similarity':
        findT_fcn = findNonreflectiveSimilarity
        options["K"] = 2
    elif method == 'similarity':
        findT_fcn = findSimilarity
        options["K"] = 3
    elif method == 'affine':
        findT_fcn = findAffineTransform
        options["K"] = 3
    elif method == 'projective':
        findT_fcn = findProjectiveTransform
        options["K"] = 4
    elif method == 'polynomial':
        findT_fcn = findPolynomialTransform
        if "order" in kwargs:
            options["order"] = kwargs["order"]
            options["K"] = (options["order"] + 1) * (options["order"] + 2) / 2
        else:
            error('[Error]:WarpPoints:an order value is needed - ' +"(", +method + ")")
    elif method == 'piecewise linear':
        findT_fcn = findPiecewiseLinear
        options["K"] = 4
    elif method == 'lwm':
        findT_fcn = findLWM
        options["order"] = 2
        options["K"] = (options["order"] + 1) * (options["order"] + 2) / 2
        if "N" not in kwargs:
            options["N"] =  options["K"] * 2
        else:
            if  options["N"] < options["K"]:
                error('[Error]:cp2tform:invalidInputN')
    else:
        error('[Error] WarpPoints:internalProblem')

    # error if user enters too few control point pairs
    M = uv.shape[0]
    if "K" in options and M < options["K"]:
        error('[Error]:WarpPoints:rankError - '+ str(options["K"]),+"(",+method+")")

    # get offsets to apply to before/after spatial transformation
    uvShift = getShift(uv)
    xyShift = getShift(xy)
    needToShift = np.any(np.hstack((uvShift,xyShift)) != 0)

    if not needToShift:
        # infer transform
        [trans, output] = findT_fcn(uv,xy,options)
    else:
        # infer transform for shifted data
        [tshifted, output] = findT_fcn(applyShift(uv,uvShift),
                                       applyShift(xy,xyShift),options)

        # construct custom tform with tshifted between forward and inverse shifts
        tdata = {'uvShift':uvShift,'xyShift':xyShift,'tshifted':tshifted}
        trans = maketform('custom',2,2,fwd,inverse,tdata)

    if 'piecewise linear' in method:
        uv = undoShift(output["uv"],uvShift)  # No-ops if needToShift
        xy = undoShift(output["xy"],xyShift)  # is false.
        uv_dev = output["uv_dev"]
        xy_dev = output["xy_dev"]

    return trans,uv,xy,uv_dev,xy_dev


def findTranslationRotation(uv,xy,options):
    # For a nonreflective similarity:
    #
    # let sc = s*cos(theta)
    # let ss = s*sin(theta)
    #
    #                   [ sc -ss
    # [u v] = [x y 1] *   ss  sc
    #                     tx  ty]
    #
    # There are 4 unknowns: sc,ss,tx,ty.
    #
    # Another way to write this is:
    #
    # u = [x y 1 0] * [sc
    #                  ss
    #                  tx
    #                  ty]
    #
    # v = [y -x 0 1] * [sc
    #                   ss
    #                   tx
    #                   ty]
    #
    # With 2 or more correspondence points we can combine the u equations and
    # the v equations for one linear system to solve for sc,ss,tx,ty.
    #
    # [ u1  ] = [ x1  y1  1  0 ] * [sc]
    # [ u2  ]   [ x2  y2  1  0 ]   [ss]
    # [ ... ]   [ ...          ]   [tx]
    # [ un  ]   [ xn  yn  1  0 ]   [ty]
    # [ v1  ]   [ y1 -x1  0  1 ]
    # [ v2  ]   [ y2 -x2  0  1 ]
    # [ ... ]   [ ...          ]
    # [ vn  ]   [ yn -xn  0  1 ]
    #
    # Or rewriting the above matrix equation:
    # U = X * r, where r = [sc ss tx ty]'
    # so r = X\U.
    #

    K = 2
    if "K" in options: K = options["K"]

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print '--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print '--->X.shape: ', X.shape
    # print 'X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print '--->U.shape: ', U.shape
    # print 'U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        if X.shape[0] == X.shape[1]:
            r = solve(X, U)
        else:
            r, _, _, _ = lstsq(X, U,rcond=None)
            r = np.squeeze(r)
    else:
        error('[Error]: cp2tform:twoUniquePointsReq')

    # print '--->r:\n', r

    sc = 1
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ],dtype=np.float)

    # print '--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print '--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    trans = maketform('affine', T)
    output = {}

    return trans, output


def findNonreflectiveSimilarity(uv, xy, options=None):
    '''
    Function:
    ----------
        Find Non-reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans
    Parameters:
    ----------
        @uv: Kx2 np.array
            source points each row is a pair of coordinates (x, y)
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed
        @option: not used, keep it as None
    Returns:
        @trans: 3x3 np.array
            transform matrix from uv to xy
        @trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    Matlab:
    ----------
    % For a nonreflective similarity:
    %
    % let sc = s*cos(theta)
    % let ss = s*sin(theta)
    %
    %                   [ sc -ss
    % [u v] = [x y 1] *   ss  sc
    %                     tx  ty]
    %
    % There are 4 unknowns: sc,ss,tx,ty.
    %
    % Another way to write this is:
    %
    % u = [x y 1 0] * [sc
    %                  ss
    %                  tx
    %                  ty]
    %
    % v = [y -x 0 1] * [sc
    %                   ss
    %                   tx
    %                   ty]
    %
    % With 2 or more correspondence points we can combine the u equations and
    % the v equations for one linear system to solve for sc,ss,tx,ty.
    %
    % [ u1  ] = [ x1  y1  1  0 ] * [sc]
    % [ u2  ]   [ x2  y2  1  0 ]   [ss]
    % [ ... ]   [ ...          ]   [tx]
    % [ un  ]   [ xn  yn  1  0 ]   [ty]
    % [ v1  ]   [ y1 -x1  0  1 ]
    % [ v2  ]   [ y2 -x2  0  1 ]
    % [ ... ]   [ ...          ]
    % [ vn  ]   [ yn -xn  0  1 ]
    %
    % Or rewriting the above matrix equation:
    % U = X * r, where r = [sc ss tx ty]'
    % so r = X \ U.
    %
    '''
    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print '--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print '--->X.shape: ', X.shape
    # print 'X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print '--->U.shape: ', U.shape
    # print 'U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        if X.shape[0] == X.shape[1]:
            r = solve(X, U)
        else:
            r, _, _, _ = lstsq(X, U, rcond=None)
            r = np.squeeze(r)
    else:
        error('[Error]: cp2tform:twoUniquePointsReq')

    # print '--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ],dtype=np.float)

    # print '--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print '--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    trans = maketform('affine', T)
    output = {}

    return trans,output


def findSimilarity(uv, xy, options=None):
    '''
    Function:
    ----------
        Find Reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans
    Parameters:
    ----------
        @uv: Kx2 np.array
            source points each row is a pair of coordinates (x, y)
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed
        @option: not used, keep it as None
    Returns:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy
        @trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    Matlab:
    ----------
    % The similarities are a superset of the nonreflective similarities as they may
    % also include reflection.
    %
    % let sc = s*cos(theta)
    % let ss = s*sin(theta)
    %
    %                   [ sc -ss
    % [u v] = [x y 1] *   ss  sc
    %                     tx  ty]
    %
    %          OR
    %
    %                   [ sc  ss
    % [u v] = [x y 1] *   ss -sc
    %                     tx  ty]
    %
    % Algorithm:
    % 1) Solve for trans1, a nonreflective similarity.
    % 2) Reflect the xy data across the Y-axis,
    %    and solve for trans2r, also a nonreflective similarity.
    % 3) Transform trans2r to trans2, undoing the reflection done in step 2.
    % 4) Use TFORMFWD to transform uv using both trans1 and trans2,
    %    and compare the results, Returnsing the transformation corresponding
    %    to the smaller L2 norm.
    % Need to reset options.K to prepare for calls to findNonreflectiveSimilarity.
    % This is safe because we already checked that there are enough point pairs.
    '''

    options = {'K': 2}

#    uv = np.array(uv)
#    xy = np.array(xy)

    # Solve for trans1
    trans1, trans1_out = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_out = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    ## matlab code
    # trans2 = maketform('affine', trans2r.tdata.T * TreflectY)
    trans2 = maketform('affine', trans2r.tdata.T @ TreflectY)
    # trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd_x(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd_x(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_out
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2r_out


def findAffineTransform():
    pass

def findProjectiveTransform():
    pass

def findPolynomialTransform():
    pass

def findPiecewiseLinear():
    pass

def findLWM():
    pass

def tformfwd_x(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

def tforminv_x(trans, uv):
    """
    Function:
    ----------
        apply the inverse of affine transform 'trans' to uv
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed coordinates (x, y)
    """
    Tinv = inv(trans)
    xy = tformfwd_x(Tinv, uv)
    return xy

def getShift(points):
    tol = 1e+3
    minPoints = np.min(points, axis=0)
    maxPoints = np.max(points, axis=0)
    center = (minPoints + maxPoints) / 2
    span = maxPoints - minPoints
    if (span[0] > 0 and (abs(center[0])/span[0] > tol)) or (span[1] > 0 and (abs(center[1]) / span[1] > tol)):
        shift = center
    else:
        shift = [0,0]
    return shift

def applyShift(points,shift):
    shiftedPoints = points-shift
    return shiftedPoints

def undoShift(shiftedPoints,shift):
    points = shiftedPoints+shift
    return points

#-------------------------------
def fwd(u,t):
    x = undoShift(tformfwd_x(applyShift(u,t["tdata"]["uvShift"]),t["tdata"]["tshifted"]),t["tdata"]["xyShift"])
    return x
#-------------------------------
def inverse(x,t):
    u = undoShift(tforminv_x(applyShift(x,t["tdata"]["xyShift"]),t["tdata"]["tshifted"]),t["tdata"]["uvShift"])
    return u






