import numpy as np
from matplotlib.transforms import Affine2D
from numpy import linalg
from numpy.linalg import norm, matrix_rank, inv, solve, lstsq
from skimage.transform import ProjectiveTransform, AffineTransform, PolynomialTransform, SimilarityTransform

from utils.funcs import error


def transPntForward(pt, T):
    newPt = np.zeros(2, dtype=pt.dtype)
    newPt[0] = T[0, 0] * pt[0] + T[1, 0] * pt[1] + T[2, 0]
    newPt[1] = T[0, 1] * pt[0] + T[1, 1] * pt[1] + T[2, 1]
    return newPt


def transPntsForwardWithSameT(pts, T):
    if pts.ndim != 2:
        raise Exception("Must 2-D array")
    newPts = np.zeros(pts.shape)
    newPts[:, 0] = T[0, 0] * pts[:, 0] + T[1, 0] * pts[:, 1] + T[2, 0]
    newPts[:, 1] = T[0, 1] * pts[:, 0] + T[1, 1] * pts[:, 1] + T[2, 1]
    return newPts


def transPntsForwardWithDiffT(pts, Ts):
    if pts.ndim != 2:
        raise Exception("Must 2-D array")
    nPts = np.zeros(pts.shape)
    pntNum = pts.shape[0]

    for i in range(pntNum):
        T = Ts[i]
        nPts[i, 0] = T[0, 0] * pts[i, 0] + T[1, 0] * pts[i, 1] + T[2, 0]
        nPts[i, 1] = T[0, 1] * pts[i, 0] + T[1, 1] * pts[i, 1] + T[2, 1]
    return nPts


def fitGeoTrans(src, dst, mode="nonreflectivesimilarity",**kwargs):
    """
    This function is the same as matlab fitgeotrans
    """
    src = np.around(src, decimals=4).astype(np.float32)
    dst = np.around(dst, decimals=4).astype(np.float32)
    if "nonreflectivesimilarity" == mode:
        tform = findNonreflectiveSimilarity(src, dst)
        tform = tform.params
        tform_x = tform.T
    elif 'similarity' == mode:
        # tform = findSimilarityTransform(src, dst)
        # tform = tform.params
        tform_x = SimilarityTransform()
        tform_x.estimate(src, dst)
        tform_x = tform_x.params
    elif 'affine' == mode:
        # tform = findAffineTransform(src, dst)
        # tform = tform.params
        tform_x = AffineTransform()
        tform_x.estimate(src, dst)
        tform_x = tform_x.params
    elif 'projective' ==mode:
        # tform = findProjectiveTransform(src, dst)
        # tform = tform.params
        tform_x = ProjectiveTransform()
        tform_x.estimate(src, dst)
        tform_x = tform_x.params
    elif 'polynomial'==mode:
        if "order" in kwargs:
            order = kwargs["order"]
        else:
            order = 2
        tform = PolynomialTransform()
        tform.estimate(src, dst, order=order)
        tform = tform.params
        tform_x = tform

    else:
        raise Exception("Unsupported transformation")

    return tform_x


def findNonreflectiveSimilarity(uv, xy):
    uv, normMatSrc = normalizeControlPoints(uv)
    xy, normMatDst = normalizeControlPoints(xy)

    M = xy.shape[0]
    minRequiredNonCollinearPairs = 2

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

    ### X*r = U, Solve the r by least squared error
    if np.linalg.matrix_rank(X) >= 2 * minRequiredNonCollinearPairs:
        if X.shape[0] == X.shape[1]:
            r = solve(X, U)
        else:
            r = lstsq(X, U, rcond=None)[0]
            r = np.squeeze(r)
    else:
        error('[Error]: At least 2 noncollinear Pts')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ], dtype=np.float)
    if  normMatDst.shape[0] == normMatDst.shape[1]:
        Tinv = np.linalg.solve(normMatDst, Tinv@normMatSrc)
    else:
        Tinv = np.linalg.lstsq(normMatDst, Tinv @ normMatSrc, rcond=None)[0]

    T = np.linalg.inv(Tinv)
    T[:, 2] = [0, 0, 1]
    tform = AffineTransform(T)
    return tform


def normalizeControlPoints(pts):
    ptNum = pts.shape[0]
    cent = np.mean(pts, axis=0)
    ptsNorm = np.subtract(pts, cent)
    distSum = np.sum(np.power(ptsNorm, 2))
    if distSum > 0:
        scaleFactor = np.sqrt(2 * ptNum) / np.sqrt(distSum)
    else:
        scaleFactor = 1

    ptsNorm = scaleFactor * ptsNorm
    normMatInv = np.array(((1 / scaleFactor, 0, 0),
                           (0, 1 / scaleFactor, 0),
                           (cent[0], cent[1], 1)))
    return ptsNorm, normMatInv

def findSimilarityTransform(uv, xy):
    #
    # The similarities are a superset of the nonreflective similarities as they may
    # also include reflection.
    #
    # let sc = s*cos(theta)
    # let ss = s*sin(theta)
    #
    #                   [ sc -ss
    # [u v] = [x y 1] *   ss  sc
    #                     tx  ty]
    #
    #          OR
    #
    #                   [ sc  ss
    # [u v] = [x y 1] *   ss -sc
    #                     tx  ty]
    #
    # Algorithm:
    # 1) Solve for trans1, a nonreflective similarity.
    # 2) Reflect the xy data across the Y-axis, 
    #    and solve for trans2r, also a nonreflective similarity.
    # 3) Transform trans2r to trans2, undoing the reflection done in step 2.
    # 4) Use TFORMFWD to transform uv using both trans1 and trans2, 
    #    and compare the results, returning the transformation corresponding 
    #    to the smaller L2 norm.
    
    minRequiredNonCollinearPairs = 3
    
    M = uv.shape[0]
    if M < minRequiredNonCollinearPairs:
        error('images:geotrans:requiredNonCollinearPoints' +str(minRequiredNonCollinearPairs) + 'similarity')

    
    # Solve for trans1
    trans1 = findNonreflectiveSimilarity(uv,xy)
    
    # Solve for trans2
    
    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:,0] = -1*xyR[:,1]
    
    trans2r = findNonreflectiveSimilarity(uv,xyR)
    
    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = [[-1,0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
                  
    trans2 = AffineTransform(trans2r.T * TreflectY)
    
    # Figure out if trans1 or trans2 is better
    xy1 = transformPointsForward(trans1,uv[:,0],uv[:,1])
    norm1 = norm(xy1-xy)
    
    xy2 = transformPointsForward(trans2,uv[:,0],uv[:,1])
    norm2 = norm(xy2-xy)
    
    if norm1 <= norm2:
        tform = trans1
    else:
        tform = trans2
    return tform

def transformPointsForward(tform,v1,v2):
    if v1.ndim != 1 or v2.ndim != 1:
        print('Vectors must be column-shaped!')
        return
    elif v1.shape[0] != v2.shape[0]:
        print('Vectors must be of equal length!')
        return

    vecSize = v1.shape[0]

    concVec = np.stack((v1,v2),axis=1)
    onesVec = np.ones((vecSize,1))

    U = np.concatenate((concVec,onesVec),axis=1)

    retMat = np.dot(U,tform[:2,:].T)

    return np.stack((retMat[:,0].reshape((vecSize,)), retMat[:,1].reshape((vecSize,))),axis=1)

def transformPointsForward_x(tform,U):
    U = np.pad(U,((0,0),(0,1)),'constant', constant_values=(1, 1))
    X = U @ tform.T

    return X[:,:2]


def findAffineTransform(uv,xy):
    #
    # For an affine transformation:
    #
    #
    #                     [ A D 0 ]
    # [u v 1] = [x y 1] * [ B E 0 ]
    #                     [ C F 1 ]
    #
    # There are 6 unknowns: A,B,C,D,E,F
    #
    # Another way to write this is:
    #
    #                   [ A D ]
    # [u v] = [x y 1] * [ B E ]
    #                   [ C F ]
    #
    # Rewriting the above matrix equation:
    # U = X * T, where T = reshape([A B C D E F],3,2)
    #
    # With 3 or more correspondence points we can solve for T,
    # T = X\U which gives us the first 2 columns of T, and
    # we know the third column must be [0 0 1]'.

    [uv,normMatrix1] = normalizeControlPoints(uv)
    [xy,normMatrix2] = normalizeControlPoints(xy)

    minRequiredNonCollinearPairs = 3
    M = xy.shape[0]
    X = np.hstack([xy,np.ones((M,1))])

    # just solve for the first two columns of T
    U = uv

    # We know that X * T = U
    if matrix_rank(X)>=minRequiredNonCollinearPairs:
        if X.shape[0]==X.shape[1]:
            Tinv = linalg.solve(X, U)
        else:
            Tinv = linalg.lstsq(X,U)
    else:
        error('images:geotrans:requiredNonCollinearPoints '+str(minRequiredNonCollinearPairs)+'affine')

    # add third column
    Tinv[:,3] = [0,0,1]

    if normMatrix2.shape[0] == normMatrix2.shape[1]:
        Tinv = linalg.solve(normMatrix2,(Tinv @ normMatrix1))
    else:
        Tinv = linalg.lstsq(normMatrix2, (Tinv @ normMatrix1))

    T = inv(Tinv)
    T[:,3] = [0,0,1]

    tform = AffineTransform(T)
    return tform

def findProjectiveTransform(uv,xy):
    # For a projective transformation:
    #
    # u = (Ax + By + C)/(Gx + Hy + I)
    # v = (Dx + Ey + F)/(Gx + Hy + I)
    #
    # Assume I = 1, multiply both equations, by denominator:
    #
    # u = [x y 1 0 0 0 -ux -uy] * [A B C D E F G H]'
    # v = [0 0 0 x y 1 -vx -vy] * [A B C D E F G H]'
    #
    # With 4 or more correspondence points we can combine the u equations and
    # the v equations for one linear system to solve for [A B C D E F G H]:
    #
    # [ u1  ] = [ x1  y1  1  0   0   0  -u1*x1  -u1*y1 ] * [A]
    # [ u2  ] = [ x2  y2  1  0   0   0  -u2*x2  -u2*y2 ]   [B]
    # [ u3  ] = [ x3  y3  1  0   0   0  -u3*x3  -u3*y3 ]   [C]
    # [ u1  ] = [ x4  y4  1  0   0   0  -u4*x4  -u4*y4 ]   [D]
    # [ ... ]   [ ...                                  ]   [E]
    # [ un  ] = [ xn  yn  1  0   0   0  -un*xn  -un*yn ]   [F]
    # [ v1  ] = [ 0   0   0  x1  y1  1  -v1*x1  -v1*y1 ]   [G]
    # [ v2  ] = [ 0   0   0  x2  y2  1  -v2*x2  -v2*y2 ]   [H]
    # [ v3  ] = [ 0   0   0  x3  y3  1  -v3*x3  -v3*y3 ]
    # [ v4  ] = [ 0   0   0  x4  y4  1  -v4*x4  -v4*y4 ]
    # [ ... ]   [ ...                                  ]  
    # [ vn  ] = [ 0   0   0  xn  yn  1  -vn*xn  -vn*yn ]
    #
    # Or rewriting the above matrix equation:
    # U = X * Tvec, where Tvec = [A B C D E F G H]'
    # so Tvec = X\U.
    #
    
    [uv,normMatrix1] = normalizeControlPoints(uv)
    [xy,normMatrix2] = normalizeControlPoints(xy)
    
    minRequiredNonCollinearPairs = 4
    M = xy.shape[0]
    x = xy[:,0]
    y = xy[:,1]
    vec_1 = np.ones((M,1))
    vec_0 = np.zeros((M,1))
    u = uv[:,0]
    v = uv[:,1]
    
    U = np.vstack([u,v])
    
    X = [[x,      y,      vec_1,  vec_0,  vec_0,  vec_0,  -u*x,  -u*y],
         [vec_0,  vec_0,  vec_0,  x,      y,      vec_1,  -v*x,  -v*y] ]
    
    # We know that X * Tvec = U
    if matrix_rank(X) >= 2*minRequiredNonCollinearPairs:
        if X.shape[0]==X.shape[1]:
            Tvec = linalg.solve(X, U)
        else:
            Tvec = linalg.lstsq(X,U)
    else:
        error('images:geotrans:requiredNonCollinearPoints',+str(minRequiredNonCollinearPairs)+'projective')
    
    # We assumed I = 1
    Tvec[9] = 1
    
    Tinv = Tvec.reshape(3,3)

    if normMatrix2.shape[0] == normMatrix2.shape[1]:
        Tinv = linalg.solve(normMatrix2,(Tinv @ normMatrix1))
    else:
        Tinv = linalg.lstsq(normMatrix2, (Tinv @ normMatrix1))

    T = inv(Tinv)
    T = T / T[3,3]
    
    tform = ProjectiveTransform(T)

    return tform

    


