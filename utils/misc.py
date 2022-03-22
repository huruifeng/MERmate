import bisect

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.path import Path
from scipy.sparse import isspmatrix
from scipy.signal import convolve2d, correlate,correlate2d
from skimage.morphology import reconstruction, disk, square, ball, cube

from utils.funcs import *

################################################
def is_prime(n):
  if n == 2 or n == 3: return True
  if n < 2 or n%2 == 0: return False
  if n < 9: return True
  if n%3 == 0: return False
  r = int(n**0.5)
  # since all primes > 3 are of the form 6n ± 1
  # start with f=5 (which is prime)
  # and test f, f+2 for being prime
  # then loop by 6.
  f = 5
  while f <= r:
    print('\t',f)
    if n % f == 0: return False
    if n % (f+2) == 0: return False
    f += 6
  return True


#################################################
def frref(A, TOL=None, TYPE=''):
    '''
    #FRREF   Fast reduced row echelon form.
    #   R = FRREF(A) produces the reduced row echelon form of A.
    #   [R,jb] = FRREF(A,TOL) uses the given tolerance in the rank tests.
    #   [R,jb] = FRREF(A,TOL,TYPE) forces frref calculation using the algorithm
    #   for full (TYPE='f') or sparse (TYPE='s') matrices.
    #
    #
    #   Description:
    #   For full matrices, the algorithm is based on the vectorization of MATLAB's
    #   RREF function. A typical speed-up range is about 2-4 times of
    #   the MATLAB's RREF function. However, the actual speed-up depends on the
    #   size of A. The speed-up is quite considerable if the number of columns in
    #   A is considerably larger than the number of its rows or when A is not dense.
    #
    #   For sparse matrices, the algorithm ignores the TOL value and uses sparse
    #   QR to compute the rref form, improving the speed by a few orders of
    #   magnitude.
    #
    #   Authors: Armin Ataei-Esfahani (2008)
    #            Ashish Myles (2012)
    #            Snehesh Shrestha (2020)
    #
    #   Revisions:
    #   25-Sep-2008   Created Function
    #   21-Nov-2012   Added faster algorithm for sparse matrices
    #   30-June-2020  Ported to python.
    #   TODO: Only do_full implemented. The remaining of the function. See frref_orig below.
    '''

    m = np.shape(A)[0]
    n = np.shape(A)[1]

    # Process Arguments
    # ----------------------------------------------------------
    # TYPE -- Sparce (s) or non-Sparce (Full, f)
    if TYPE == '':   # set TYPE if sparse or not
        if isspmatrix(A):
            TYPE = 's'
        else:
            TYPE = 'f'
    else:   # Set type
        if not type(TYPE) is str or len(TYPE) > 1:  # Check valid type
            print('Unknown matrix TYPE! Use "f" for full and "s" for sparse matrices.')
            exit()

        TYPE = TYPE.lower()
        if not TYPE == 'f' and not TYPE == 's':
            print(
                'Unknown matrix TYPE! Use ''f'' for full and ''s'' for sparse matrices.')
            exit()

    if TYPE=='f':
        # TOLERENCE
        # Compute the default tolerance if none was provided.
        if TOL is None:
            # Prior commit had TOL default to 1e-6
            # TOL = max(m,n)*eps(class(A))*norm(A,'inf')
            TOL = max(m, n)*np.spacing(type(A)(1))*np.linalg.norm(A, np.inf)

    # Non-Sparse
    # ----------------------------------------------------------
    if not isspmatrix(A) or TYPE == 'f':
        # Loop over the entire matrix.
        i = 0
        j = 0
        jb = []

        while (i < m) and (j < n):
            # Find value (p) and index (k) of largest element in the remainder of column j.
            abscol = np.array(np.abs(A[i:m, j]))
            p = np.max(abscol)
            k = np.argmax(abscol, axis=0)
            if np.ndim(k) > 1:
                k = k[0]
            else:
                k = int(k)

            k = k+i  # -1 #python zero index, not needed

            if p <= TOL:
                # The column is negligible, zero it out.
                A[i:m, j] = 0  #(faster for sparse) %zeros(m-i+1,1);
                j += 1
            else:
                # Remember column index
                jb.append(j)

                # Swap i-th and k-th rows.
                A = np.array(A)
                A[[i, k], j:n] = A[[k, i], j:n]

                # Divide the pivot row by the pivot element.
                Ai = np.nan_to_num(A[i, j:n] / A[i, j])
                Ai = np.matrix(Ai).T.T

                # Subtract multiples of the pivot row from all the other rows.
                A[:, j:n] = A[:, j:n] - np.dot(A[:, [j]], Ai)
                A[i, j:n] = Ai
                i += 1
                j += 1

        return A, jb

    # Sparse
    # ----------------------------------------------------------
    else:
        A = np.array(A.toarray())
        return frref(A, TYPE='f')

        # TODO: QR-decomposition of a Sparse matrix is not so simple in Python -- still need to figure out a solution
        # Non-pivoted Q-less QR decomposition computed by Matlab actually
        # produces the right structure (similar to rref) to identify independent
        # columns.
        # R = numpy.linalg.qr(A)

        # i_dep = pivot columns = dependent variables
        #       = left-most non-zero column (if any) in each row
        # indep_rows (binary vector) = non-zero rows of R
        # [indep_rows, i_dep] = np.max(R ~= 0, [], 2)     # TODO
        # indep_rows = full[indep_rows]; # probably more efficient
        # i_dep = i_dep[indep_rows]
        # i_indep = setdiff[1:n, i_dep]

        # solve R(indep_rows, i_dep) x = R(indep_rows, i_indep)
        #   to eliminate all the i_dep columns
        #   (i.e. we want F(indep_rows, i_dep) = Identity)
        # F = sparse([],[],[], m, n)
        # F[indep_rows, i_indep] = R[indep_rows, i_dep] \ R[indep_rows, i_indep]
        # F[indep_rows, i_dep] = speye(length(i_dep))

        # result
        # A = F
        # jb = i_dep

        # return A, jb

def de2bi(d, n):
    d=np.atleast_1d(d)
    power=2**np.arange(n)
    b = np.floor((d[...,None]%(2*power))/power)
    return b

def bi2de(b,direction="right"):
    b = list(b)
    if direction == "left":
        b=b[::-1]
    d = 0
    for i,b_i in enumerate(b):
        if int(b_i) > 1:
            error("[Error]: Bit in a binary string is larger than 1. Cannot convert it to a decimal number.")
        d += int(b_i)* (2**i)
    return d


import itertools
def nchoosek(a_ls,k, rep=False):
    if rep:
        all_combos = np.array(list(itertools.combinations_with_replacement(a_ls, k)))
    else:
        all_combos = np.array(list(itertools.combinations(a_ls, k)))
    return all_combos

def xcorr2(a,b,mode='full'):
    if mode == 'same':
        c = np.rot90(convolve2d(np.rot90(a, 2), np.rot90(b, 2), mode=mode), 2)
    else:
        c = convolve2d(a,np.rot90(np.conj(b),2))
    return c

def xcorr2_bkp(a,b,mode='full'):
    c = correlate2d(a,b,mode=mode)
    return c

def sub2ind(array_shape, rows, cols,order="C"):
    if order=="F":
        ind = cols*array_shape[0] + rows
    else:
        ind = rows * array_shape[1] + cols
    return ind

def ind2sub(array_shape, ind,order="C"):
    if order=="F":
        ind = np.asarray(ind)
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
        cols = (ind.astype('int') // array_shape[0])
        rows = ind % array_shape[0]
    else:
        ind = np.asarray(ind)
        ind[ind < 0] = -1
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        rows = (ind.astype('int') // array_shape[1])
        cols = ind % array_shape[1]
    return (rows, cols)


def ind2sub3d(array_shape3d, ind,order="C"):
    if len(array_shape3d) == 3:
        array_shape = array_shape3d[0:2]
        zpos = ind // (array_shape[0]*array_shape[1])
        ind = ind - zpos * (array_shape[0]*array_shape[1])
    else:
        array_shape = array_shape3d
        zpos = 0
    if order=="F":
        ind = np.asarray(ind)
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
        cols = (ind.astype('int') // array_shape[0])
        rows = ind % array_shape[0]
    else:
        ind = np.asarray(ind)
        ind[ind < 0] = -1
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        rows = (ind.astype('int') // array_shape[1])
        cols = ind % array_shape[1]
    return (rows, cols,zpos)[:len(array_shape3d)]


def Ncolor(I,*args):
    # ------------------------------------------------------------------------
    # Io = Ncolor(I);
    #                        -- Returns a 3D RGB matrix which has mapped each
    #                        of the z-dimensions of the input image to a
    #                        different color in RGB space.
    # Io = Ncolor(I,cMap);
    #                        -- Convert the N layer matrix I into an RGB image
    #                           according to colormap, 'cMap'.
    #------------------------------------------------------------------------
    # Inputs
    # I double / single / uint16 or uint8,
    #                       -- HxWxN where each matrix I(:,:,n) is
    #                          to be assigned a different color.
    # cMap
    #                       -- a valid matlab colormap. leave blank for default
    #                       hsv (which is RGB for N=3).  Must be Nx3
    #-------------------------------------------------------------------------
    # Outputs
    # Io same class as I.
    #                       -- HxWx3 RGB image
    #--------------------------------------------------------------------------
    # Alistair Boettiger
    # boettiger.alistair@gmail.com
    # October 10th, 2012
    #
    # Version 1.0
    #--------------------------------------------------------------------------
    # Creative Commons License 3.0 CC BY
    #--------------------------------------------------------------------------

    clrmap = []
    if len(args) == 1:
        clrmap = args[0]
        clrmap_obj = args[0]
        if isinstance(clrmap_obj, (matplotlib.colors.ListedColormap,matplotlib.colors.LinearSegmentedColormap)):
            clrmap = cm.get_cmap(clrmap_obj.name)(np.linspace(0, 1,clrmap_obj.N))
    elif len(args) > 1:
        error('wrong number of inputs')

    h,w,numColors = I.shape

    if isinstance(clrmap,(list,np.ndarray)) and len(clrmap)==0 and numColors ==1:
        clrmap = cm.get_cmap('jet')(np.linspace(0, 1, 256))
        clrmap_obj = cm.get_cmap('jet',256)
    elif isinstance(clrmap,(list,np.ndarray)) and len(clrmap)==0 and numColors < 10:
        clrmap =cm.get_cmap('hsv')(np.linspace(0, 1, numColors))
        clrmap_obj = cm.get_cmap('hot', numColors)

    if numColors == 1 and isinstance(clrmap,np.ndarray) and clrmap.shape[0] > 10:
        Io = I
    else:
        if isinstance(clrmap,str):
            try:
                clrmap = cm.get_cmap(clrmap)(np.linspace(0, 1, numColors))
                clrmap_obj = cm.get_cmap(clrmap, numColors)
            except Exception as e:
                print(clrmap,' is not a valid colormap name')

        Io = np.zeros((h,w,3),dtype=np.float)
        try:
            for c in range(numColors):
                for cc in [0,1,2]:
                    Io[:,:,cc] = Io[:,:,cc] + I[:,:,c]*clrmap[c,cc]
        except Exception as e:
            print(e)
            error('[Error]:Error running Ncolor')

    return Io,clrmap_obj

def StripWords(words):
    new_wrods= []
    fieldsToKeep = ['intCodeword', 'geneName', 'isExactMatch', 'isCorrectedMatch',
                    'imageX', 'imageY', 'wordCentroidX', 'wordCentroidY', 'cellID']
    for w_i in words:
        new_w_i = {}
        for f_i in fieldsToKeep:
            if f_i in w_i:
                new_w_i[f_i] = w_i[f_i]
        new_wrods.append(new_w_i)
    return new_wrods

_parula_data  = [[0.2422, 0.1504, 0.6603],[0.2444, 0.1534, 0.6728],[0.2464, 0.1569, 0.6847],
[0.2484, 0.1607, 0.6961],[0.2503, 0.1648, 0.7071],[0.2522, 0.1689, 0.7179],[0.254, 0.1732, 0.7286],
[0.2558, 0.1773, 0.7393],[0.2576, 0.1814, 0.7501],[0.2594, 0.1854, 0.761],[0.2611, 0.1893, 0.7719],
[0.2628, 0.1932, 0.7828],[0.2645, 0.1972, 0.7937],[0.2661, 0.2011, 0.8043],[0.2676, 0.2052, 0.8148],
[0.2691, 0.2094, 0.8249],[0.2704, 0.2138, 0.8346],[0.2717, 0.2184, 0.8439],[0.2729, 0.2231, 0.8528],
[0.274, 0.228, 0.8612],[0.2749, 0.233, 0.8692],[0.2758, 0.2382, 0.8767],[0.2766, 0.2435, 0.884],
[0.2774, 0.2489, 0.8908],[0.2781, 0.2543, 0.8973],[0.2788, 0.2598, 0.9035],[0.2794, 0.2653, 0.9094],
[0.2798, 0.2708, 0.915],[0.2802, 0.2764, 0.9204],[0.2806, 0.2819, 0.9255],[0.2809, 0.2875, 0.9305],
[0.2811, 0.293, 0.9352],[0.2813, 0.2985, 0.9397],[0.2814, 0.304, 0.9441],[0.2814, 0.3095, 0.9483],
[0.2813, 0.315, 0.9524],[0.2811, 0.3204, 0.9563],[0.2809, 0.3259, 0.96],[0.2807, 0.3313, 0.9636],
[0.2803, 0.3367, 0.967],[0.2798, 0.3421, 0.9702],[0.2791, 0.3475, 0.9733],[0.2784, 0.3529, 0.9763],
[0.2776, 0.3583, 0.9791],[0.2766, 0.3638, 0.9817],[0.2754, 0.3693, 0.984],[0.2741, 0.3748, 0.9862],
[0.2726, 0.3804, 0.9881],[0.271, 0.386, 0.9898],[0.2691, 0.3916, 0.9912],[0.267, 0.3973, 0.9924],
[0.2647, 0.403, 0.9935],[0.2621, 0.4088, 0.9946],[0.2591, 0.4145, 0.9955],[0.2556, 0.4203, 0.9965],
[0.2517, 0.4261, 0.9974],[0.2473, 0.4319, 0.9983],[0.2424, 0.4378, 0.9991],[0.2369, 0.4437, 0.9996],
[0.2311, 0.4497, 0.9995],[0.225, 0.4559, 0.9985],[0.2189, 0.462, 0.9968],[0.2128, 0.4682, 0.9948],
[0.2066, 0.4743, 0.9926],[0.2006, 0.4803, 0.9906],[0.195, 0.4861, 0.9887],[0.1903, 0.4919, 0.9867],
[0.1869, 0.4975, 0.9844],[0.1847, 0.503, 0.9819],[0.1831, 0.5084, 0.9793],[0.1818, 0.5138, 0.9766],
[0.1806, 0.5191, 0.9738],[0.1795, 0.5244, 0.9709],[0.1785, 0.5296, 0.9677],[0.1778, 0.5349, 0.9641],
[0.1773, 0.5401, 0.9602],[0.1768, 0.5452, 0.956],[0.1764, 0.5504, 0.9516],[0.1755, 0.5554, 0.9473],
[0.174, 0.5605, 0.9432],[0.1716, 0.5655, 0.9393],[0.1686, 0.5705, 0.9357],[0.1649, 0.5755, 0.9323],
[0.161, 0.5805, 0.9289],[0.1573, 0.5854, 0.9254],[0.154, 0.5902, 0.9218],[0.1513, 0.595, 0.9182],
[0.1492, 0.5997, 0.9147],[0.1475, 0.6043, 0.9113],[0.1461, 0.6089, 0.908],[0.1446, 0.6135, 0.905],
[0.1429, 0.618, 0.9022],[0.1408, 0.6226, 0.8998],[0.1383, 0.6272, 0.8975],[0.1354, 0.6317, 0.8953],
[0.1321, 0.6363, 0.8932],[0.1288, 0.6408, 0.891],[0.1253, 0.6453, 0.8887],[0.1219, 0.6497, 0.8862],
[0.1185, 0.6541, 0.8834],[0.1152, 0.6584, 0.8804],[0.1119, 0.6627, 0.877],[0.1085, 0.6669, 0.8734],
[0.1048, 0.671, 0.8695],[0.1009, 0.675, 0.8653],[0.0964, 0.6789, 0.8609],[0.0914, 0.6828, 0.8562],
[0.0855, 0.6865, 0.8513],[0.0789, 0.6902, 0.8462],[0.0713, 0.6938, 0.8409],[0.0628, 0.6972, 0.8355],
[0.0535, 0.7006, 0.8299],[0.0433, 0.7039, 0.8242],[0.0328, 0.7071, 0.8183],[0.0234, 0.7103, 0.8124],
[0.0155, 0.7133, 0.8064],[0.0091, 0.7163, 0.8003],[0.0046, 0.7192, 0.7941],[0.0019, 0.722, 0.7878],
[0.0009, 0.7248, 0.7815],[0.0018, 0.7275, 0.7752],[0.0046, 0.7301, 0.7688],[0.0094, 0.7327, 0.7623],
[0.0162, 0.7352, 0.7558],[0.0253, 0.7376, 0.7492],[0.0369, 0.74, 0.7426],[0.0504, 0.7423, 0.7359],
[0.0638, 0.7446, 0.7292],[0.077, 0.7468, 0.7224],[0.0899, 0.7489, 0.7156],[0.1023, 0.751, 0.7088],
[0.1141, 0.7531, 0.7019],[0.1252, 0.7552, 0.695],[0.1354, 0.7572, 0.6881],[0.1448, 0.7593, 0.6812],
[0.1532, 0.7614, 0.6741],[0.1609, 0.7635, 0.6671],[0.1678, 0.7656, 0.6599],[0.1741, 0.7678, 0.6527],
[0.1799, 0.7699, 0.6454],[0.1853, 0.7721, 0.6379],[0.1905, 0.7743, 0.6303],[0.1954, 0.7765, 0.6225],
[0.2003, 0.7787, 0.6146],[0.2061, 0.7808, 0.6065],[0.2118, 0.7828, 0.5983],[0.2178, 0.7849, 0.5899],
[0.2244, 0.7869, 0.5813],[0.2318, 0.7887, 0.5725],[0.2401, 0.7905, 0.5636],[0.2491, 0.7922, 0.5546],
[0.2589, 0.7937, 0.5454],[0.2695, 0.7951, 0.536],[0.2809, 0.7964, 0.5266],[0.2929, 0.7975, 0.517],
[0.3052, 0.7985, 0.5074],[0.3176, 0.7994, 0.4975],[0.3301, 0.8002, 0.4876],[0.3424, 0.8009, 0.4774],
[0.3548, 0.8016, 0.4669],[0.3671, 0.8021, 0.4563],[0.3795, 0.8026, 0.4454],[0.3921, 0.8029, 0.4344],
[0.405, 0.8031, 0.4233],[0.4184, 0.803, 0.4122],[0.4322, 0.8028, 0.4013],[0.4463, 0.8024, 0.3904],
[0.4608, 0.8018, 0.3797],[0.4753, 0.8011, 0.3691],[0.4899, 0.8002, 0.3586],[0.5044, 0.7993, 0.348],
[0.5187, 0.7982, 0.3374],[0.5329, 0.797, 0.3267],[0.547, 0.7957, 0.3159],[0.5609, 0.7943, 0.305],
[0.5748, 0.7929, 0.2941],[0.5886, 0.7913, 0.2833],[0.6024, 0.7896, 0.2726],[0.6161, 0.7878, 0.2622],
[0.6297, 0.7859, 0.2521],[0.6433, 0.7839, 0.2423],[0.6567, 0.7818, 0.2329],[0.6701, 0.7796, 0.2239],
[0.6833, 0.7773, 0.2155],[0.6963, 0.775, 0.2075],[0.7091, 0.7727, 0.1998],[0.7218, 0.7703, 0.1924],
[0.7344, 0.7679, 0.1852],[0.7468, 0.7654, 0.1782],[0.759, 0.7629, 0.1717],[0.771, 0.7604, 0.1658],
[0.7829, 0.7579, 0.1608],[0.7945, 0.7554, 0.157],[0.806, 0.7529, 0.1546],[0.8172, 0.7505, 0.1535],
[0.8281, 0.7481, 0.1536],[0.8389, 0.7457, 0.1546],[0.8495, 0.7435, 0.1564],[0.86, 0.7413, 0.1587],
[0.8703, 0.7392, 0.1615],[0.8804, 0.7372, 0.165],[0.8903, 0.7353, 0.1695],[0.9, 0.7336, 0.1749],
[0.9093, 0.7321, 0.1815],[0.9184, 0.7308, 0.189],[0.9272, 0.7298, 0.1973],[0.9357, 0.729, 0.2061],
[0.944, 0.7285, 0.2151],[0.9523, 0.7284, 0.2237],[0.9606, 0.7285, 0.2312],[0.9689, 0.7292, 0.2373],
[0.977, 0.7304, 0.2418],[0.9842, 0.733, 0.2446],[0.99, 0.7365, 0.2429],[0.9946, 0.7407, 0.2394],
[0.9966, 0.7458, 0.2351],[0.9971, 0.7513, 0.2309],[0.9972, 0.7569, 0.2267],[0.9971, 0.7626, 0.2224],
[0.9969, 0.7683, 0.2181],[0.9966, 0.774, 0.2138],[0.9962, 0.7798, 0.2095],[0.9957, 0.7856, 0.2053],
[0.9949, 0.7915, 0.2012],[0.9938, 0.7974, 0.1974],[0.9923, 0.8034, 0.1939],[0.9906, 0.8095, 0.1906],
[0.9885, 0.8156, 0.1875],[0.9861, 0.8218, 0.1846],[0.9835, 0.828, 0.1817],[0.9807, 0.8342, 0.1787],
[0.9778, 0.8404, 0.1757],[0.9748, 0.8467, 0.1726],[0.972, 0.8529, 0.1695],[0.9694, 0.8591, 0.1665],
[0.9671, 0.8654, 0.1636],[0.9651, 0.8716, 0.1608],[0.9634, 0.8778, 0.1582],[0.9619, 0.884, 0.1557],
[0.9608, 0.8902, 0.1532],[0.9601, 0.8963, 0.1507],[0.9596, 0.9023, 0.148],[0.9595, 0.9084, 0.145],
[0.9597, 0.9143, 0.1418],[0.9601, 0.9203, 0.1382],[0.9608, 0.9262, 0.1344],[0.9618, 0.932, 0.1304],
[0.9629, 0.9379, 0.1261],[0.9642, 0.9437, 0.1216],[0.9657, 0.9494, 0.1168],[0.9674, 0.9552, 0.1116],
[0.9692, 0.9609, 0.1061],[0.9711, 0.9667, 0.1001],[0.973, 0.9724, 0.0938],[0.9749, 0.9782, 0.0872],
[0.9769, 0.9839, 0.0805]]

from matplotlib.colors import LinearSegmentedColormap
parula_map = LinearSegmentedColormap.from_list('parula', _parula_data)

def fgauss2D(size=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[size],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in size]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def conndef(num_dims, conntype):
    if (not isinstance(num_dims,int)) or num_dims <= 0:
        error ("Function: conndef - number of dimensions must be a positive integer.")
    if not isinstance(conntype,str) or conntype not in ["max","min"]:
        error ("Function: conndef - conntype argument must be a string with type of connectivity: ['maximal','minimal'].")

    ## support for 1 dimension does not exist in Matlab where num_dims must be >= 2
    if (num_dims == 1):
        conn = np.array([1, 1, 1]);
    ## matlab is case insensitive when checking the type
    elif (conntype == "min"):
        if (num_dims == 2):
            conn = np.array([[0,1,0],[1,1,1],[0,1,0]])
        else:
            shape = np.tile(3, (num_dims,))
            conn = np.zeros(shape)
            conn = conn.flatten()
            conn[int((conn.size+ 1) / 2-1)] = 1
            stride = [3. ** i for i in range(num_dims)]
            center = np.sum(stride)
            for k in range(num_dims):
                conn[int(center - stride[k])] = 1
                conn[int(center + stride[k])] = 1

            conn = conn.reshape(shape)

    elif conntype == "max":
        conn = np.ones(np.tile(3, (num_dims,)))
    else:
        error("invalid type of connectivity:", conntype)

    return conn

def sphere(r):
    ## Same as skimage.morphology.ball
    [x,y,z] = np.meshgrid(np.arange(-r,r+1),np.arange(-r,r+1),np.arange(-r,r+1))
    nhood =  ( (x/r)*2 + (y/r)*2 + (z/r)*2 ) <= 1
    return nhood


## https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python/44529776#44529776
def imadjust(src, tol=1, vin=[0,65535], vout=(0,65535)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(65536)),range=(0,65535))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    if vin[1]==vin[0]:
        scale = 1
    else:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst

def imadjust_x(x,a=0,b=65535,c=0,d=65535,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def imposemin(img, map):
    marker = np.full(img.shape, np.inf)
    marker[map == 1] = 0
    mask = np.minimum((img + 1), marker)
    return reconstruction(marker, mask, method='erosion')

def imimposemin(I, BW, conn=None, max_value=255):
    if not I.ndim in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")

    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn == None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW]                 = -np.inf
        fm[np.logical_not(BW)] = np.inf
    except:
        fm[BW]                 = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J

    try:
        J[BW] = -np.inf
    except:
        J[BW] = -float("inf")

    return J

## https://github.com/AILab121/OOSE/blob/3b50772a2f7ee7f02e048f462c4779840b762ef8/src/mir_robot/robot5g/Utils/Inpolygon.py
# Matlab原始版 inpolygon函數
def inpolygon_Matlab(x_point, y_point, x_area, y_area):
    """
    Reimplement inpolygon in matlab
    :type x_point: np.ndarray
    :type y_point: np.ndarray
    :type x_area: np.ndarray
    :type y_area: np.ndarray
    """
    # 合併xv和yv為頂點數組
    vertices = np.vstack((x_area, y_area)).T
    # 定義Path對象
    path = Path(vertices)
    # 把xq和yq合併為test_points
    test_points = np.hstack([x_point.reshape(x_point.size, -1), y_point.reshape(y_point.size, -1)])
    # 得到一個test_points是否嚴格在path內的mask，是bool值數組
    _in = path.contains_points(test_points)
    # 得到一個test_points是否在path內部或者在路徑上的mask
    _in_on = path.contains_points(test_points, radius=-1e-10)
    # 得到一個test_points是否在path路徑上的mask
    _on = _in ^ _in_on
    return _in_on, _on


# 簡化inpolygon函數
def inpolygon(xq, yq, xv, yv):
    """
    reimplement inpolygon in matlab
    :type xq: np.ndarray
    :type yq: np.ndarray
    :type xv: np.ndarray
    :type yv: np.ndarray
    """
    # 合併xv和yv為頂點數組
    vertices = np.vstack((xv, yv)).T
    # 定義Path對象
    path = Path(vertices)
    # 把xq和yq合併為test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # 得到一個test_points是否嚴格在path內的mask，是bool值數組
    _in = path.contains_points(test_points)
    # 得到一個test_points是否在path內部或者在路徑上的mask
    _in_on = path.contains_points(test_points, radius=-1e-10)

    return _in_on

## https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_area(x,y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)

def polygon_area_maxb(x,y):
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)


















