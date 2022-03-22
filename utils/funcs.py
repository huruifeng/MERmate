import os
import sys
import datetime
from copy import deepcopy

import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS as tifTAGS

from scipy import interpolate
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def PageBreak(s="=",n=80):
    print(s*n)

def ParseArguments(dict_obj={}, arg_names=[]):
    args = dict_obj
    ## initialize the values:
    for arg_name_i in arg_names:
        if arg_name_i in dict_obj:
            args[arg_name_i] = dict_obj[arg_name_i]
        else:
            args[arg_name_i] = ""
    return args


## print current time in a format: 2021-09-20 10:20:15
def tic(flag=99):
    now = datetime.datetime.now()
    if flag == 0:
        return now
    elif flag == 1:
        return (now.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print('Start at: ', now.strftime("%Y-%m-%d %H:%M:%S"))
        return now


def toc(st, flag=99):
    now = datetime.datetime.now()
    return (now - st).total_seconds()


## read sequences from FASTA file
def fastaread(fasta_file):
    if not os.path.exists(fasta_file):
        raise ("Error: Fasta file not exist:" + str(fasta_file))

    fp = open(fasta_file)
    fasta_dict = {}
    seq_i = ""

    for line in fp:
        if line[0] == '>':  # a new FASTA record starts
            if seq_i != "":  # check if the record is the first one
                # save the previous one record
                fasta_dict[header_i] = seq_i
                seq_i = ""

            header_i = line.strip()[1:]
        else:
            seq_i = seq_i + line.strip()

    # the last fasta record
    fasta_dict[header_i] = seq_i

    ## One alternative solution:
    # for line in fp:
    #     if line[0] == '>' and seq_i == '':
    #         header_i = line.strip()
    #     elif line[0] != '>':
    #         seq = seq + line.strip()
    #     elif line[0] == '>' and seq_i != '':
    #         fasta_dict[header_i] = seq_i
    #
    #         header_i = line.strip()
    #         seq = ''
    #
    # fasta_dict[header_i] = seq_i

    fp.close()

    return fasta_dict


## Write a FASTA dict into file
def fastawrite(fasta_file, fasta_dict, mode="w"):
    wrap = 70
    with open(fasta_file, mode) as fp:
        for header, seq in fasta_dict.items():
            fp.write(">" + header + "\n")
            for i in range(0, len(seq), wrap):
                fp.write(f'{seq[i:i + wrap]}\n')


## convert A,C,G,T to 1,2,3,4
def nt2int(seq, ACGT=False):
    int_ls = []
    if len(seq) == 0:
        error("Error: Sequence length is 0, cannot be converted to Int")
    else:
        int_ls = []
        for s_i in seq:
            if s_i.upper() == "A":
                int_ls.append(1)
            elif s_i.upper() == "C":
                int_ls.append(2)
            elif s_i.upper() == "G":
                int_ls.append(3)
            elif s_i.upper() == "T":
                int_ls.append(4)
            elif s_i.upper() == "U":
                int_ls.append(4)
            else:
                if ACGT:
                    int_ls.append(0)
                else:
                    int_ls.append(9)
    return int_ls

## convert A,C,G,T to 1,2,3,4
def int2nt(seq, ACGT=False):
    nt_str = ""
    if len(seq) == 0:
        error("Error: Sequence length is 0, cannot be converted to seq_string")
    else:
        for s_i in seq:
            if s_i == 1:
                nt_str += "A"
            elif s_i == 2:
                nt_str += "C"
            elif s_i == 3:
                nt_str += "G"
            elif s_i == 4:
                nt_str += "T"
            else:
                error("Error: Invalid sequence input, cannot be converted to seq_string")
    return nt_str


## Error: report error
def error(msg):
    print(msg)
    raise ("Running halted")

def seqrcomplement(seq):
    res_str = ""
    for n in seq:
        if n.upper() == "A":
            res_str += "T"
        elif n.upper() == "C":
            res_str += "G"
        elif n.upper() == "G":
            res_str += "C"
        elif n.upper() == "T":
            res_str += "A"
        elif n.upper() == "U":
            res_str += "A"
        else:
            res_str += "N"
    return res_str[::-1]

## MATLAB [~,I] = histc(x,binrange)
def histc(x, binranges):
  indices = np.searchsorted(binranges, x)
  return np.mod(indices+1, len(binranges)+1)



def knnsearch_loop(R, Q, k=1):
    """
    KNNSEARCH   Linear k-nearest neighbor (KNN) search
    IDX = knnsearch(R,Q,K) searches the reference data set R (n x d array
    representing n points in a d-dimensional space) to find the k-nearest
    neighbors of each query point represented by eahc row of Q (m x d array).
    The results are stored in the (m x K) index array, IDX.

    Rationality
    Linear KNN search is the simplest appraoch of KNN. The search is based on
    calculation of all distances. Therefore, it is normally believed only
    suitable for small data sets. However, other advanced approaches, such as
    kd-tree and delaunary become inefficient when d is large comparing to the
    number of data points.
    %
    See also, kdtree, nnsearch, delaunary, dsearch
    Original By Yi Cao at Cranfield University on 25 March 2008
    Modified: Ruifeng HU, 203092022,BWH Boston
    """

    N, M = Q.shape
    idx = np.zeros((N, k), dtype=int)
    D = np.zeros((N, k))
    if k == 1:
        for i in range(0, N):
            d = np.sqrt(np.sum((Q[i, :] - R[:, :]) ** 2, axis=1))
            D[i,:] = np.min(d)
            idx[i,:] = np.argmin(d)
    else:
        for i in range(0, N):
            d = np.sqrt(np.sum((Q[i, :] - R[:, :]) ** 2, axis=1))
            D[i, :] = np.sort(d)[:k]
            idx[i, :] = np.argsort(d)[:k]
    # print("==>Nearest neighbour search completed!")
    return idx, D


# Locate the most similar neighbors
def knnsearch2d(A, B, k=1):
    D = cdist(A, B)  ## high memory consuming, if A, B is large
    k=min(k,D.shape[0])
    if k == 1:
        k_i = D.argmin(0).reshape((1,-1))
    else:
        if k == D.shape[0]:
            k_i = D.argpartition(k-1, axis=0)[:k]
        else:
            k_i = D.argpartition(k, axis=0)[:k]

    k_d = np.take_along_axis(D, k_i, axis=0)

    sorted_idx = k_d.argsort(axis=0)
    k_i_sorted = np.take_along_axis(k_i, sorted_idx, axis=0)
    k_d_sorted = np.take_along_axis(k_d, sorted_idx, axis=0)

    return k_i_sorted, k_d_sorted

def knnsearch_ckdtree(x, y, k=1):
    """
    Search k neighbor for similar patches.
    """
    YourTreeName = cKDTree(x)
    id_x, dist_x = [], []
    for item in y:
        dist, id = YourTreeName.query(item, k)
        id_x.append(id)
        dist_x.append(dist)
    return np.array(id_x), np.array(dist_x)

def weighted_avg_and_var(values,axis=None, weights=None):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=axis, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2,axis=axis, weights=weights)
    return (average, variance)

def weighted_nanvar(values,axis=None, weights=None):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    temp_a = values
    ma = np.ma.MaskedArray(temp_a, mask=np.isnan(temp_a))
    weighted_average_nan = np.ma.average(ma, axis=axis, weights=weights)

    temp_a = temp_a - weighted_average_nan
    ma = np.ma.MaskedArray(temp_a, mask=np.isnan(temp_a))
    variance = np.average(ma ** 2, axis=axis, weights=weights)
    return variance

def tiffimginfo(tiffFileName):
    info_ls = []
    im = Image.open(tiffFileName)
    ## next
    try:
        n = 0
        while 1:
            im.seek(n)
            exifdata = im.getexif()
            tiffInfo = {}
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = tifTAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                # decode bytes
                if isinstance(data, bytes):
                    data = data.decode()
                tiffInfo[tag] = data
            info_ls.append(tiffInfo)
            n += 1
    except EOFError:
        pass

    im.close()

    return info_ls

def bitFlip(x,n):
    b = 2**n
    r = x^b
    return r



def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    if len(D.shape)==1:
        m = D.shape[0]
        n=1
    else:
        m,n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C


def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    if len(good[0]) == 0:
        return np.nan_to_num(A)
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B
