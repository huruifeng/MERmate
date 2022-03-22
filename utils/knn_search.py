from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

def knnsearch_a(x, y, k=1):
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

def knnsearch_b(A, B, k=1):
    indices = np.empty((len(B), k))
    distances = np.empty((len(B), k))
    for i,b in enumerate(B):
        dif = np.abs(A - b)
        min_ind = np.argpartition(dif.T,k-1)[:k] # Returns the indexes of the n smallest numbers but not necessarily sorted
        ind = min_ind[np.argsort(dif[min_ind])] # sort output of argpartition just in case
        indices[i, :] = ind
        distances[i, :] = dif[ind]
    return indices, distances


