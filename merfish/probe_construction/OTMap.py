## ------------------------------------------------------------------------
#  OTMap Classs
## ------------------------------------------------------------------------
# Original version:
# Jeffrey Moffitt
# lmoffitt@mcb.harvard.edu
# jeffrey.moffitt@childrens.harvard.edu
# April 27, 2015
#--------------------------------------------------------------------------
# Copyright Presidents and Fellows of Harvard College, 2016.
#--------------------------------------------------------------------------

# --------------------------------------------------------------------------
# This python version is developed by Ruifeng Hu from the Original version
# 09-20-2021
# huruifeng.cn@hotmail.com
# --------------------------------------------------------------------------

## OTMap can be converted to OTMap2:
# obj_OTMap2 = OTMap2(obj_OTMap.GetTable())


import os
import numpy as np
from utils.funcs import *

class OTMap:
    # ------------------------------------------------------------------------
    # OTMap = OTMap(initialData, varargin)
    # This class provides an interface to a key/value storage system.
    #
    # This class stores key value pairs as an 2xN array and performs
    # addition operations via intersection with new data. This approach can
    # be faster/more efficient than that utilized by OTMap2 in some situations.
    #
    # See OTMap2 and OTTable.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # initialData -- An 2xN array of key (1) and value (2) pairs. Both must be
    # doubles. The key values do not need to be unique.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # None

    # -------------------------------------------------------------------------
    # Define properties
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Define constructor
    # -------------------------------------------------------------------------
    def __init__(self,initialData=[]):
        # obj = OTMap(initialData)
        # initialData is a 2XN array of key/value pairs

        self.data = np.zeros((2,0), dtype=np.double)  # Storage of the key/value pairs

        # -------------------------------------------------------------------------
        # Check input
        # -------------------------------------------------------------------------
        if len(initialData) < 1:
            initialData = np.zeros((2,0), dtype=np.double)

        if (initialData.dtype != np.double) or (initialData.shape[0] != 2):
            error('Error:invalidArguments - initialData must be a np.array(...,dtype=np.double) of size 2xN')

        # -------------------------------------------------------------------------
        # Find unique keys and accumulate values
        # -------------------------------------------------------------------------
        uniqueKeys, ia, ic= np.unique(initialData[0,:],return_index = True, return_inverse=True)
        values = np.bincount(ic,  weights=initialData[1,:])

        # -------------------------------------------------------------------------
        # Set data
        # -------------------------------------------------------------------------
        self.data = np.array([uniqueKeys,values],dtype=np.double)


    # -------------------------------------------------------------------------
    # AddToMap
    # -------------------------------------------------------------------------
    def AddToMap(self, newData):
        # Add additional key/value pairs to an existing class
        # obj.AddToMap(newData)

        # -------------------------------------------------------------------------
        # Check data
        # -------------------------------------------------------------------------
        if (newData.dtype != np.double) or (newData.shape[0] != 2):
            error('Error:invalidArguments - newData must be a np.array(...,dtype=np.double) of size 2xN')

        # -------------------------------------------------------------------------
        # Find overlapping values, sum where needed, and reassign
        # -------------------------------------------------------------------------
        merged_data = np.concatenate((self.data, newData), axis=1)

        [uniqueKeys, ia, ic] = np.unique(merged_data[0,:],return_index = True, return_inverse=True)
        values = np.bincount(ic, weights=merged_data[1, :])

        self.data = np.array([uniqueKeys, values])


    # -------------------------------------------------------------------------
    # Return values
    # -------------------------------------------------------------------------
    def GetValues(self, keys):
        # Return values for specified keys
        # values = obj.GetValues(keys)

        # -------------------------------------------------------------------------
        # Prepare output
        # -------------------------------------------------------------------------
        values = np.zeros(len(keys),dtype=np.double)

        # -------------------------------------------------------------------------
        # Find keys
        # -------------------------------------------------------------------------
        [c, i1, i2] = np.intersect1d(self.data[0,:], keys,return_indices=True)

        # -------------------------------------------------------------------------
        # Return values
        # -------------------------------------------------------------------------
        values[i2] = self.data[1,i1]

        return values


    # -------------------------------------------------------------------------
    # Return Table
    # -------------------------------------------------------------------------
    def GetTable(self): # return data
        # Return the internal data array
        # data = obj.GetTable()
        return self.data

    # -------------------------------------------------------------------------
    # Return keys
    # -------------------------------------------------------------------------
    def keys(self):
        # Return all keys from map
        # keys = obj.keys()

        return self.data[0,:]

    # -------------------------------------------------------------------------
    # Return Values
    # -------------------------------------------------------------------------
    def values(self):
        # Return all values from map
        # values = obj.values()

        return self.data[1,:]

    # -------------------------------------------------------------------------
    # Return length
    # -------------------------------------------------------------------------
    def length(self):
        # Return the number of key/value pairs
        # numEntries = obj.length()

        return self.data.shape[1]
