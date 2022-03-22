## ------------------------------------------------------------------------
#  OTMap2 Classs
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
# obj_OTMap = OTMap(obj_OTMap2.GetTable())

import os
import numpy as np
from utils.funcs import *

class OTMap2:
    # ------------------------------------------------------------------------
    # OTMap2 = OTMap2(initialData, varargin)
    # This class provides an interface to a key/value storage system.
    #
    # This class stores key value pairs as an 2xN array and performs
    # addition operations via intersection with new data. This approach can
    # be faster/more efficient than a simple 2XN array in some situations.
    #
    # See OTMap and OTTable.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # initialData -- An 2xN array of key (1) and value (2) pairs. Both must be
    # doubles. The key values do not need to be unique.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # None

    # -------------------------------------------------------------------------
    # Define constructor
    # -------------------------------------------------------------------------
    def __init__(self,initialData=[]):
        # obj = OTMap(initialData)
        # initialData is a 2XN array of key/value pairs

        self.data = {}  # Storage of the key/value pairs

        # -------------------------------------------------------------------------
        # Check input
        # -------------------------------------------------------------------------
        if len(initialData) < 1:
            initialData = np.zeros((2,0), dtype=np.double)

        if (initialData.dtype != np.double) or (initialData.shape[0] != 2):
            error('[Error]:invalidArguments - initialData must be a np.array(...,dtype=np.double) of size 2xN')

        # -------------------------------------------------------------------------
        # Find unique keys and accumulate values
        # -------------------------------------------------------------------------
        uniqueKeys, ia, ic= np.unique(initialData[0,:],return_index = True, return_inverse=True)
        values = np.bincount(ic,  weights=initialData[1,:])

        # -------------------------------------------------------------------------
        # Set data
        # -------------------------------------------------------------------------
        self.data = dict(zip(uniqueKeys,values))


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
            error('[Error]:invalidArguments - newData must be a np.array(...,dtype=np.double) of size 2xN')

        # -------------------------------------------------------------------------
        # Find overlapping values, sum where needed, and reassign
        # -------------------------------------------------------------------------
        [uniqueKeys, ia, ic] = np.unique(newData[0,:],return_index = True, return_inverse=True)
        values = np.bincount(ic, weights=newData[1, :])

        newData_dict = dict(zip(uniqueKeys,values))

        for k_i in newData_dict:
            if k_i in self.data:
                self.data[k_i] = self.data[k_i]  + newData_dict[k_i]
            else:
                self.data[k_i] = newData_dict[k_i]

    # -------------------------------------------------------------------------
    # Return values
    # -------------------------------------------------------------------------
    def GetValues(self, keys):
        # Return values for specified keys
        # values = obj.GetValues(keys)

        # -------------------------------------------------------------------------
        # Return values
        # -------------------------------------------------------------------------
        values = [self.data[k_i] if k_i in self.data else 0 for k_i in keys]

        return list(values)

    # -------------------------------------------------------------------------
    # Return Table
    # -------------------------------------------------------------------------
    def GetTable(self): # return data
        # Return the internal data array
        # data = obj.GetTable()
        keys = list(self.data.keys())
        values = list(self.data.values())
        return np.array([keys,values])

    def GetTable2(self): # return data
        # Return the internal data array
        # data = obj.GetTable()
        keys = list(self.data.keys())
        values = list(self.data.values())
        return dict(zip(keys, values))
    # -------------------------------------------------------------------------
    # Return keys
    # -------------------------------------------------------------------------
    def keys(self):
        # Return all keys from map
        # keys = obj.keys()

        return list(self.data.keys())

    # -------------------------------------------------------------------------
    # Return Values
    # -------------------------------------------------------------------------
    def values(self):
        # Return all values from map
        # values = obj.values()

        return list(self.data.values())

    # -------------------------------------------------------------------------
    # Return length
    # -------------------------------------------------------------------------
    def length(self):
        # Return the number of key/value pairs
        # numEntries = obj.length()
        # numEntries = len(obj)

        return len(self.data)
