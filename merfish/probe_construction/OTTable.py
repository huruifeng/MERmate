## ------------------------------------------------------------------------
#  OTTable Classs
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

import os
import numpy as np
import pickle
from scipy.signal import lfilter
from pathos.multiprocessing import ProcessPool as Pool
from utils.funcs import *
from .Transcritome import Transcriptome
from .OTMap import OTMap
from .OTMap2 import OTMap2


def do_hash_build_func(targetSequences_splited_i,hashBase,seedLength,weights):
    # Create separate key/value matrix for each worker
    data_tmp = OTMap()
    targetSequences_splited_i_dict = dict(targetSequences_splited_i)
    for seq_i in targetSequences_splited_i_dict:
        localSeq = np.array(nt2int(targetSequences_splited_i_dict[seq_i]), dtype=np.double) - 1
        hash = lfilter(hashBase, 1, localSeq)
        hash = hash[seedLength - 1:]

        isValid = lfilter(np.ones(seedLength) / seedLength, 1, ((localSeq > 3) | (localSeq < 0)))
        isValid = np.logical_not(np.array(isValid[seedLength - 1:], dtype=bool))  # Kludge...

        data_tmp.AddToMap(np.concatenate(([hash[isValid]], [np.zeros(sum(isValid)) + weights[seq_i]]), axis=0))
    return data_tmp

class OTTable:
    # ------------------------------------------------------------------------
    # [otTable, parameters] = OTTable(targetSequences, seedLength, **varargin)
    # This class returns a look up table that contains the penalty
    # assigned to each seed sequence based on the frequency with which it appears
    # in a set of off-target sequences. A seed sequence is every unique n-mer
    # (where n is defined by the seedLength).
    #
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # targetSequences -- A array of sequences in the fasta structure format,
    #   i.e. each element needs a Header and a Sequence entry. Alternatively,
    #	this can be a Transcriptome object or a list of sequences.
    # seedLength -- the length of the seed sequences
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #  -- weights: An array with a weight to apply to each of the sequences in
    #     targetSequences. The default is 1. These values can adjust for the
    #     relative abundance of each of the targetSequences.
    #--------------------------------------------------------------------------

    ###################################################################################
    # -------------------------------------------------------------------------
    # Define properties
    # -------------------------------------------------------------------------

    ## class varible with default values
    verbose = False       # A boolean that determines whether or not the classes displays progress

    # -------------------------------------------------------------------------
    # Define constructor
    # -------------------------------------------------------------------------
    def __init__(self, targetSequences="", seedLength=-1, **kwargs):
        # Create the OTTable object
        # obj = OTTable(targetSequences, seedLength) creates an OTTable
        # using the targetSequences and a seedLength.
        # targetSequences can be a Transcriptome object,
        # a dictionary with Header and Sequence fields, or a list of sequences.
        #
        # obj = OTTable(..., weights={}) # Specify the penalty weight for each object
        #
        # obj = OTTable(..., mapType=“”) # Specify the type of map
        #
        # obj = OTTable(..., parallel=True)
        #
        # obj = OTTable(..., transferAbund=True) # Are the abundances
        # in the transcriptome object used as the penalty weights?
        #
        # obj = OTTable(..., 'name', string) # A name for the table

        # instance attributes with default values
        self.name = ""  # A string which defines the name of the OT Table, useful for indexing arrays of OTTables
        self.seedLength=seedLength     		# The length of the seed region
        self.numPar=1          			    # The number of parallel workers
        self.mapType="OTMap2"        			# The type of map
        self.numEntries = 0  			    # The number of entries in the map
        self.uniformWeight = False 		    # Whether every sequence is weighted equally or not

        self.weights = {}            # Weight to apply to each target sequence {1:w1),2:w2,...]
        self.sequences = []             # sequences [(1,s1),(2,s2),...]

        self.transferAbund = False      # A boolean that specifies whether or not to add abundances from transcriptome obj

        self.data = ""			    # Either a OTMap or OTMap2 instance, stores the key for each n-mer and the associated penalty
        self.hashBase = ""				    # The hash base used to quickly convert a n-mer sequence to an integer

        ## initialize the values:
        for k,v in kwargs.items() :
            if k == "verbose": OTTable.verbose = v
            if k == "name": self.name = v
            if k == "seedLength": self.seedLength = v
            if k == "numPar": self.numPar = v
            if k == "mapType": self.mapType = v
            if k == "numEntries": self.numEntries = v
            if k == "uniformWeight": self.uniformWeight = v

            if k == "transferAbund": self.transferAbund = v

            if k == "data": self.data = v
            if k == "hashBase": self.hashBase = v
        # print(kwargs)

        # -------------------------------------------------------------------------
        # Prepare empty object
        # -------------------------------------------------------------------------
        if targetSequences == "":
            return

        # -------------------------------------------------------------------------
        # Parse necessary input
        # -------------------------------------------------------------------------
        # Check number of required inputs
        if (targetSequences != "" and seedLength <= 1):
            error("Error:invalidArguments - A valid set of target sequences in fasta format or a transcriptome object as well as a valid seed length must be provided.")
        
        # Check properties of seed length
        if (not isinstance(seedLength, int)) or (seedLength <= 0):
            error('[Error]:invalidArgument - Seed length must be a positive integer')

        # Archive seed length and create hash base
        self.seedLength = seedLength
        self.hashBase = [4.0** i for i in range(self.seedLength)][::-1]
        self.data = OTMap() # Create empty map of the desired class

        # Check for valid targetSequence arguments
        if targetSequences=="": # Handle empty class request
            return
        else:
            if isinstance(targetSequences,Transcriptome):
                pass
                # Do nothing. This is a valid input and the class definition
                # guarantees that all required attributes are present
            elif isinstance(targetSequences,dict): # This must be a dictionary with the output fields of fastaread{header:seq,...}
                if len(targetSequences)==0:
                    error('[Error]:invalidArguments - Dict of targetSequences is empty.')
            elif isinstance(targetSequences,list):
                if len(targetSequences) == 0:
                    print('[Warning]: List of targetSequences is empty. An empty OTTable is created.')
                    return;
            else:
                error('[Error]:invalidArguments - targetSequences must be either a Transcriptome object, '
                      'a dictionary with Header and Sequence fields, or a list of sequences.')

        # -------------------------------------------------------------------------
        # Transfer abundances/weights if needed
        # -------------------------------------------------------------------------
        if isinstance(targetSequences, Transcriptome): # A transcriptome object was provided
            if self.transferAbund and targetSequences.abundLoaded: # transfer abundances to weights
                self.sequences = [(id_i, targetSequences.Sequences[id_i]) for id_i in targetSequences.ids]
                self.weights = dict([(id_i, targetSequences.abundance[id_i]) for id_i in targetSequences.ids])
            else:
                self.weights = {}
            targetSequences = self.sequences
        elif isinstance(targetSequences, dict):
            targetSequences = [(str(i),s) for i,s in enumerate(list(targetSequences.values()))]
            self.weights = {}
        elif isinstance(targetSequences, list):
            targetSequences = [(str(i), s) for i,s in enumerate(targetSequences)]
            self.weights = {}
        
        # -------------------------------------------------------------------------
        # Check provided weights
        # -------------------------------------------------------------------------
        if len(self.weights)==0:
            self.weights =dict([(str(i),1) for i in range(len(targetSequences))])
            self.uniformWeight = True
        elif len(self.weights) != len(targetSequences):
            error('[Error]: Weights must be equal in length to targetSequences')
        else:
            self.uniformWeight = False

        # -------------------------------------------------------------------------
        # Display information on table construction
        # -------------------------------------------------------------------------
        if OTTable.verbose:
            print('-------------------------------------------------------------------------')
            print('Creating OT Table for ', len(targetSequences), ' sequences and a seed length of ',self.seedLength)
            print('Start at ',tic(1))
            st = tic(0)

        # -------------------------------------------------------------------------
        # Allocate and build hash tables
        # -------------------------------------------------------------------------
        if self.numPar <= 1:
            self.numPar = 1

        # Display progress
        if OTTable.verbose:
            print('Utilizing ',self.numPar, ' parallel workers')

        # Create local variables to handle problem passing object
        # attributes into a parallel process
        seedLength = self.seedLength
        weights = self.weights
        
        # Define properties for fast hash function
        hashBase = self.hashBase
        if self.numPar <= 1:
            ## using only one worker
            data_tmp = OTMap()
            targetSequences_dict = dict(targetSequences)
            for seq_i in targetSequences_dict:
                localSeq = np.array(nt2int(targetSequences_dict[seq_i]), dtype=np.double) - 1.0
                hash = lfilter(hashBase, 1, localSeq)
                hash = hash[seedLength - 1:]

                isValid = lfilter(np.ones(seedLength) / seedLength, 1, ((localSeq > 3) | (localSeq < 0)))
                isValid = np.array(1 - np.array(isValid[seedLength - 1:], dtype=bool), dtype=bool)  # Kludge...

                data_tmp.AddToMap(np.concatenate(([hash[isValid]], [np.zeros(sum(isValid)) + weights[seq_i]]), axis=0))
            data_ls = [data_tmp]
        else:
            # Loop through transcriptome in parallel
            targetSequences_splited = np.array_split(targetSequences, self.numPar)
            ## ***********************************************
            ## multiple processing [1，2，...,i,...,numPar]
            pool_args = [(targetSequences_splited[i],hashBase,seedLength,weights) for i in range(self.numPar)]
            pool = Pool(processes=self.numPar)
            data_ls = pool.map(do_hash_build_func, pool_args)
            pool.close()
            pool.join()
            
        # Add maps from parallel workers
        for j in range(len(data_ls)):
            self.data.AddToMap(data_ls[j].GetTable())

        # -------------------------------------------------------------------------
        # Convert MapType: OTMap is faster to build and OTMap2 is faster
        # for lookup
        # -------------------------------------------------------------------------
        if (self.mapType=="OTMap2") and (isinstance(self.data, OTMap)):
            self.data = OTMap2(self.data.GetTable())

        # -------------------------------------------------------------------------
        # Collect properties of table
        # -------------------------------------------------------------------------
        self.numEntries = self.data.length()

        # -------------------------------------------------------------------------
        # Display information on table construction
        # -------------------------------------------------------------------------
        if OTTable.verbose:
            print('...Completed in  ', (tic(0) - st).total_seconds(), ' s')

    # -------------------------------------------------------------------------
    # CheckValidity
    # -------------------------------------------------------------------------
    def IsValidSequence(self,localSeq): ## return isValid[...]
        if len(localSeq) < self.seedLength:
            return 0
        if isinstance(localSeq,str): # Coerce to integer if need be
            localSeq = np.array(nt2int(localSeq, ACGT=True),dtype=np.double) -1.0

        isValid = lfilter(np.ones(self.seedLength)/self.seedLength ,1, ((localSeq > 3) | (localSeq < 0)))
        isValid = np.array(1 - np.array(isValid[self.seedLength-1:],dtype=bool),dtype=bool)  # Kludge...

        return isValid

    # -------------------------------------------------------------------------
    # Calculate Penalty
    # -------------------------------------------------------------------------
    def CalculatePenalty(self,seq): ## return [penalty, hash]
        # Calculate the penalty associated with a given sequence.
        # penalty = self.CalculatePenalty(seq)

        # Calculate number of seeds within the specified sequence
        numSeed = len(seq) - self.seedLength + 1
        
        # Check for short sequences
        if numSeed < 1:
            return [[],[]]
        
        # Convert only if needed
        if isinstance(seq,str):
            seq = np.array(nt2int(seq, ACGT=True), dtype=np.double)-1.0
        
        # Hash sequence
        hash = lfilter(self.hashBase, 1, seq)
        hash = hash[self.seedLength-1:]
        # print(hash)
        
        # Find >3 which corresponds to ambiguous nucleotides
        isValid = lfilter(np.ones(self.seedLength)/self.seedLength, 1, ((seq >3) | (seq < 0)))
        isValid = np.array(1 - np.array(isValid[self.seedLength-1:],dtype=bool),dtype=bool)  # Kludge...

        # Calculate penalty associated with hash
        penalty = np.empty(numSeed) # Initialize as nan
        penalty[:] = np.nan
        penalty[isValid] = self.data.GetValues(hash[isValid])

        return [penalty,hash]
    
    # -------------------------------------------------------------------------
    # Return Keys
    # -------------------------------------------------------------------------
    def keys(self):  # return keyValues
        return self.data.keys()
   
    # -------------------------------------------------------------------------
    # Return Penalty Values
    # -------------------------------------------------------------------------
    def values(self):  ## return penaltyValues
        return self.data.values()
    
    # -------------------------------------------------------------------------
    # SetParallel
    # -------------------------------------------------------------------------
    def SetNumPar(self, p):
        self.numPar = p

	# -------------------------------------------------------------------------
    # Overload of plus
    # -------------------------------------------------------------------------
    @staticmethod
    def plus(obj1, obj2): ## return obj3 = obj1+obj2
        # Add two OTTables
        # obj3 = OTTables.plus(obj1, obj2)
        
        # -------------------------------------------------------------------------
        # Check validity of objects
        # -------------------------------------------------------------------------
        if (not isinstance(obj1, OTTable)) or (not isinstance(obj2, OTTable)):
            error('[Error]: invalidClass - the two objects should both be OTTable instances')
        
        # -------------------------------------------------------------------------
        # Check compatibility of objects
        # -------------------------------------------------------------------------
        if obj1.seedLength != obj2.seedLength:
            error('[Error]:invalidAddition - Cannot add two OTTables with different seed lengths.')

        if obj1.mapType != obj2.mapType:
            print('[Warning]: OTTables have different mapTypes. Using the mapType of the first object.')

        temp_mapType = obj1.mapType
        # -------------------------------------------------------------------------
        # Create empty OTTable
        # -------------------------------------------------------------------------
        obj3 = OTTable("", obj1.seedLength, mapType=temp_mapType)
        
        # -------------------------------------------------------------------------
        # Combine data from each object
        # -------------------------------------------------------------------------
        ## Get data tables
        data1 = obj1.data.GetTable()
        data2 = obj2.data.GetTable()
        
        # Find unique keys
        [keys, ia, ib] = np.unique(np.concatenate((data1[0,:], data2[0,:]), axis = 1),return_index = True, return_inverse=True)
        
        # Accumulate values for unique keys
        values = np.concatenate((data1[1,:], data2[1,:]), axis = 1)
        values = np.bincount(ib, weights=values)
        
        # -------------------------------------------------------------------------
        # Create new map
        # -------------------------------------------------------------------------
        if temp_mapType == "OTMap":
            obj3.data = obj1.OTMap()
            obj3.data.AddToMap(np.concatenate((keys, values), axis=0))
            obj3.numEntries = len(keys)

        if temp_mapType == "OTMap2":
            obj3.data = obj1.OTMap2()
            obj3.data.AddToMap(np.concatenate((keys, values), axis=0))
            obj3.numEntries = len(keys)

        return obj3


    
    # -------------------------------------------------------------------------
    # Overload of sum
    # -------------------------------------------------------------------------
    @staticmethod
    def sum(otTables): ## return Obj
        # obj = sum(otTables)
        
        # -------------------------------------------------------------------------
        # Check otTable array for validity
        # -------------------------------------------------------------------------
        seedLength_ls = [i.seedLength for i in otTables]
        if len(set(seedLength_ls)) != 1:
            error('[Error]: invalidArguments - otTable list must have the same seed length for all elements')

        uniformWeight_ls = [i.uniformWeight for i in otTables]
        if len(set(uniformWeight_ls)) != 1:
            print('[Warning]: Adding OTTables with different weighting')

        # -------------------------------------------------------------------------
        # Compile all data tables
        # -------------------------------------------------------------------------
        # Display progress
        if otTables[0].verbose:
            PageBreak()
            print('Compiling all data tables')
            print('Start at: ', tic(1))
            st = tic(0)
        # Allocate memory
        totalEntries = sum([i.numEntries for i in otTables])
        allData = np.zeros((2, totalEntries))
        
        # Add tables
        count = 0
        c_i = 0
        for ot_i in otTables:
            c_i += 1
            localData = ot_i.data.GetTable()
            if localData.size != 0:
                allData[:, count + np.arange(localData.shape[1])] = localData ## Why does not use np.concatenate(a1,a2,1)
                count = count + localData.shape[1]
            
            # Display progress
            if (c_i%1000 == 0) and otTables[0].verbose:
                print('...completed ',c_i,' of ', len(otTables))
        
        # Display completion message
        if otTables[0].verbose:
            print('...completed in ' ,toc(st), ' s')
        
        # -------------------------------------------------------------------------
        # Identify and sum redundant entries
        # -------------------------------------------------------------------------
        # Display progress
        if otTables[0].verbose:
            PageBreak()
            print('Adding redundant entries')
            print('Start at: ', tic(1))
            st = tic(0)
        
        # Identify unique keys and sum values that share the same key
        [keys, ia, ic] = np.unique(allData[0,:], return_index=True, return_inverse=True)
        values = np.bincount(ic, allData[1,:])
        
        # Display completion message
        if otTables[0].verbose:
            print('...completed in ',toc(st), ' s')
        
        # -------------------------------------------------------------------------
        # Create new otTable
        # -------------------------------------------------------------------------
        if otTables[0].verbose:
            PageBreak()
            print('Creating new OTTable...')
            print('Start at: ', tic(1))
            st = tic(0)

        # Define seed length and uniformWeight
        seedLength = otTables[0].seedLength
        uniformWeight = otTables[0].uniformWeight
        temp_mapType = otTables[0].mapType

        # Create empty OTTable
        obj = OTTable("", seedLength, verbose = otTables[0].verbose, mapType=temp_mapType)
        obj.uniformWeight = uniformWeight # Transfer uniformWeight flag
        obj.hashBase = [4.0** i for i in range(obj.seedLength)][::-1]

        # Use the data table to create a new map
        new_data = np.concatenate(([keys], [values]), axis=0)
        # print(new_data)
        if temp_mapType == "OTMap":
            obj.data = OTMap(new_data)
            obj.numEntries = len(keys)

        if temp_mapType == "OTMap2":
            obj.data = OTMap2(new_data)
            obj.numEntries = len(keys)

        if otTables[0].verbose:
            print('...completed in ', toc(st), ' s')

        return obj

        

    # -------------------------------------------------------------------------
    # Save Function
    # -------------------------------------------------------------------------
    def Save(self, dirPath):
        # Save the OTTable object (or array of OTTable objects) in a directory specified by dirPath
        # self.Save(dirPath)
       # -------------------------------------------------------------------------
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        save_dict = {
            "verbose": self.verbose,
            "name": self.name,
            "seedLength": self.seedLength,
            "numPar": self.numPar,
            "mapType": self.mapType,
            "numEntries": self.numEntries,
            "uniformWeight": self.uniformWeight,
            "weights": self.weights,
            "seqs": self.sequences,
            "transferAbund": self.transferAbund,
            "data": self.data
        }

        # -------------------------------------------------------------------------
        # Save data
        # -------------------------------------------------------------------------
        with open(dirPath + '/'+self.name+'_OTTable.pkl', 'wb') as fout:
            pickle.dump(save_dict, fout, pickle.HIGHEST_PROTOCOL)


    # -------------------------------------------------------------------------
    # Build a OTTable or an OTTable array from a saved version
    # -------------------------------------------------------------------------
    @staticmethod
    def Load(filePath, verbose=True, **varargin):
        # obj = OTTable.Load(dirPath)
        # obj = OTTable.Load(..., verbose=True) # Determine verbosity of class
        # obj = OTTable.Load(..., mapType="OTMap") # Determine mapType
        
        # -------------------------------------------------------------------------
        # Check provided path
        # -------------------------------------------------------------------------
        if not os.path.exists(filePath):
            error('[Error]:invalidArguments Invalid directory path for loading the OTTable object.')

        # Record progress
        if verbose:
            print('Loading OTTable data:', filePath)
        # -------------------------------------------------------------------------
        # Load and create maps
        # -------------------------------------------------------------------------
        obj = OTTable()
        with open(filePath, 'rb') as fin:
            loaded_dict = pickle.load(fin)

        obj.verbose = loaded_dict["verbose"]
        obj.name = loaded_dict["name"]
        obj.seedLength =  loaded_dict["seedLength"]
        obj.numPar = loaded_dict["numPar"]
        obj.mapType = loaded_dict["mapType"]
        obj.numEntries = loaded_dict["numEntries"]
        obj.uniformWeight = loaded_dict["uniformWeight"]
        obj.weights = loaded_dict["weights"]
        obj.sequences = loaded_dict["seqs"]
        obj.transferAbund = loaded_dict["transferAbund"]
        obj.data = loaded_dict["data"]

        obj.hashBase = [4.0** i for i in range(obj.seedLength)][::-1]

        if "mapType" in varargin and varargin["mapType"] == "OTMap2" and isinstance(obj.data,OTMap):
            obj.data = OTMap2(obj.data.GetTable())
            obj.mapType = "OTMap2"
        if "mapType" in varargin and varargin["mapType"] == "OTMap" and isinstance(obj.data,OTMap2):
            obj.data = OTMap(obj.data.GetTable())
            obj.mapType = "OTMap"
        return obj

        








