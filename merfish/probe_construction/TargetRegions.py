## ------------------------------------------------------------------------
#  TargetRegions Classs
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
from utils.funcs import fastawrite as util_fastawrite
from utils.funcs import error, ParseArguments

class TargetRegions:
    # ------------------------------------------------------------------------
    # [targetRegions] = TargetRegions(varargin)
    # This class stores target region information. 

    # -------------------------------------------------------------------------
    # Define properties
    # -------------------------------------------------------------------------
    def __init__(self,**kwargs):
        # Create the TargetRegions object
        # obh = TargetRegions() # Create empty class
        # obj = TargetRegions(..., 'geneName', name)
        # obj = TargetRegions(..., 'id', geneAccession)
        # obj = TargetRegions(..., 'geneSequence', seq)
        # obj = TargetRegions(..., 'startPos', posVec)
        # obj = TargetRegions(..., 'regionLength', lengthVec)
        # obj = TargetRegions(..., 'GC', gcVec)
        # obj = TargetRegions(..., 'Tm', tmVec)
        # obj = TargetRegions(..., 'specificity', specVec)
        # obj = TargetRegions(..., 'penalties', penMat)
        # obj = TargetRegions(..., 'penaltyNames', {name1, name2, ...})
        # obj = TargetRegions(..., 'sequence', parsedSeqs)

        self.geneName = ''           # Common name of the gene
        self.id = ''                 # Accession for the gene
        self.sequence = []           # Sequences of the target regions
        self.startPos = []           # Starting position for each target region
        self.regionLength = []       # Length for each target region
        self.GC = []                 # GC for each target region
        self.Tm = []                 # Melting temperature for each target region
        self.specificity = []        # Specificty of each target region wrt to all other genes (not including isoforms)
        self.isoSpecificity = []     # Specificity of each target region wrt to other isoforms
        self.penalties = []          # Penalty values for each target region (by row)
        self.penaltyNames = []      # Name of each penalty (i.e. each row in penalities)
        self.numRegions = 0          # Number of target regions in class

        self.geneSequence = []

        self.map = np.array(list('*ACGTRYKMSWBDHVN-'))

        # -------------------------------------------------------------------------
        # Handle empty object request
        # -------------------------------------------------------------------------
        if len(kwargs) < 1:
            return

        ## initialize the values:
        if "numRegions" not in kwargs: kwargs["numRegions"] = 0
        arg_ls = ["geneName", "id", "startPos", "regionLength", "GC", "Tm", "specificity","isoSpecificity",
                  "penalties", "penaltyNames", "numRegions", "geneSequence"]
        parameters = ParseArguments(kwargs, arg_ls)
        # print(kwargs)

        if "loadfromfile" in kwargs and kwargs["loadfromfile"]:
            self.geneName = parameters["geneName"]
            self.id = parameters["id"]
            self.sequence = parameters["sequence"]
            self.startPos = parameters["startPos"]
            self.regionLength = parameters["regionLength"]
            self.GC = parameters["GC"]
            self.Tm = parameters["Tm"]
            self.specificity = parameters["specificity"]
            self.isoSpecificity = parameters["isoSpecificity"]
            self.penalties = parameters["penalties"]
            self.penaltyNames = parameters["penaltyNames"]
            self.numRegions = parameters["numRegions"]

            self.geneSequence = parameters["geneSequence"]
            return


        self.geneName = parameters["geneName"]
        self.id = parameters["id"]
        # self.sequence = parameters["sequence"] ## sequence is used to save the results, not need to initialization
        self.startPos = parameters["startPos"]
        self.regionLength =parameters["regionLength"]
        self.GC = parameters["GC"]
        self.Tm = parameters["Tm"]
        self.specificity = parameters["specificity"]
        self.isoSpecificity =parameters["isoSpecificity"]
        self.penalties = parameters["penalties"]
        self.penaltyNames = parameters["penaltyNames"]
        self.numRegions = parameters["numRegions"]

        self.geneSequence = parameters["geneSequence"]

        # -------------------------------------------------------------------------
        # Parse sequences if a gene sequence is provided
        # -------------------------------------------------------------------------
        self.numRegions = len(self.startPos);
        if len(parameters["geneSequence"])>0 and isinstance(parameters["geneSequence"],str):
            for i in range(self.numRegions):
                self.sequence.append(parameters["geneSequence"][int(self.startPos[i]):int(self.startPos[i] + self.regionLength[i])])

        if len(parameters["geneSequence"])>0 and (not isinstance(parameters["geneSequence"],str)):
            for i in range(self.numRegions):
                self.sequence.append(self.map[parameters["geneSequence"][self.startPos[i]:(self.startPos[i] + self.regionLength[i])] + 2])
                # Note: 2 is added to map the unknown characters to 1.

    # -------------------------------------------------------------------------
    # Fasta write
    # -------------------------------------------------------------------------
    def fastawrite(self, filePath,overwrite=False, **varargin):
        # Write the targetRegions to an existing or new fasta file
        # self.fastawrite(filePath)
        # self.fastawrite(filePath, 'overwrite', boolean)

        # -------------------------------------------------------------------------
        # Prepare file for ovewrite
        # -------------------------------------------------------------------------
        mode="a"
        if os.path.exists(filePath) and overwrite:
            print('Deleting existing file: ', filePath)
            os.remove(filePath);
            mode="w"

        # -------------------------------------------------------------------------
        # Format headers
        # -------------------------------------------------------------------------
        headers = []
        for i in range(self.numRegions):
            localHeader = ['id='+self.id,
                           'geneName='+self.geneName,
                           'startPos='+str(self.startPos[i]),
                           'regionLength='+str(self.regionLength[i]),
                           'GC='+str(self.GC[i]),
                           'Tm='+str(self.Tm[i]),
                           'specificity='+str(self.specificity[i]),
                           'isoSpecificity='+str(self.isoSpecificity[i]),]
            # Add penalties
            for j in range(len(self.penaltyNames)):
                localHeader += ['p_'+self.penaltyNames[j]+'='+str(self.penalties[j][i])];

            headers.append((" ".join(localHeader)).strip());

            # -------------------------------------------------------------------------
            # Write fasta
            # -------------------------------------------------------------------------
            header_dict = dict(zip(headers,self.sequence))
            util_fastawrite(filePath, header_dict,mode=mode);


    # -------------------------------------------------------------------------
    # Save Function
    # -------------------------------------------------------------------------
    def Save(self, dirPath):
        # Save the transcriptome object in a directory specified by dirPath
        # self.Save(dirPath)

        # -------------------------------------------------------------------------
        # Check directory validity
        # -------------------------------------------------------------------------
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        # -------------------------------------------------------------------------
        # Save data
        # -------------------------------------------------------------------------
        save_dict = {
            "geneName": self.geneName,
            "id": self.id,
            "sequence": self.sequence,
            "startPos": self.startPos,
            "regionLength": self.regionLength,
            "GC": self.GC,
            "Tm": self.Tm,
            "penalties": self.penalties,
            "specificity": self.specificity,
            "isoSpecificity": self.isoSpecificity,
            "penaltyNames": self.penaltyNames,
            "numRegions": self.numRegions,
            "geneSequence":self.geneSequence,
            "map":self.map
        }

        with open(dirPath + "/"+self.id+"_TargetRegions.pkl", "wb") as fout:
            pickle.dump(save_dict, fout, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def Load(dirPath, verbose=False,**varargin):
        # obj = TargetRegions.Load(dirPath)
        # obj = TargetRegions.Load(..., 'map', boolean)

        # -------------------------------------------------------------------------
        # Check provided path
        # -------------------------------------------------------------------------
        if not os.path.exists(dirPath):
            error('[Error]:invalidArguments - Invalid directory path for loading the TargetRegions object.')
        targetRegions_ls = []
        for file_i in os.listdir(dirPath):
            with open(dirPath + "/"+file_i, 'rb') as fin:
                loaded_dict = pickle.load(fin)

            obj = TargetRegions(geneName = loaded_dict["geneName"],
                                id = loaded_dict["id"],
                                sequence = loaded_dict["sequence"],
                                startPos = loaded_dict["startPos"],
                                regionLength = loaded_dict["regionLength"],
                                GC = loaded_dict["GC"],
                                Tm = loaded_dict["Tm"],
                                penalties = loaded_dict["penalties"],
                                specificity = loaded_dict["specificity"],
                                isoSpecificity = loaded_dict["isoSpecificity"],
                                penaltyNames = loaded_dict["penaltyNames"],
                                numRegions = loaded_dict["numRegions"],
                                geneSequence = loaded_dict["geneSequence"],
                                map = loaded_dict["map"],
                                loadfromfile=True)
            targetRegions_ls.append(obj)

        return targetRegions_ls





