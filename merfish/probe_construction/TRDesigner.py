## ------------------------------------------------------------------------
#  TRDesigner Classs
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
from utils.funcs import *
from .Transcritome import Transcriptome
from .OTTable import OTTable
from .TargetRegions import TargetRegions

class TRDesigner:
    # ------------------------------------------------------------------------
    # [trDesignerObj] = TRDesigner(varargin)
    # This class designs target regions given a transcriptome object.
    #
    # See Transcriptome and PrimerDesigner
    # ------------------------------------------------------------------------

    ###################################################################################
    # -------------------------------------------------------------------------
    # Define properties
    # -------------------------------------------------------------------------

    ## class varible with default values
    verbose=True 	# A boolean that determines whether the class prints progress

    # -------------------------------------------------------------------------
    # Define constructor
    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        # Create the TRDesigner object
        # obj = TRDesigner(transcriptome=transcriptomeObject)
        # obj = TRDesigner(..., OTTables=[OTTableObj1, OTTableObj2, ...])
        # obj = TRDesigner(..., OTTableNames=[tableName1, tableName2, ...])
        # obj = TRDesigner(..., numPar=1)
        # obj = TRDesigner(..., verbose=True)
        # obj = TRDesigner(..., forbiddenSeqs=[seq1, seq2, ...])
        # obj = TRDesigner(..., specificityTable=OTTableObj)
        # obj = TRDesigner(..., isoSpecificityTables=OTTableObjArray)
        # obj = TRDesigner(..., alwaysDefinedSpecificity=True)

        # -------------------------------------------------------------------------
        # Parse variable inputs
        # -------------------------------------------------------------------------
        # Define defaults

        # instance attributes with default values
        self.transcriptome=""          # A transcriptome object that defines the sequences to target
        self.numPar=1                  # The number of parallel workers

        self.dG={}                    # The free energies associated with all nearest neighbours in all sequences

        self.gc={}                     # The GC content of all sequences
        self.isValid={}                 # The validity of all sequences, i.e. is each base A,C,T,U, or G

        self.OTTableNames=[]            # Names of the off-target tables
        self.OTTables=[]                # An array of off-target tables
        self.penalties=[]               # The penalties associated with each sequence for each table

        self.forbiddenSeqs=[]           # The list of forbidden sequences
        self.isForbiddenSeq=[]          # The presence of forbidden sequences

        self.specificity={}             # The fraction of each sequence that is unique to that sequence
        self.specificityTable=""        # The OTTable used to calculate region specificity

        self.isoSpecificity={}          # The fraction of each sequence that is unique to the isoforms of that gene
        self.isoSpecificityTables=[]    # An array of OTTables used to calculate isoform specificity

        self.alwaysDefinedSpecificity=False  # Whether ill-defined specificities (0/0) are set to 1 or left ill-defined

        if len(kwargs)<1:
            return

        ## initialize the values:
        if "numPar" not in kwargs: kwargs["numPar"] = 1
        if "alwaysDefinedSpecificity" not in kwargs: kwargs["alwaysDefinedSpecificity"] = False
        if "verbose" not in kwargs: kwargs["verbose"] = True

        arg_ls = ["transcriptome","numPar","dG","gc","isValid","OTTableNames", "OTTables","penalties",
                  "forbiddenSeqs","isForbiddenSeq","specificity","specificityTable","isoSpecificity",
                  "isoSpecificityTables","alwaysDefinedSpecificity"]
        parameters = ParseArguments(kwargs,arg_ls)
        # print(kwargs)

        if "loadfromfile" in kwargs and kwargs["loadfromfile"]:
            self.transcriptome = parameters["transcriptome"]
            self.numPar = parameters["numPar"]
            self.gc = parameters["gc"]
            self.dG = parameters["dG"]
            self.isValid = parameters["isValid"]
            self.OTTableNames = parameters["OTTableNames"]
            self.OTTables = parameters["OTTables"]
            self.penalties = parameters["penalties"]
            self.forbiddenSeqs = parameters["forbiddenSeqs"]
            self.isForbiddenSeq = parameters["isForbiddenSeq"]
            self.specificity = parameters["specificity"]
            self.specificityTable = parameters["specificityTable"]
            self.isoSpecificity = parameters["isoSpecificity"]
            self.isoSpecificityTables = parameters["isoSpecificityTables"]
            self.alwaysDefinedSpecificity = parameters["alwaysDefinedSpecificity"]

            TRDesigner.verbose = parameters["verbose"]
            return


        self.alwaysDefinedSpecificity = parameters["alwaysDefinedSpecificity"]
        self.isoSpecificityTables = parameters["isoSpecificityTables"]
        self.specificityTable = parameters["specificityTable"]
        self.transcriptome=parameters["transcriptome"]
        TRDesigner.verbose = parameters["verbose"]

        self.gc = {}
        self.dG = {}
        self.isValid = {}
        self.isForbiddenSeq = []
        self.specificity = {}
        self.isoSpecificity = {}

        # -------------------------------------------------------------------------
        # Check input values
        # -------------------------------------------------------------------------
        if parameters["transcriptome"]!="" and not(isinstance(parameters["transcriptome"], Transcriptome)):
            error('[Error]:invalidArguments - Provided transcriptome must be a Transcriptome() object')

        if (len(parameters["OTTables"])>0) and (len(parameters["OTTables"])!=len(parameters["OTTableNames"])):
            error('[Error]:invalidArguments - An equal number of OTTable Names as OTTables must be provided.')

        # -------------------------------------------------------------------------
        # Create annotations for penalty table 
        # -------------------------------------------------------------------------
        for i in range(len(parameters["OTTables"])):
            self.AddOTTable(parameters["OTTables"][i], parameters["OTTableNames"][i])

        # -------------------------------------------------------------------------
        # Add specificity table
        # -------------------------------------------------------------------------
        if parameters["specificityTable"]!="" or len(parameters["isoSpecificityTables"])>0:
            self.AddSpecificityTable(parameters["specificityTable"], parameters["isoSpecificityTables"])
        
        # -------------------------------------------------------------------------
        # Create annotations for GC and Tm calculations
        # -------------------------------------------------------------------------
        seqs = parameters["transcriptome"].Sequences
        gc = {}
        dG = {}
        valid = {}
        for id_i in seqs:
            seq_i = np.array(list(seqs[id_i]))
            gc[id_i] = ((seq_i=="G") | (seq_i == "C"))       # Determine G/C
            valid[id_i]= np.isin(seq_i,["A","C","G","T","U"])      # Determine validity
            dG[id_i] = TRDesigner.SantaLuciaNearestNeighbor(seqs[id_i])  # Calculate thermodynamic properties
        # Store values
        self.gc = gc
        self.dG = dG
        self.isValid = valid

        # -------------------------------------------------------------------------
        # Create forbidden sequence annotations  
        # -------------------------------------------------------------------------
        for i in parameters["forbiddenSeqs"]:
            self.AddForbiddenSeq[i]

    
    # -------------------------------------------------------------------------
    # Add a forbidden sequence
    # -------------------------------------------------------------------------
    def AddForbiddenSeq(self, seq="", replace=False, **varargin):
        # Add a forbidden sequence or replace existing sequences
        # self.AddForbiddenSeq(sequence)
        # self.AddForbbidenSeq(..., 'replace', boolean) -- Replace the
        #   existing sequences

        # -------------------------------------------------------------------------
        # Parse necessary input
        # -------------------------------------------------------------------------
        if (seq=="") or (not isinstance(seq,str)) or (not np.all(np.isin(seq,["A","C","G","T","U"]))):
            error('[Error]: AddForbiddenSeq - The provided sequence is invalid.')

        # -------------------------------------------------------------------------
        # Reset sequences if requested
        # -------------------------------------------------------------------------
        if replace:
            self.forbiddenSeqs = []
            self.isForbiddenSeq = []

        # -------------------------------------------------------------------------
        # Add sequence
        # -------------------------------------------------------------------------
        if isinstance(seq,str):
            intseq = np.array(nt2int(seq, ACGT=True),np.double) - 1.0

        self.forbiddenSeqs.append(seq)
        
        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Finding all occurrences of ', seq)
            st = tic(99)
        
        # -------------------------------------------------------------------------
        # Calculate penalty
        # -------------------------------------------------------------------------
        hashBase = [4.0** i for i in range(len(seq))][::-1]
        forbiddenHash = sum(hashBase * intseq)

        seqs = self.transcriptome.Sequences
         #Loop over transcriptome
        hasForbiddenSeq = {}
        for id_i in seqs:
            seq_i = np.array(nt2int(seqs[id_i]), np.double) - 1.0
            # Hash sequence
            seqHash = lfilter(hashBase, 1, seq_i)
            seqHash = seqHash[len(hashBase)-1:]

            # Find forbidden sequences
            hasForbiddenSeq[id_i] = (seqHash == forbiddenHash)

        self.isForbiddenSeq.append(hasForbiddenSeq)
        
        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(st), 's at ', tic(1))
    
    # -------------------------------------------------------------------------
    # Add a penalty table and calculate the appropriate penalties
    # -------------------------------------------------------------------------
    def AddOTTable(self, otTable, tableName, replace=False, **varargin):
        # Add or modify an existing penalty table
        # self.AddOTTable(self, OTTable, OTTableName)
        # self.AddOTTable(..., 'replace', boolean) -- replace existing
        #  tables
        
        # -------------------------------------------------------------------------
        # Parse necessary input
        # -------------------------------------------------------------------------
        if (tableName == "") or (not isinstance(otTable, OTTable)) or (not isinstance(tableName,str)):
            error('matlabdefs:invalidArguments', 'Both an OTTable and a name must be provided')

        # -------------------------------------------------------------------------
        # Reset tables if requested
        # -------------------------------------------------------------------------
        if replace:
            self.OTTableNames = []
            self.OTTables = []
            self.penalties = []
        
        # -------------------------------------------------------------------------
        # Add table and name
        # -------------------------------------------------------------------------
        if len(self.OTTables)==0: # Handle initial table/name pair
            self.OTTables = [otTable]
            self.OTTableNames = [tableName]
        else:
            self.OTTables.append(otTable)
            self.OTTableNames.append(tableName)
        
        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Calculating penalty values for ', self.OTTableNames[-1])
            st = tic(99)
        
        # -------------------------------------------------------------------------
        # Build penalty
        # -------------------------------------------------------------------------
        # Make a local copy of the sequences
        seqs = self.transcriptome.Sequences
                        
        # Make reference to table
        table = self.OTTables[-1]
        
        # Loop over all sequences
        penalty = {}
        for s in seqs:
            penalty[s] = table.CalculatePenalty(seqs[s])[0] # Use table to calculate penalty

        # Save penalty
        self.penalties.append(penalty)
        
        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(st), 's at ',tic(1))

    # -------------------------------------------------------------------------
    # Add a table for determining region specificity
    # -------------------------------------------------------------------------
    def AddSpecificityTable(self, specificityTable, isoSpecificityTables):
        # Add or replace a specificity table
        # self.AddSpecificityTable(self, specificityTable, isoSpecificityTables)
        
        # -------------------------------------------------------------------------
        # Parse necessary input
        # -------------------------------------------------------------------------
        # Handle case that isoSpecificityTable is not provided
        if (isoSpecificityTables=="") or (len(isoSpecificityTables)==0):
            isoSpecificityTables = []

        if not isinstance(specificityTable, OTTable):
            error('[Error]:AddSpecificityTable-The provided specificity table is not a valid OTTable')

        # -------------------------------------------------------------------------
        # Update table
        # -------------------------------------------------------------------------
        self.specificityTable = specificityTable
        self.isoSpecificityTables = isoSpecificityTables
        
        # -------------------------------------------------------------------------
        # Check seed lengths
        # -------------------------------------------------------------------------
        if len(isoSpecificityTables) > 0:
            isoSeedLength = list(set([i.seedLength for i in self.isoSpecificityTables]))
            if len(isoSeedLength) != 1:
                error('[Error]: AddSpecificityTable - The OTTable list contains elements with different seed lengths!')
            else:
                if isoSeedLength[0] != self.specificityTable.seedLength:
                    error('[Error] :invalidArguments The seed lengths must be equal for the specificity table and the isoform specificity tables!')

        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Calculating region specificity for a seed length of ',self.specificityTable.seedLength)
            st = tic(99)
        
        # -------------------------------------------------------------------------
        # Build specificity score
        # -------------------------------------------------------------------------
        
        # Determine normalization based on the weighting of the table
        if self.specificityTable.uniformWeight:
            normalization = dict([(id_i,1.0) for id_i in self.transcriptome.ids])
        else:
            normalization = self.transcriptome.abundance
        
        # Determine if isoform specificity should be calculated
        if len(self.isoSpecificityTables)==0:   # If no isoform specificity tables are provided, calculate specificity directly
            print('Calculating specificity without isoform information...')
            s_i = 0
            for s in self.transcriptome.Sequences:
                s_i += 1
                self.specificity[s] = normalization[s] / (self.specificityTable.CalculatePenalty(self.transcriptome.Sequences[s])[0])

                if self.verbose and (s_i%1000==0):
                    print('... completed ',s_i,' seqs')
        else:
            print('Utilizing isoform information...')
            s_i = 0
            for s in self.transcriptome.Sequences:
                s_i += 1
                # Get sequence name and find OTTables
                localGeneName = self.transcriptome.id2name[s]
                localTable = [i for i in self.isoSpecificityTables if i.name==localGeneName]

                # Confirm that a table was found
                if len(localTable) > 1:
                    print('[Warning]: Found more than one OTTable for ', localGeneName)
                    print("Picking one randomly...")
                elif len(localTable)==0:
                    print('[Warning]: Did not find an OTTable for ', localGeneName)
                    continue
                #
                # Calculate isoform adjusted specificity
                isoCounts = localTable[0].CalculatePenalty(self.transcriptome.Sequences[s])[0]
                isoCounts = np.array(isoCounts)
                if np.any(isoCounts==0): print("Divided by zero:",localGeneName,"(",s,s_i,")")
                self.isoSpecificity[s] = normalization[s] / isoCounts
                self.specificity[s] = isoCounts / (self.specificityTable.CalculatePenalty(self.transcriptome.Sequences[s])[0])

                # Update progress
                # print(s,s_i)
                if self.verbose and (s_i%1000 == 0):
                    print('... completed ',s_i, ' seqs')

        # -------------------------------------------------------------------------
        # Handle ill-defined specificity values
        # -------------------------------------------------------------------------
        if self.alwaysDefinedSpecificity:
            if self.verbose:
                print('... setting all ill-defined specificity values to 1')

            # Loop over specificity values
            for s in self.specificity:
                localSpecificity = np.array(self.specificity[s])
                localSpecificity[np.isnan(localSpecificity)] = 1
                self.specificity[s] = localSpecificity
        
        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ',toc(st), 's at ',tic(1))

    
    # -------------------------------------------------------------------------
    # Get the forbidden seq penalty for all putative regions
    # -------------------------------------------------------------------------
    def GetRegionForbiddenSeqs(self, lenx,**varargin): # return[noForbiddenSeqs, ids, names]
        # Return a boolean for each valid probe
        # [noForbiddenSeqs, ids, names] = self.GetRegionForbiddenSeqs(len)
        # [noForbiddenSeqs, ids, names] = self.GetRegionForbiddenSeqs(len, geneName=names)
        # [noForbiddenSeqs, ids, names] = self.GetRegionForbiddenSeqs(len, geneID=ids)
        # noForbiddenSeqs specifies if the target region starting at each position
        # does not contain any of the forbidden sequences

        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls= varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val,v_ls)
        
        # -------------------------------------------------------------------------
        # print Progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            st = tic(99)
            print('Finding forbidden sequences for ',len(inds), ' transcripts')
        
        # -------------------------------------------------------------------------
        # Search for forbidden sequences
        # -------------------------------------------------------------------------
        for s in range(len(self.forbiddenSeqs)): # Loop over forbidden sequences
            # print progress
            if self.verbose:
                print('Finding ', self.forbiddenSeqs[s])
            
            # Calculate filter window
            windowLen = lenx - len(self.forbiddenSeqs[s]) + 1
            
            # Handle first forbidden sequence
            noForbiddenSeqs=[]
            if s==0:
                for l in range(len(inds)): # Loop over transcriptome
                    hasSeq = lfilter(np.ones(windowLen)/windowLen, 1, self.isForbiddenSeq[s][inds[l]])
                    noForbiddenSeqs.append(np.array(1-np.array(hasSeq[windowLen-1:],dtype=bool),dtype=bool))
            else:
                for l in range(len(inds)): # Loop over transcriptome
                    hasSeq = lfilter(np.ones(windowLen)/windowLen, 1, self.isForbiddenSeq[s][inds[l]])
                    noForbiddenSeqs.append(np.array(1-np.array(hasSeq[windowLen-1:],dtype=bool),dtype=bool) & noForbiddenSeqs[l])

        # -------------------------------------------------------------------------
        # print Progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ',toc(st), ' s')

        return noForbiddenSeqs

    # -------------------------------------------------------------------------
    # Get all valid regions
    # -------------------------------------------------------------------------
    def GetRegionValidity(self, lenx, **varargin): # return [isValid, ids, names]
        # Return a boolean for each valid probe
        # [isValid, ids, names] = self.GetRegionValidity(len)
        # [isValid, ids, names] = self.GetRegionValidity(len, 'geneName', names)
        # [isValid, ids, names] = self.GetRegionValidity(len, 'geneID', ids)
        # isValid specifies if the target region starting at each position
        # is valid

        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls = varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # Compute sliding GC window
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            st = tic(99)
            print('Calculating validity for ', len(ids), ' transcripts')

        isValid = []
        for l in ids:
            seq_l = np.array(nt2int(self.transcriptome.Sequences[l]),dtype=np.double) -1.0
            valid = lfilter(np.ones(lenx)/lenx, 1, (seq_l > 3) | (seq_l < 0))
            isValid.append(np.array(1-np.array(valid[lenx-1:],dtype=bool),dtype=bool))

        if self.verbose:
            print('... completed in ', toc(st), 's')

        return isValid


    # -------------------------------------------------------------------------
    # Get GC values for all putative regions
    # -------------------------------------------------------------------------
    def GetRegionGC(self, lenx, **varargin): ## [GC, ids, names]
        # Return the GC content of a set of probes of specified length
        # [GC, ids, names, validRequest] = self.GetRegionGC(len)
        # [GC, ids, names, validRequest] = self.GetRegionGC(len, 'geneName', names)
        # [GC, ids, names, validRequest] = self.GetRegionGC(len, 'geneID', ids)
        # GC is a cell array of all GC values 
        # ids are the gene ids associated with each GC curve
        # names are the gene names associated with each GC curve
        # validRequest is a boolean that can be used to confirm that a
        # requested gene/id is actually in the transcriptome. 
        
        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls = varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # Compute sliding GC window
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            st = tic(99)
            print('Calculating GC content for ',len(inds),' transcripts')

        GC = []
        for l in range(len(inds)):
            gc = lfilter(np.ones(lenx)/lenx, 1, self.gc[ids[l]])
            GC.append(gc[lenx-1:])
        
        if self.verbose:
            print('... completed in ',toc(st),"s")

        return GC
        
    
           
    # -------------------------------------------------------------------------
    # GetRegionPenalty
    # -------------------------------------------------------------------------
    def GetRegionPenalty(self, lenx, OTtableName, **varargin):
        # Return the penalty associated with all possible probes of specified length
        # [penalty, ids, names] = self.GetRegionPenalty(len, OTtableName)
        # [penalty, ids, names] = self.GetRegionPenalty(len, OTtableName, 'geneName', names)
        # [penalty, ids, names] = self.GetRegionPenalty(len, OTtableName, 'geneID', ids)
        # penalty is a cell array of all penalties 
        # ids are the gene ids associated with each curve
        # names are the gene names associated with each curve
        
        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls = varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # Find table ind
        # -------------------------------------------------------------------------
        id = [i for i,e in enumerate(self.OTTableNames) if e == OTtableName]
        if len(id)==0:
            error('[Error]:invalidArgument - The specified table does not exist')
        
        if len(id) > 1:
            print('[Warning]: Multiple tables matching the specified name were found')
        
        id = id[0]
        localPenalties = self.penalties[id]
        
        # -------------------------------------------------------------------------
        # Compute appropriate window length
        # -------------------------------------------------------------------------
        windowLen = lenx - self.OTTables[id].seedLength + 1
        
        # -------------------------------------------------------------------------
        # Compute sliding GC window
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            st = tic(99)
            print('Calculating penalty for ',len(inds),' transcripts using table: ', OTtableName)
        penalty= []
        for l in ids:
            pen = lfilter(np.ones(windowLen), 1, localPenalties[l]) # Return sum, not average
            penalty.append(pen[windowLen-1:])
        
        if self.verbose:
            print('... completed in ',toc(st),"s")
        
        return penalty
    
    # -------------------------------------------------------------------------
    # Get the specificity of putative target regions
    # -------------------------------------------------------------------------
    def GetRegionSpecificity(self, lenx, **varargin): ## return [specificity, ids, names]
        # Return the penalty associated with all possible probes of specified length
        # [specificity, ids, names] = self.GetRegionSpecificity(len)
        # ... = self.GetRegionSpecificity(len, 'geneName', names)
        # ... = self.GetRegionSpecificity(len, 'geneID', ids)
        
        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls = varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # Compute appropriate window length
        # -------------------------------------------------------------------------
        windowLen = lenx - self.specificityTable.seedLength + 1
        
        # -------------------------------------------------------------------------
        # Compute sliding window
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            timer = tic(99)
            print('Calculating specificity for ', len(inds), ' transcripts')

        specificity = []
        for l in ids:
            spec = lfilter(np.ones(windowLen)/windowLen, 1, self.specificity[l]) # Return average
            specificity.append(spec[windowLen-1:])
        
        if self.verbose:
            print('... completed in ',toc(timer), "s")

        return specificity
        
    
    
    # -------------------------------------------------------------------------
    # Get the isoform specificity of putative target regions
    # -------------------------------------------------------------------------
    def GetRegionIsoSpecificity(self, lenx, **varargin): ## return [isoSpecificity, ids, names] =
        # Return the penalty associated with all possible probes of specified length
        # [specificity, ids, names] = self.GetRegionIsoSpecificity(len)
        # ... = self.GetRegionIsoSpecificity(len, 'geneName', names)
        # ... = self.GetRegionIsoSpecificity(len, 'geneID', ids)
        
        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls = varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # Compute appropriate window length
        # -------------------------------------------------------------------------
        windowLen = lenx - self.isoSpecificityTables[0].seedLength + 1
        
        # -------------------------------------------------------------------------
        # Compute sliding window
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            timer = tic(99)
            print('Calculating isoform specificity for ', len(inds), ' transcripts')
        isoSpecificity = []
        for l in ids:
            spec = lfilter(np.ones(windowLen)/windowLen, 1, self.isoSpecificity[l]) # Return average
            isoSpecificity.append(spec[windowLen-1:])
        
        if self.verbose:
            print('... completed in ', toc(timer))

        return isoSpecificity
        
    
    
    # -------------------------------------------------------------------------
    # Return the Tm for all regions of the desired transcripts
    # -------------------------------------------------------------------------
    def GetRegionTm(self, lenx, monovalentSalt = 0.3,probeConc = 5e-9, **varargin):
        # Return the Tm for all specified probes of length, len
        # [Tm, ids, names, validRequest] = self.GetRegionTm(len)
        # [Tm, ids, names, validRequest] = self.GetRegionTm(len, 'geneName', names)
        # [Tm, ids, names, validRequest] = self.GetRegionTm(len, 'geneID', ids)
        # [Tm, ids, names, validRequest] = self.GetRegionTm(len, 'monovalentSalt', saltConc)
        # [Tm, ids, names, validRequest] = self.GetRegionTm(len, 'probeConc', probeConc)
        # Tm is a cell array of all Tm values 

        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        if "ind" in varargin: k_val = "ind"
        if "name" in varargin: k_val = "name"
        if "id" in varargin: k_val = "id"
        v_ls = varargin[k_val]

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # print Progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            timer = tic(99)
            print('Calculating Tm for ', len(inds), ' transcripts '
                'with ',monovalentSalt,' M salt and ',probeConc,' M probe')
        
        
        # -------------------------------------------------------------------------
        # Calculate Tm
        # -------------------------------------------------------------------------
        localDG = [self.dG[i] for i in ids] # Local thermodynamic properties
        seqs = [self.transcriptome.Sequences[i] for i in ids] # Local copy of sequences
        Tm = []
        for l in range(len(inds)): # Loop over requested sequences
            # Get local sequence
            intSeq = np.array(nt2int(seqs[l]),dtype=np.double)-1
            
            # Calculate total H and S per putative probe
            H = lfilter(np.ones(lenx-1), 1, localDG[l][0,:]) # len-1 nn per sequence of len
            S = lfilter(np.ones(lenx-1), 1, localDG[l][1,:])
            H = H[(lenx-2):]
            S = S[(lenx-2):]
            
            # Determine s
            fivePrimeAT = (intSeq[:-lenx+1] == 0) | (intSeq[:-lenx+1] == 3)
            threePrimeAT = (intSeq[lenx-1:] == 0) | (intSeq[lenx-1:] == 3)
            
            # Add  corrections
            H = H + 0.2 + 2.2*fivePrimeAT + 2.2*threePrimeAT
            S = S + -5.7 + 6.9*fivePrimeAT + 6.9*threePrimeAT
            
            # Apply salt correction
            S = S + 0.368*(lenx-1)*np.log(monovalentSalt)
            
            # Calculate Tm in C
            Tm.append(H*1000 / (S + 1.9872 * np.log(probeConc)) - 273.15 )
            
            # NOTE: For the future, I should provide two concentrations,
            # probe and target, and this should be log(probeC - targetC/2)
            # where probeC > targetC. They are switched in the
            # concentrations are switched.
        
        
        # -------------------------------------------------------------------------
        # print Progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ',toc(timer),"s")

        return Tm
    
    # -------------------------------------------------------------------------
    # Design target regions
    # -------------------------------------------------------------------------
    def DesignTargetRegions(self,regionLength=[30],Tm=[],GC=[],specificity=[],isoSpecificity=[],
                            OTTables = {}, monovalentSalt=0.3,probeConc=5e-9,
                            includeSequence=True,threePrimeSpace=0,removeForbiddenSeqs=False,
                            **varargin):
        # This method returns target regions, tiled to be non-overlapping,
        # and subject to a variety of constraints.  
        # targetRegions = self.DesignTargetRegions(..., 'regionLength', [allPossibleLengthValues])
        # targetRegions = self.DesignTargetRegions(..., 'Tm', [low,up])
        # targetRegions = self.DesignTargetRegions(..., 'GC', [low,up])
        # targetRegions = self.DesignTargetRegions(..., 'OTTables', {'name', [low, up], 'name', range, ...})
        # targetRegions = self.DesignTargetRegions(..., 'geneName', names)
        # targetRegions = self.DesignTargetRegions(..., 'geneID', ids)
        # targetRegions = self.DesignTargetRegions(..., 'specificity', [low, up])
        # targetRegions = self.DesignTargetRegions(..., 'isoSpecificity', [low, up])

        Tm_ls = Tm
        GC_ls = GC
        specificity_ls = specificity
        isoSpecificity_ls = isoSpecificity
        OTTable_ls = OTTables
                
        # -------------------------------------------------------------------------
        # Get internal indices for the requested transcripts
        # -------------------------------------------------------------------------
        v_ls = []
        k_val = "" ## ind / name / id
        if "k_val" in varargin:
            k_val = varargin["k_val"]
        if "v_ls" in varargin:
            v_ls = varargin[v_ls]

        if k_val !="" and (k_val not in ["ind","name","id"]):
            error('[Error]: Invalid key value,k_val = "ind" / "name" / "id ')

        [inds, ids, names] = self.transcriptome.GetInternalInds(k_val, v_ls)
        
        # -------------------------------------------------------------------------
        # Handle case of no valid requested transcripts
        # -------------------------------------------------------------------------
        if len(inds)==0:
            print('[Warning]:noValidEntries - None of the requested entries were found')
            targetRegions = TargetRegions()
            return
        
        
        # -------------------------------------------------------------------------
        # Initialize variables for loops
        # -------------------------------------------------------------------------
        targetRegions = []
        GC={}
        Tm={}
        specificity={}
        isoSpecificity={}
        penalties = {}
        indsToKeep = {}
        
        # -------------------------------------------------------------------------
        # Loop over probe lengths
        # -------------------------------------------------------------------------
        for l in range(len(regionLength)):
            penalties[l] = {}
            # -------------------------------------------------------------------------
            # print progress
            # -------------------------------------------------------------------------
            if self.verbose:
                PageBreak()
                timer1= tic(99)
                print('Designing target regions of length ', regionLength[l])

            # -------------------------------------------------------------------------
            # Calculate all region properties
            # -------------------------------------------------------------------------
            GC[l] = self.GetRegionGC(regionLength[l], id=ids)
            Tm[l] = self.GetRegionTm(regionLength[l], id=ids)

            for p in range(len(self.OTTables)):
                penalties[l][p] = self.GetRegionPenalty(regionLength[l], self.OTTableNames[p], id=ids)
            
            specificity[l]= self.GetRegionSpecificity(regionLength[l], id=ids)
            isoSpecificity[l] = self.GetRegionIsoSpecificity(regionLength[l], id=ids)
            
            # -------------------------------------------------------------------------
            # Define numerical pad: to address round off error in filter
            # -------------------------------------------------------------------------
            numPad = 10*np.finfo(float).eps
            
            # -------------------------------------------------------------------------
            # Initialize Inds to Keep
            # -------------------------------------------------------------------------
            indsToKeep[l] = self.GetRegionValidity(regionLength[l], id=ids)

            # -------------------------------------------------------------------------
            # Determine probe properties (only if needed) and cut
            # -------------------------------------------------------------------------
            if removeForbiddenSeqs:
                noForbiddenSeqs = self.GetRegionForbiddenSeqs(regionLength[l], id=ids)
                for i in range(len(noForbiddenSeqs)):
                    indsToKeep[l][i]= indsToKeep[l][i] & noForbiddenSeqs[i]
            if len(GC_ls) >0:
                for i in range(len(GC[l])):
                    indsToKeep[l][i] = indsToKeep[l][i] & (GC[l][i] >= (GC_ls[0]-numPad)) & (GC[l][i] <= (GC_ls[1]+numPad))
            if len(Tm_ls)>0:
                for i in range(len(Tm[l])):
                    indsToKeep[l][i] = indsToKeep[l][i] & (Tm[l][i] >= (Tm_ls[0]-numPad)) & (Tm[l][i] <= (Tm_ls[1]+numPad))
            if len(specificity_ls) >0:
                for i in range(len(specificity[l])):
                    indsToKeep[l][i] = indsToKeep[l][i] & (specificity[l][i] >= (specificity_ls[0]-numPad)) \
                                       & (specificity[l][i] <= (specificity_ls[1]+numPad))
            if len(isoSpecificity_ls) > 0:
                for i in range(len(isoSpecificity[l])):
                    indsToKeep[l][i] = indsToKeep[l][i] & (isoSpecificity[l][i] >= (isoSpecificity_ls[0]-numPad)) \
                                       & (isoSpecificity[l][i] <= (isoSpecificity_ls[1]+numPad))
            if len(OTTable_ls) >0:
                for t in OTTable_ls:
                    if t not in self.OTTableNames:
                        error('[Error] DesignTargetRegions() - Unrecognized OTTable name');
                    pid = self.OTTableNames.index(t)
                    for i in range(len(ids)):
                        indsToKeep[l][i] = indsToKeep[l][i] & (penalties[l][pid][i] >= (OTTable_ls[t][0]-numPad)) \
                                           & (penalties[l][pid][i] <= (OTTable_ls[t][1]+numPad));

            # -------------------------------------------------------------------------
            # print progress
            # -------------------------------------------------------------------------
            if self.verbose:
                print('... completed identification of possible targets of length ',
                      regionLength[l], ' nt in ', toc(timer1), ' s')
        
        
        # -------------------------------------------------------------------------
        # print Progress
        # -------------------------------------------------------------------------
        # print progress
        if self.verbose:
            PageBreak()
            print('Tiling and compiling target regions...')
            timer2 = tic(99)
        

        # -------------------------------------------------------------------------
        # Prepare variables
        # -------------------------------------------------------------------------
        targetRegions = [1] * len(inds)
        ids = [self.transcriptome.ids[i] for i in inds]
        geneNames = [self.transcriptome.geneNames[i] for i in inds]
        Sequences = [self.transcriptome.Sequences[i] for i in ids]
        
        # -------------------------------------------------------------------------
        # Loop over objects in parallel
        # -------------------------------------------------------------------------
        for i in range(len(inds)):
            # Compile region properties
            regionProps = np.zeros((6,0))
            for l in range(len(regionLength)):
                startPos = np.argwhere(indsToKeep[l][i]).ravel()
                localProps = np.zeros((6,len(startPos)))
                if len(startPos)>0: # Only build object if there are any valid starting positions
                    localProps[0,:] = startPos
                    localProps[1,:] = regionLength[l]*np.ones(sum(indsToKeep[l][i]))
                    localProps[2,:] = Tm[l][i][indsToKeep[l][i]]
                    localProps[3,:] = GC[l][i][indsToKeep[l][i]]
                    localProps[4,:] = specificity[l][i][indsToKeep[l][i]]
                    localProps[5,:] = isoSpecificity[l][i][indsToKeep[l][i]]
                    regionProps = np.concatenate((regionProps,localProps), axis=1)
            # print("i=",i,ids[i])
            # Tile regions and return properties of selected regions
            selectedRegionProps = TRDesigner.TileRegions(regionProps, threePrimeSpace)
            
            # Build a new target region object
            newTR = TargetRegions(id = ids[i],
                                  geneName = geneNames[i],
                                  geneSequence = Sequences[i],
                                  startPos = selectedRegionProps[0,:],
                                  regionLength = selectedRegionProps[1,:],
                                  Tm = selectedRegionProps[2,:],
                                  GC = selectedRegionProps[3,:],
                                  specificity = selectedRegionProps[4,:],
                                  isoSpecificity = selectedRegionProps[5,:]
                                  )
            
            # App to growing list of target regions
            targetRegions[i] = newTR
        

        # -------------------------------------------------------------------------
        # print progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer2),' s')
                
        return targetRegions
        
    # -------------------------------------------------------------------------
    # Save def
    # -------------------------------------------------------------------------
    def Save(self, dirPath):
        # Save the TRDesigner object in a directory specified by dirPath
        # self.Save(dirPath)

        if self.verbose:
            PageBreak()
            print('Saving TRDesigner object to ', dirPath + '/trDesigner.pkl')
            st = tic(99)

        # -------------------------------------------------------------------------
        # Check directory validity
        # -------------------------------------------------------------------------
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        # -------------------------------------------------------------------------
        # Save fields
        # -------------------------------------------------------------------------
        save_dict = {
            "verbose": self.verbose,
            "transcriptome": self.transcriptome,
            "numPar": self.numPar,
            "dG": self.dG,
            "gc": self.gc,
            "isValid": self.isValid,
            "OTTableNames": self.OTTableNames,
            "OTTables": self.OTTables,
            "penalties": self.penalties,
            "forbiddenSeqs": self.forbiddenSeqs,
            "isForbiddenSeq": self.isForbiddenSeq,
            "specificity": self.specificity,
            "specificityTable": self.specificityTable,
            "isoSpecificity": self.isoSpecificity,
            "isoSpecificityTables": self.isoSpecificityTables,
            "alwaysDefinedSpecificity": self.alwaysDefinedSpecificity
        }

        # -------------------------------------------------------------------------
        # Save data
        # -------------------------------------------------------------------------
        with open(dirPath + '/trDesigner.pkl', 'wb') as fout:
            pickle.dump(save_dict, fout, pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            print('... completed in ', toc(st),' s')


    # -------------------------------------------------------------------------
    # Static methods
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Build a TRDesigner object from a saved version
    # -------------------------------------------------------------------------
    @staticmethod
    def Load(filePath,lightweight=False, **varargin):
        # obj = TRDesigner.Load(dirPath)
        # obj = TRDesigner.Load(dirPath, lightweight=true/false)
                
        # -------------------------------------------------------------------------
        # Check provided path
        # -------------------------------------------------------------------------

        if not os.path.exists(filePath):
            error('[Error]:invalidArguments - Invalid directory path for loading the TRDesigner object.')

        # Remove fields for the lightweight option if requested
        if lightweight:
            print('Loading a lightweight version TRDesigner. The following attributes will not be available')
            fieldsToExclude = ['specificityTable', 'isoSpecificityTables']

            for F in fieldsToExclude:
                print('...',F)

        with open(filePath, 'rb') as fin:
            loaded_dict = pickle.load(fin)

        # -------------------------------------------------------------------------
        # Create empty object (to define fields to load)
        # -------------------------------------------------------------------------
        obj = TRDesigner(verbose = loaded_dict["verbose"],
                         transcriptome = loaded_dict["transcriptome"],
                         numPar = loaded_dict["numPar"],
                         dG = loaded_dict["dG"],
                         gc = loaded_dict["gc"],
                         isValid = loaded_dict["isValid"],
                         OTTableNames = loaded_dict["OTTableNames"],
                         OTTables = loaded_dict["OTTables"],
                         penalties = loaded_dict["penalties"],
                         forbiddenSeqs = loaded_dict["forbiddenSeqs"],
                         isForbiddenSeq = loaded_dict["isForbiddenSeq"],
                         specificity = loaded_dict["specificity"],
                         specificityTable = loaded_dict["specificityTable"],
                         isoSpecificity = loaded_dict["isoSpecificity"],
                         isoSpecificityTables = loaded_dict["isoSpecificityTables"],
                         alwaysDefinedSpecificity = loaded_dict["alwaysDefinedSpecificity"],
                         loadfromfile=True)
        return obj
    
    
    # -------------------------------------------------------------------------
    # SantaLucia Nearest Neighor Calculations  
    # -------------------------------------------------------------------------
    @staticmethod
    def SantaLuciaNearestNeighbor(Seq):
        # Return the enthalpy and entropy associated with each nearest
        # neighbor pair in the sequence
        # dG = TRDesigner.SantaLuciaNearestNeighbor(intSeq)

        if len(Seq) <=0:
            error("[Error]:SantaLuciaNearestNeighbor() - Seq length is 0. ")

        # -------------------------------------------------------------------------
        # Coerce sequence format  
        # -------------------------------------------------------------------------
        if isinstance(Seq,str):
            intSeq = np.array(nt2int(Seq),dtype=np.double) - 1.0
        elif isinstance(Seq[0],np.float) or isinstance(Seq,np.ndarray):
            intSeq = Seq

        # -------------------------------------------------------------------------
        # Initialize dG: dG(1,:) is the enthalpy (kcal/mol) dG(2,:) is the entropy (cal/(mol K) 
        # -------------------------------------------------------------------------
        dG = np.empty((2, len(intSeq)-1)) # nan is a not valid flag
        dG[:] = np.nan
        # -------------------------------------------------------------------------
        # Assign terms: AA = 1, AC = 2, ... CA = 5 ... TT = 16
        # -------------------------------------------------------------------------
        nnID = 4*intSeq[:-1] + intSeq[1:] # Convert pairs to index
        # AC -> 4*0 + 1 + 1= 2 CA = 4*1 + 0 + 1 = 5
        isValid = (intSeq[:-1] <= 3) & (intSeq[1:] <= 3) & (intSeq[:-1] >= 0) & (intSeq[1:] >=0)  # Check for ambiguious nucleotides

        # -------------------------------------------------------------------------
        # Nearest neighbor H and S from Santa Lucia Jr and Hicks, Annu.
        # Rev. Biomol. Struct. 2004 33:415-40. Units of kcal/mol (H) or
        # cal/(mol K) (S).
        # -------------------------------------------------------------------------
        H = [-7.6, -8.4, -7.8, -7.2, -8.5, -8.0, -10.6, -7.8,  # AA/TT GT/CA CT/GA AT/TA CA/GT GG/CC CG/GC CT/GA
             -8.2, -9.8, -8.0, -8.4, -7.2, -8.2, -8.5, -7.6]       # GA/CT GC/CG GG/CC GT/CA TA/AT GA/CT CA/GT AA/TT
        S = [-21.3, -22.4, -21.0, -20.4, -22.7, -19.9, -27.2, -21.0,    # AA/TT GT/CA CT/GA AT/TA CA/GT GG/CC CG/GC CT/GA
            -22.2, -24.4, -19.9, -22.4, -21.3, -22.2, -22.7, -21.3]       # GA/CT GC/CG GG/CC GT/CA TA/AT GA/CT CA/GT AA/TT
        # Note the different values for AA/TT from Santa Lucia Jr PNAS 1998

        # -------------------------------------------------------------------------
        # Nearest neighbour H and S from Santa Lucia Jr, PNAS 95, 1460-65
        # (1998)
        # -------------------------------------------------------------------------
#         H = [-7.9, -8.4, -7.8, -7.2, -8.5, -8.0, -10.6, -7.8,   # AA/TT GT/CA CT/GA AT/TA CA/GT GG/CC CG/GC CT/GA
#             -8.2, -9.8, -8.0, -8.4, -7.2, -8.2, -8.,5 -7.6]     # GA/CT GC/CG GG/CC GT/CA TA/AT GA/CT CA/GT AA/TT
#         S = [-22.2, -22.4, -21.0, -20.4, -22.7, -19.9, -27.2, -21.0,   # AA/TT GT/CA CT/GA AT/TA CA/GT GG/CC CG/GC CT/GA
#             -22.2, -24.4, -19.9, -22.4, -21.3, -22.2, -22.7, -21.3]    # GA/CT GC/CG GG/CC GT/CA TA/AT GA/CT CA/GT AA/TT

        # -------------------------------------------------------------------------
        # Define dG
        # -------------------------------------------------------------------------
        dG[0,isValid] = [H[int(i)] for i in nnID[isValid]]
        dG[1,isValid] = [S[int(i)] for i in nnID[isValid]]

        return dG
    
    # -------------------------------------------------------------------------
    # Tile Regions  
    # -------------------------------------------------------------------------
    @staticmethod
    def TileRegions(regionProps, padLength):
        # Identify a non-overlapping tiling of regions separated by at
        # least padLength
        # selectedRegionData = TRDesigner.TileRegions(regionProps, padLength)
        
        # -------------------------------------------------------------------------
        # Handle empty
        # -------------------------------------------------------------------------
        if regionProps.shape[1] <= 0:
            selectedRegionData = regionProps
            return selectedRegionData

        # -------------------------------------------------------------------------
        # Find start positions, sort, and find next available positions
        # -------------------------------------------------------------------------
        startPos = regionProps[0,:]
        sind = np.argsort(startPos)
        startPos = np.sort(startPos)
        
        regionProps = regionProps[:,sind] # Sort data
        nextAvailablePos = startPos + regionProps[1,:] + padLength
        
        # -------------------------------------------------------------------------
        # Tile probes
        # -------------------------------------------------------------------------
        done = False
        indsToKeep = [0]
        while not done:
            minNextPos = nextAvailablePos[indsToKeep[-1]] # Identify the minimum starting position
            newInd = np.where(startPos >= minNextPos)
            if len(newInd[0])==0:
                done = True
            else:
                indsToKeep.append(newInd[0][0])
            
        
        
        # -------------------------------------------------------------------------
        # Return probe data
        # -------------------------------------------------------------------------
        selectedRegionData = regionProps[:,indsToKeep]

        return selectedRegionData
        



