## ------------------------------------------------------------------------
#  PrimerDesigner Classs
## ------------------------------------------------------------------------
# Original version:
# Jeffrey Moffitt
# lmoffitt@mcb.harvard.edu
# jeffrey.moffitt@childrens.harvard.edu
# May 4, 2015
#--------------------------------------------------------------------------
# Copyright Presidents and Fellows of Harvard College, 2016.
#--------------------------------------------------------------------------

# --------------------------------------------------------------------------
# This python version is developed by Ruifeng Hu from the Original version
# 09-20-2021
# huruifeng.cn@hotmail.com
# --------------------------------------------------------------------------

import os
import re
import pickle

import numpy
import numpy as np
from scipy.signal import lfilter

from merfish.probe_construction.TRDesigner import TRDesigner
from utils.funcs import *

class PrimerDesigner:
    # ------------------------------------------------------------------------
    # [pDesignerObj] = PrimerDesigner(varargin)
    # This class designs orthogonal primers. 

    def __init__(self,**kwargs):
        # Create the PrimerDesigner object
        # -------------------------------------------------------------------------
        # Define properties
        # -------------------------------------------------------------------------
        self.verbose = True          # Determines the verbosity of the class
        self.ntComposition = [0.25, 0.25, 0.25, 0.25]      # A 4x1 vector that controls the composition of the primers (A,C,G,T)
        self.OTTables = []           # OTTables an array of Off-Target Tables for calculating penalties
        self.OTTableNames = []       # A cell array of string names for each Off-Target table
        self.monovalentSalt = 0.3     # The monovalent salt concentration (M) for Tm calculations
        self.primerConc = 0.5e-6        # The concentration of primer (M) for Tm calculations
        self.seqsToRemove = ['AAAA', 'TTTT', 'GGGG', 'CCCC']      # A cell array of sequences that are not allowed, e.g. GGGG

        self.numPrimers = ""         # The number of current primers in the class
        self.primerLength = 20       # The length of the primers
        self.seqs = []               # The sequences of the primers in integer format, a NxL matrix
        self.numPar = 1              # The number of current parallel workers
        self.homologyMax= 8         # The maximum homology length accepted
        self.gc=[]                   # The GC content for each sequence
        self.Tm=[]                   # The Tm (C) for each sequence
        self.penalties=[]            # The penalty values associated with each sequence for each OTTable

        self.homologyMat=[]          # A sparse matrix specifying the primer pairs that share homology
        self.seqHash=[]              # The hash values for all sequences
        self.seqRCHash=[]            # The hash values for the reverse complement of all sequences

        # -------------------------------------------------------------------------
        # Handle empty object request
        # -------------------------------------------------------------------------
        if len(kwargs) < 1:
            return

        ## initialize the values:
        if "verbose" not in kwargs: kwargs["verbose"] = True
        if "ntComposition" not in kwargs: kwargs["ntComposition"] = [0.25, 0.25, 0.25, 0.25]
        if "OTTables" not in kwargs: kwargs["OTTables"] = []
        if "OTTableNames" not in kwargs: kwargs["OTTableNames"] = []
        if "seqs" not in kwargs: kwargs["seqs"] = []
        if "primerLength" not in kwargs: kwargs["primerLength"] = 20
        if "numPrimersToGenerate" not in kwargs: kwargs["numPrimersToGenerate"] = 1e6
        if "homologyMax" not in kwargs: kwargs["homologyMax"] = 8
        if "monovalentSalt" not in kwargs: kwargs["monovalentSalt"] = 0.3
        if "primerConc" not in kwargs: kwargs["primerConc"] = 0.5e-6
        if "seqsToRemove" not in kwargs: kwargs["seqsToRemove"] = ['AAAA', 'TTTT', 'GGGG', 'CCCC']

        arg_ls = ["verbose", "ntComposition", "OTTables", "OTTableNames", "monovalentSalt", "primerConc", "seqsToRemove",
                  "numPrimers", "primerLength", "seqs", "numPar", "homologyMax", "gc", "Tm", "penalties", "homologyMat",
                  "seqHash", "seqRCHash"]
        parameters = ParseArguments(kwargs, arg_ls)
        # print(kwargs)

        if "loadfromfile" in kwargs and kwargs["loadfromfile"]:
            self.verbose = parameters["verbose"]
            self.ntComposition = parameters["ntComposition"]
            self.OTTables = parameters["OTTables"]
            self.OTTableNames = parameters["OTTableNames"]
            self.monovalentSalt = parameters["monovalentSalt"]
            self.primerConc = parameters["primerConc"]
            self.seqsToRemove = parameters["seqsToRemove"]
            self.numPrimers = parameters["numPrimers"]
            self.primerLength = parameters["primerLength"]
            self.seqs = parameters["seqs"]
            self.homologyMax = parameters["homologyMax"]
            self.gc = parameters["gc"]
            self.Tm = parameters["Tm"]
            self.penalties = parameters["penalties"]
            self.homologyMat = parameters["homologyMat"]
            self.seqHash = parameters["seqHash"]
            self.seqRCHash = parameters["seqRCHash"]
            self.numPar = 1
            return
        
        # Transfer values to object
        self.verbose = parameters["verbose"]
        self.ntComposition = parameters["ntComposition"]
        self.OTTables = parameters["OTTables"]
        self.OTTableNames = parameters["OTTableNames"]
        self.monovalentSalt = parameters["monovalentSalt"]
        self.primerConc = parameters["primerConc"]
        self.seqsToRemove = parameters["seqsToRemove"]

        self.numPrimers = parameters["numPrimers"]
        # self.primerLength = parameters["primerLength"]
        self.seqs = parameters["seqs"]
        self.numPar = parameters["numPar"]
        self.homologyMax = parameters["homologyMax"]
        self.gc = parameters["gc"]
        self.Tm = parameters["Tm"]
        self.penalties = parameters["penalties"]

        self.homologyMat = parameters["homologyMat"]
        self.seqHash = parameters["seqHash"]
        self.seqRCHash = parameters["seqRCHash"]

        # -------------------------------------------------------------------------
        # Check validity of input sequences if provided
        # -------------------------------------------------------------------------
        if len(self.seqs)>0:
            if isinstance(self.seqs[0], str):
                self.seqs = np.array([np.array(nt2int(i))-1 for i in self.seqs])
            if np.any(self.seqs > 3) or np.any(self.seqs < 0):
                error('[Error]:invalidArgument - invalid sequence')

            self.primerLength = self.seqs.shape[1]
            self.numPrimers = self.seqs.shape[0]
        else: # Generate randome sequences
            self.seqs = np.empty((0,self.primerLength))
            self.AddRandomSequences(**parameters)

        
    # -------------------------------------------------------------------------
    # Generate Random Sequences 
    # -------------------------------------------------------------------------
    def AddRandomSequences(self, **kwargs):
        # Add random sequences to the class
	
        # -------------------------------------------------------------------------
        # Handle variable input
        # -------------------------------------------------------------------------
        ## initialize the values:
        if "ntComposition" not in kwargs: kwargs["ntComposition"] = [0.25, 0.25, 0.25, 0.25]
        if "primerLength" not in kwargs: kwargs["primerLength"] = 20
        if "numPrimersToGenerate" not in kwargs: kwargs["numPrimersToGenerate"] = 1e6

        arg_ls = ["ntComposition", "primerLength","numPrimersToGenerate"]
        parameters = ParseArguments(kwargs, arg_ls)

        # -------------------------------------------------------------------------
        # Parse nucleotide composition 
        # -------------------------------------------------------------------------
        if len(parameters["ntComposition"])>0:
            if len(parameters["ntComposition"]) != 4:
                error('[Error]: nt composition must have four entries')
            parameters["ntComposition"] = np.array(parameters["ntComposition"])
            parameters["ntComposition"] = parameters["ntComposition"]/sum(parameters["ntComposition"])
        else:
            parameters["ntComposition"] = 0.25 * np.ones((1,4))
        
        # -------------------------------------------------------------------------
        # Display Progress  
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Creating ', parameters["numPrimersToGenerate"],
                  'new primers of ',parameters["primerLength"], '-nt length')

        # -------------------------------------------------------------------------
        # Create random sequences: Following randseq
        # -------------------------------------------------------------------------
        np.random.seed(12)
        ## MATLAB
        # rseq = rand(parameters.numPrimersToGenerate, parameters.primerLength)
        # edges = [0, cumsum(parameters.ntComposition)]
        # edges(end) = 1
        # [~, seq] = histc(rseq, edges)
        # seq = int8(seq - 1) # Shift

        ## Python v1
        # rseq = np.random.rand(parameters["numPrimersToGenerate"], parameters["primerLength"])
        # edges = [0] + list(np.cumsum(parameters["ntComposition"]))
        # edges.append(1)
        #
        # seq = np.digitize(rseq, edges)
        # seq = np.array(seq, dtype=np.int) - 1  # Shift

        ##Python v2
        seq = np.random.randint(4, size=(int(parameters["numPrimersToGenerate"]), int(parameters["primerLength"])))

        ## For debug
        seq = np.loadtxt("F:\\Harvard_BWH\projects\\1001_MERFISH\\MERFISH_analysis\\random_seq_matlab.txt",dtype=np.int, delimiter=",")

        # -------------------------------------------------------------------------
        # Update sequences
        # -------------------------------------------------------------------------
        self.seqs = np.concatenate((self.seqs,seq),axis=0)
        self.primerLength = self.seqs.shape[1]
        self.numPrimers = self.seqs.shape[0]

        # -------------------------------------------------------------------------
        # Recalculate Tm and GC and penalties
        # -------------------------------------------------------------------------
        self.CalculatePrimerProperties()


    # -------------------------------------------------------------------------
    # AddPrimer
    # -------------------------------------------------------------------------
    def AddPrimer(self, seq=""):
        # Add a specific primer sequence to the list of sequences
        # self.AddPrimer('ACTG....')

        # -------------------------------------------------------------------------
        # Check input sequence
        # -------------------------------------------------------------------------
        if not(seq!="" and isinstance(seq,str)):
            error('[Error]:invalidArguments - A valid sequence string must be provided')
        if len(seq) != self.primerLength:
            error('[Error]:invalidArguments - The provided sequence must match the length of all primers')

        # -------------------------------------------------------------------------
        # Convert if necessary and check the sequence space
        # -------------------------------------------------------------------------

        intSeq = np.array(nt2int(seq, ACGT=True),dtype=np.float)-1
        if  np.any(self.seqs > 3) or np.any(self.seqs < 0):
            error('[Error]:invalidArguments - The sequence can only contain A, C, T, or G')

        # -------------------------------------------------------------------------
        # Add to the primer list and update the number of primers
        # -------------------------------------------------------------------------
        self.seqs = np.concatenate(self.seqs, intSeq,axis=0)
        self.numPrimers = self.seqs.shape[0]

        # -------------------------------------------------------------------------
        # Recalculate Tm and GC and penalties
        # -------------------------------------------------------------------------
        self.CalculatePrimerProperties() ## IN FUTURE VERSIONS THIS COULD BE DONE MORE EFFICIENTLY

    # -------------------------------------------------------------------------
    # Calculate Primer Properties
    # -------------------------------------------------------------------------
    def CalculatePrimerProperties(self, **kwargs):
        # Calculate the Tm, GC, and penalties of all sequences
        # self.CalculatePrimerProperties()
        # self.CalculatePrimerProperties(..., 'monovalentSalt',saltConcInM)
        # self.CalculatePrimerProperties(..., 'primerConc',primerConcInM)

        # -------------------------------------------------------------------------
        # Handle variable input
        # -------------------------------------------------------------------------
        if "monovalentSalt" not in kwargs: kwargs["monovalentSalt"] = ""
        if "primerConc" not in kwargs: kwargs["primerConc"] = ""

        arg_ls = ["monovalentSalt", "primerConc"]
        parameters = ParseArguments(kwargs, arg_ls)

        # -------------------------------------------------------------------------
        # Update object salt and probe concentrations
        # -------------------------------------------------------------------------
        if parameters["monovalentSalt"] != "":
            self.monovalentSalt = parameters["monovalentSalt"]

        if parameters["primerConc"] != "":
            self.primerConc = parameters["primerConc"]

        # -------------------------------------------------------------------------
        # Display Progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Calculating Tm and GC for ',self.numPrimers,' primers with ',
                  self.monovalentSalt, ' M salt and ', self.primerConc, ' M primer concentration')
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Calculate primer GC
        # -------------------------------------------------------------------------
        filterLength = self.primerLength
        zi_ndim = self.numPrimers

        self.gc = lfilter(np.ones(filterLength)/filterLength, 1,
                          (self.seqs == 1) | (self.seqs == 2), axis=1,zi=np.zeros((zi_ndim,filterLength-1)))[0]
        self.gc = self.gc[:, filterLength-1:]

        # -------------------------------------------------------------------------
        # Calculate Tm
        # -------------------------------------------------------------------------
        self.Tm = np.empty((self.numPrimers,1))
        self.Tm[:] = np.nan
        for i in range(self.numPrimers):
            # Derive enthalpy and entropy
            dG = TRDesigner.SantaLuciaNearestNeighbor(self.seqs[i,:])
            dG = np.sum(dG,1)

            # Calculate 5' and 3' corrections
            fivePrimeAT = (self.seqs[i,0] == 0) or (self.seqs[i,0] == 3)
            threePrimeAT = (self.seqs[i,-1] == 0) or (self.seqs[i,-1] == 3)

            dG[0] = dG[0] + 0.2 + 2.2*fivePrimeAT + 2.2*threePrimeAT
            dG[1] = dG[1] + -5.7 + 6.9*fivePrimeAT + 6.9*threePrimeAT

            # Apply salt corrections
            dG[1] = dG[1] + 0.368*(self.primerLength-1)* np.log(self.monovalentSalt)

            # Calculate Tm
            self.Tm[i] = dG[0]*1000 / (dG[1] + 1.9872 * np.log(self.primerConc)) - 273.15

            # NOTE: For the future, I should provide two concentrations,
            # probe and target, and this should be log(probeC - targetC/2)
            # where probeC > targetC.

        # -------------------------------------------------------------------------
        # Display Progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer), 's')

        # -------------------------------------------------------------------------
        # Calculate Penalty Values
        # -------------------------------------------------------------------------
        self.penalties = np.empty((self.numPrimers, len(self.OTTables)))
        self.penalties[:] = np.nan
        for o in range(len(self.OTTables)):
            if self.verbose:
                PageBreak()
                print('Calculating penalty for the <',self.OTTableNames[o],
                      '> table with seed length:', self.OTTables[o].seedLength)
                timer = tic(99)

            for s in range(self.numPrimers):
                # Calculate total penalty for the sequence and for its
                # reverse complement
                self.penalties[s,o] = np.sum(self.OTTables[o].CalculatePenalty(self.seqs[s,:])[0]) + \
                                      sum(self.OTTables[o].CalculatePenalty((3-self.seqs[s,:])[::-1])[0])

            if self.verbose:
                print('... completed in ', toc(timer), ' s')

    # -------------------------------------------------------------------------
    # Cut primers on GC, Tm, or penalty
    # -------------------------------------------------------------------------
    def CutPrimers(self, Tm=[],GC=[],OTTables={}):
        # Cut primers based on their GC, Tm, or penalty
        # self.CutPrimers(..., 'Tm', [low,up])
        # self.CutPrimers(..., 'GC', [low,up])
        # self.CutPrimers(..., 'OTTables', {'name', [low, up], 'name', range'})

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Keeping primers with ')
            if len(GC) > 0:
                print('   >>GC in [', GC[0], ',', GC[1], ']')
            if len(Tm) > 0:
                print('   >>Tm in [',Tm[0],',',Tm[1], ']')

            if len(OTTables) > 0:
                for t in OTTables:
                    if t not in self.OTTableNames:
                        error('[Error] CutPrimers() - Unrecognized OTTable name');
                    pid = self.OTTableNames.index(t)
                    print('   >>[',self.OTTableNames[pid],
                          '] penalty in [', OTTables[t][0],',', OTTables[t][1], ']')
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Cut properties if needed
        # -------------------------------------------------------------------------
        indsToKeep = np.full((self.numPrimers,1),True, dtype=bool)
        if len(GC) > 0:
            indsToKeep = indsToKeep & (self.gc >= GC[0]) & (self.gc <= GC[1])
        if len(Tm) > 0:
            indsToKeep = indsToKeep & (self.Tm >= Tm[0]) & (self.Tm <= Tm[1])

        if len(OTTables) > 0:
            if t not in self.OTTableNames:
                error('[Error] DesignTargetRegions() - Unrecognized OTTable name')
            pid = self.OTTableNames.index(t)
            indsToKeep = indsToKeep & (self.penalties[:,[pid]] >= OTTables[t][0]) & (self.penalties[:,[pid]] <= OTTables[t][1])

        # -------------------------------------------------------------------------
        # Cut oligos
        # -------------------------------------------------------------------------
        indsToKeep = indsToKeep.ravel()
        self.seqs = self.seqs[indsToKeep,]
        self.numPrimers = np.sum(indsToKeep)
        self.gc = self.gc[indsToKeep]
        self.Tm = self.Tm[indsToKeep]
        self.penalties = self.penalties[indsToKeep,]

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('...completed in ',toc(timer), ' s')
            print('   >> Removed ', np.sum(np.logical_not(indsToKeep)), ' primers')
            print('   >> Kept ', np.sum(indsToKeep), ' primers')

        return indsToKeep

    # -------------------------------------------------------------------------
    # Remove sequences that are not permitted
    # -------------------------------------------------------------------------
    def RemoveForbiddenSeqs(self, seqsToRemove=[]):
        # Remove primers based on self complementarity
        # self.RemoveForbiddenSeqs()
        # self.RemoveForbiddenSeqs('seqsToRemove', {'seq1','seq2',...})

        # -------------------------------------------------------------------------
        # Update object properties
        # -------------------------------------------------------------------------
        if len(seqsToRemove) > 0:
            self.seqsToRemove = seqsToRemove

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Removing forbidden sequences')
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Loop through forbidden sequences
        # -------------------------------------------------------------------------
        indsToKeep = np.full((1, self.numPrimers),True, dtype=bool)
        for s in range(len(self.seqsToRemove)):
            # Display progress
            if self.verbose:
                print('... finding ', self.seqsToRemove[s])

            # Convert forbidden sequence
            intSeq = np.array(nt2int(self.seqsToRemove[s], ACGT=True),dtype=np.float)-1
            if np.any(intSeq < 0):
                error('[Error]:invalidArguments - Invalid forbidden sequence')

            # Hash forbidden sequence
            hashBase = [4.0** i for i in range(len(intSeq))][::-1]
            forbiddenHash = np.sum(hashBase * intSeq) + 1

            # Hash sequence
            self.seqHash = lfilter(hashBase, 1, self.seqs, axis=1,
                                   zi=np.zeros((self.numPrimers, len(hashBase)-1)))[0] + 1
            self.seqHash = self.seqHash[:,len(intSeq)-1:]

            # Find matches
            indsToKeep = indsToKeep & np.logical_not(np.any(self.seqHash == forbiddenHash,axis=1))

        # -------------------------------------------------------------------------
        # Update primers
        # -------------------------------------------------------------------------
        indsToKeep = indsToKeep.reshape((self.numPrimers,))

        self.seqs = self.seqs[indsToKeep,]
        self.numPrimers = self.seqs.shape[0]
        self.gc = self.gc[indsToKeep]
        self.Tm = self.Tm[indsToKeep]
        self.penalties = self.penalties[indsToKeep,]

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer), ' s')
            print('>>> Removed ', np.sum(np.logical_not(indsToKeep)), ' primers')
            print('>>> Kept ', np.sum(indsToKeep), ' primers')

        return indsToKeep

    # -------------------------------------------------------------------------
    # Remove primers based on internal self complementarity
    # -------------------------------------------------------------------------
    def RemoveSelfCompPrimers(self, homologyMax=""):
        # Remove primers based on self complementarity
        # self.RemoveSelfCompPrimers()
        # self.RemoveSelfCompPrimers('homologyMax', homologyRegionLength)

        # -------------------------------------------------------------------------
        # Update homology max property
        # -------------------------------------------------------------------------
        if homologyMax != "" and homologyMax > 0:
            self.homologyMax = homologyMax
        else:
            error("[Error]: RemoveSelfCompPrimers() - Invalid homologyMax value.")

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Identifying internal homology within each primer')
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Hash the sequences
        # -------------------------------------------------------------------------
        hashBase = [4.0 ** i for i in range(self.homologyMax)][::-1]
        self.seqHash = lfilter(hashBase, 1, self.seqs, axis=1,
                               zi=np.zeros((self.numPrimers, len(hashBase) - 1)))[0] + 1
        self.seqHash = self.seqHash[:,self.homologyMax-1:]

        # Hash the reverse complement: 3-seq = A=0->3=T, C=1->2=G
        self.seqRCHash = lfilter(hashBase,1, np.fliplr(3-self.seqs), axis=1,
                                zi=np.zeros((self.numPrimers,len(hashBase)-1)))[0] + 1
        self.seqRCHash = self.seqRCHash[:,self.homologyMax-1:]

        # -------------------------------------------------------------------------
        # Scan and remove primers based on internal homology
        # -------------------------------------------------------------------------
        indsToKeep = np.full((self.numPrimers,1),True, dtype=bool)
        for s in range(self.numPrimers):
            indsToKeep[s] = np.any(np.in1d(self.seqHash[s,:], self.seqRCHash[s,:]))

        indsToKeep = np.logical_not(indsToKeep)
        # -------------------------------------------------------------------------
        # Update primers
        # -------------------------------------------------------------------------
        indsToKeep = indsToKeep.ravel()
        self.seqs = self.seqs[indsToKeep, ]
        self.numPrimers = self.seqs.shape[0]
        self.gc = self.gc[indsToKeep]
        self.Tm = self.Tm[indsToKeep]
        self.penalties = self.penalties[indsToKeep, ]

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer), ' s')
            print('>>> Removed ', np.sum(np.logical_not(indsToKeep)), ' primers')
            print('>>> Kept ', np.sum(indsToKeep), ' primers')

        return indsToKeep

    # -------------------------------------------------------------------------
    # Remove primers based on cross homology
    # -------------------------------------------------------------------------
    def RemoveHomologousPrimers(self, homologyMax=""):
        # Remove primers which share homologous regions
        # self.RemoveHomologousPrimers()
        # self.RemoveHomologousPrimers('homologyMax', homologyRegionLength)

        # -------------------------------------------------------------------------
        # Update object salt and probe concentrations
        # -------------------------------------------------------------------------
        if homologyMax != "" and homologyMax > 0:
            self.homologyMax = homologyMax
        else:
            error("[Error]: RemoveHomologousPrimers() - Invalid homologyMax value.")

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Identifying cross homology within primers')
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Hash the sequences
        # -------------------------------------------------------------------------
        hashBase = [4.0 ** i for i in range(self.homologyMax)][::-1]
        self.seqHash = lfilter(hashBase, 1, self.seqs, axis=1,
                               zi=np.zeros((self.numPrimers, self.homologyMax - 1)))[0] + 1
        self.seqHash = self.seqHash[:, self.homologyMax - 1:]

        # Hash the reverse complement: 3-seq = A=0->3=T, C=1->2=G
        self.seqRCHash = lfilter(hashBase, 1, np.fliplr(3 - self.seqs), axis=1,
                                zi=np.zeros((self.numPrimers, self.homologyMax - 1)))[0] + 1
        self.seqRCHash = self.seqRCHash[:, self.homologyMax - 1:]

        # -------------------------------------------------------------------------
        # Identify unique hashes
        # -------------------------------------------------------------------------
        seq_seqR_hash = np.concatenate((self.seqHash.T.reshape(-1,1), self.seqRCHash.T.reshape(-1,1)),axis=1)
        [uniqueHash, ix, ic] = np.unique(seq_seqR_hash.T, return_index = True, return_inverse=True)

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('Identified ', len(uniqueHash), ' unique ', self.homologyMax,'-mers')

        # -------------------------------------------------------------------------
        # Create homology mat: non-zero entries indicate a shared sequence
        # -------------------------------------------------------------------------
        self.homologyMat = np.zeros((self.seqHash.shape[0], self.seqHash.shape[0]))

        # Loop over unique hash values
        for i in range(len(uniqueHash)):
            localIDs = np.where(ic == i) # find inds of sequences that hold these values
            seqIDs = localIDs[0] % self.seqHash.shape[0] # Map to sequence id
            # Loop over all sequence ids
            for k in range(len(seqIDs)):
                for l in range(len(seqIDs)):
                    if k!=l: # Don't mark self homology
                        self.homologyMat[seqIDs[k], seqIDs[l]] = 1

            if self.verbose and ((i+1)%1000==0):
                print('... completed ', i+1,' regions')

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer), 's')

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Removing primers with cross homology')
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Prepare row sum
        # -------------------------------------------------------------------------
        rowSum = np.sum(self.homologyMat,axis=0)

        # -------------------------------------------------------------------------
        # Iterate until no cross homology
        # -------------------------------------------------------------------------
        count = 1
        while np.any(rowSum>0):
            maxID = np.argmax(rowSum) # Find primer with the most connections
            rowSum = rowSum - self.homologyMat[maxID,] # Delete its connections
            rowSum[maxID] = -np.Inf # Flag as removed
            count = count + 1
            if (count%1000==0) and self.verbose:
                print('... completed ', count, ' iterations')

        # -------------------------------------------------------------------------
        # Update primers
        # -------------------------------------------------------------------------
        indsToKeep = rowSum == 0

        self.seqs = self.seqs[indsToKeep,]
        self.numPrimers = self.seqs.shape[0]
        self.gc = self.gc[indsToKeep]
        self.Tm = self.Tm[indsToKeep]
        self.penalties = self.penalties[indsToKeep,]

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer), 's')
            print('Identified ',self.numPrimers, ' orthogonal primers')

        return indsToKeep

    # -------------------------------------------------------------------------
    # Write fasta files
    # -------------------------------------------------------------------------
    def WriteFasta(self, filePath="", namePrefix="",fieldPad=""):
        # Write a fasta file
        # self.WriteFasta(filePath)
        # self.WriteFasta(..., 'namePrefix', namePrefixValue)


        # -------------------------------------------------------------------------
        # Generate random prefix
        # -------------------------------------------------------------------------
        if namePrefix=="":
            import uuid
            namePrefix = str(uuid.uuid4())
            namePrefix = namePrefix[:8]


        # -------------------------------------------------------------------------
        # Determine field padding
        # -------------------------------------------------------------------------
        if fieldPad=="":
            fieldPad = int(np.ceil(np.log10(self.numPrimers)))

        # -------------------------------------------------------------------------
        # Check file path
        # -------------------------------------------------------------------------
        if filePath =="":
            error('[Error]:invalidArguments - A valid file path must be provided')

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Writing fasta: ', filePath)
            timer = tic(99)

        # -------------------------------------------------------------------------
        # Delete existing files
        # -------------------------------------------------------------------------
        if os.path.exists(filePath):
            os.remove(filePath)
            if self.verbose:
                print('... Deleted existing file')

        # -------------------------------------------------------------------------
        # Build object
        # -------------------------------------------------------------------------
        seqs = {}
        for s in range(self.numPrimers):
            header = namePrefix+'-'+str(s+1).zfill(fieldPad)+' Tm='+str(round(self.Tm[s][0],4))+' GC='+str(round(self.gc[s][0],2))
            seqs[header] = int2nt(self.seqs[s,:]+1)

        # -------------------------------------------------------------------------
        # Write fasta
        # -------------------------------------------------------------------------
        fastawrite(filePath, seqs)

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if self.verbose:
            print('... completed in ', toc(timer), 's')

    # -------------------------------------------------------------------------
    # Save def
    # -------------------------------------------------------------------------
    def Save(self, dirPath):
        # Save the primer designer object in a directory specified by dirPath
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
            "verbose": self.verbose,
            "ntComposition": self.ntComposition,
            "OTTables": self.OTTables,
            "OTTableNames": self.OTTableNames,
            "monovalentSalt": self.monovalentSalt,
            "primerConc": self.primerConc,
            "seqsToRemove": self.seqsToRemove,
            "numPrimers": self.numPrimers,
            "primerLength": self.primerLength,
            "seqs": self.seqs,
            "homologyMax": self.homologyMax,
            "gc": self.gc,
            "Tm": self.Tm,
            "penalties": self.penalties,
            "homologyMat": self.homologyMat,
            "seqHash": self.seqHash,
            "seqRCHash": self.seqRCHash
        }

        with open(dirPath + "/PrimerDesigner.pkl", "wb") as fout:
            pickle.dump(save_dict, fout, pickle.HIGHEST_PROTOCOL)

    # -------------------------------------------------------------------------
    # Static methods
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Build a PrimerDesigner object from a saved version
    # -------------------------------------------------------------------------
    def Load(filePath):
        # obj = PrimerDesigner.Load(filePath)

        # -------------------------------------------------------------------------
        # Check provided path
        # -------------------------------------------------------------------------
        if not os.path.exists(filePath):
            error('[Error]:invalidArguments - Invalid directory path for loading the PrimerDesigner object.')

        with open(filePath, 'rb') as fin:
            loaded_dict = pickle.load(fin)
        # -------------------------------------------------------------------------
        # Create empty object (to define fields to load)
        # -------------------------------------------------------------------------
        obj = PrimerDesigner(verbose=loaded_dict["verbose"],
                             ntComposition=loaded_dict["ntComposition"],
                             OTTables= loaded_dict["OTTables"],
                             OTTableNames=loaded_dict["OTTableNames"],
                             monovalentSalt=loaded_dict["monovalentSalt"],
                             primerConc=loaded_dict["primerConc"],
                             seqsToRemove=loaded_dict["seqsToRemove"],
                             numPrimers=loaded_dict["numPrimers"],
                             primerLength=loaded_dict["primerLength"],
                             seqs=loaded_dict["seqs"],
                             homologyMax=loaded_dict["homologyMax"],
                             gc=loaded_dict["gc"],
                             Tm=loaded_dict["Tm"],
                             penalties=loaded_dict["penalties"],
                             homologyMat=loaded_dict["homologyMat"],
                             seqHash=loaded_dict["seqHash"],
                             seqRCHash=loaded_dict["seqRCHash"],
                             loadfromfile=True
                             )
        return obj
        
