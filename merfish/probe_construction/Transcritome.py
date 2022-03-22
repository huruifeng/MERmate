## ------------------------------------------------------------------------
#  Transcriptome Classs
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
import re
import pickle
from utils.funcs import *

class Transcriptome:
    # ------------------------------------------------------------------------
    # transcriptomeObj = Transcriptome(sequenceData, **kwargs)
    # This class contains information about a transcriptome, including
    # sequences, gene names, gene IDs, expression data, and isoform
    # information.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # targetSequences -- Either the path to a fasta file that contains target
    # names or a structure array that has the same organization as the output
    # of fastaread.
    #--------------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # 'abundPath' -- A path to a cufflinks isoforms.fpkm_tracking file.
    # 'headerType' -- Reserved for future expansion of abundance data types
    # 'IDType' -- Reserved for future expansion of isoform/gene ID processing

    ###################################################################################
    # -------------------------------------------------------------------------
    # Define properties
    # -------------------------------------------------------------------------

    ## class varible with default values
    numTranscripts=0       # The number of loaded transcripts
    numGenes=0             # The number of loaded genes
    verbose=True           # The verbose status of the class
    headerType="cufflinks"          # The type of the abundance data file -- reserved for future use
    IDType=""              # The type of the transcript ID -- reserved for future use
    abundPath=""           # The path to the abundance data file
    abundLoaded=False      # A boolean that determines if abundance data are loaded
    transPath=""           # The path to the transcriptome file, if provided
    version = '0.1-alpha'  # Version of the transcriptome object

    # -------------------------------------------------------------------------
    # Define constructor
    # -------------------------------------------------------------------------
    def __init__(self, transcriptome="", **kwargs):
        # Create the transcriptome object
        # obj = Transcriptome(fastaread(...))
        # obj = Transcriptome(transcriptomePath)
        # obj = Transcriptome(..., abundPath=pathToAbundanceFile)
        # obj = Transcriptome([ids, geneNames, sequences, abund, cds], **kwargs)

        # instance attributes with default values
        self.ids = []        # Accession values for each entry
        self.geneNames = []  # Common name for each entry
        self.Sequences = {}  # the sequence of each entry
        self.abundance = {}  # Abundance value for each entry, if loaded
        self.cds = {}       # A {id:[start,stop],...} dict of the start and stop places of a cds. -1 -1 if no CDS exists
        self.id2name = {}     # A map for conversion of ID to internal index
        self.name2id = {}   # A map for converstion of gene name to internal index
        self.idVersion = {}   # A version number for the transcripts
        self.transcriptomeHeader = []  # FASTA header for each entry

        ## initialize the values:
        for k,v in kwargs.items() :
            if k == "numTranscripts": Transcriptome.numTranscripts = v
            if k == "numGenes": Transcriptome.numGenes = v
            if k == "verbose": Transcriptome.verbose = v
            if k == "headerType": Transcriptome.headerType = v
            if k == "IDType": Transcriptome.IDType = v
            if k == "abundPath": Transcriptome.abundPath = v
            if k == "abundLoaded": Transcriptome.abundLoaded = v
            if k == "transPath": Transcriptome.transPath = v
            if k == "version": Transcriptome.version = v

        # print(kwargs)

        # -------------------------------------------------------------------------
        # Parse necessary input
        # -------------------------------------------------------------------------
        if transcriptome == "": # Define empty class
            return
        elif isinstance(transcriptome, str): # A path string was provided
            if os.path.exists(transcriptome): # Does this path exist
                if Transcriptome.verbose:
                    print('Loading transcriptome from ', transcriptome)
                    print('Start at: ', tic(1))
                    st = tic(0)
                Transcriptome.transPath = transcriptome
                transcriptome = fastaread(transcriptome)
                if Transcriptome.verbose:
                    print('Found ',len(transcriptome),' sequences')
                    print('... completed in', (tic(0) - st).total_seconds(), "s")
            else:
                error('Error: Transcriptome() - Invalid path to target file:' + transcriptome)

        elif isinstance(transcriptome, dict) : # A fasta dict was provided
            if len(transcriptome)==0:
                error('Error: Transcriptome() - Transcriptome dict is empty.')

        elif isinstance(transcriptome,list) and (len(transcriptome) >= 3):
            # Data provided in the cell as the format:
            #transcriptome = [ids[..], geneNames[...], seqs[...], abundance(optional), cds (optional), idVersions (optional)]
            if len(transcriptome[0]) == 0:
                error("[Error]: Transcriptome() - No ID provided, the Transcriptome cannot be constructed.")

            # Insert ids and gene names and cds and idVersion info
            if len(transcriptome[0]) != len(transcriptome[1]):
                error("[Error]: Transcriptome() - #IDs is not equal to #geneNames.")
            if len(transcriptome[0]) != len(transcriptome[2]):
                error("[Error]: Transcriptome() - #IDs is not equal to #sequence.")

            self.ids = transcriptome[0]
            self.geneNames = transcriptome[1]
            self.Sequences = dict(zip(self.ids,transcriptome[2]))

            # Insert abundances if provided
            if len(transcriptome) > 3:
                if len(transcriptome[0]) != len(transcriptome[3]):
                    error("[Error]: Transcriptome() - #IDs is not equal to #abundance values.")

                for idx in range(len(self.ids)):
                    self.abundance[self.ids[idx]] = float(transcriptome[3][idx]) if transcriptome[3][idx]!="NA" else 0.0
                Transcriptome.abundLoaded = True

                if len(self.abundance) == 0:
                    Transcriptome.abundLoaded = False
            else:
                Transcriptome.abundLoaded = False

            # Handle the case that no cds was provided
            if len(transcriptome) < 5:
                self.cds = dict(zip(self.ids,[[-1,-1]] * len(self.ids)))

            # Handle the case that no idVersions was provided
            if len(transcriptome) < 6:
                self.idVersion = dict(zip(self.ids,[""] * len(self.ids)))

        else:
            error('Error: Transcriptome() - Format of argument[transcriptome] is not supported. [line 152]')

        # -------------------------------------------------------------------------
        # Parse Headers into IDs and gene names
        # -------------------------------------------------------------------------
        if len(self.ids)==0: # Handle the direct construction
            if Transcriptome.verbose:
                PageBreak()
                print('Parsing transcriptome IDs and Gene names from header records.')
                print('Start at: ', tic(1))
                st = tic(0)
                print("headerType:", Transcriptome.headerType)
            if Transcriptome.headerType == 'cufflinks':

                for i in transcriptome:
                    # Parse the header
                    self.transcriptomeHeader.append(i)
                    parsedData = re.search(r"(?P<id>\S*) (?P<name>gene=\S*)",i)
                    temp_name = parsedData.group("name")[5:] # Strip off "gene="
                    temp_id = parsedData.group("id")

                    self.geneNames.append(temp_name)
                    self.ids.append(temp_id)
                    self.Sequences[temp_id] = transcriptome[i]

                    # Handle a transcript version if available
                    t_version = re.search('transcript_version=\S*',i)
                    if not t_version:
                        self.idVersion[temp_id] = ""
                    else:
                        self.idVersion[temp_id] = t_version.group(0)[19:]

                    # Add cds infomation if availablesort cds index
                    CDS_transcr = re.search('(CDS=\S*)',i)
                    if CDS_transcr:
                        cds_index = re.search(r"(?P<first>\d+)-(?P<second>\d+)",CDS_transcr.group(0))
                        self.cds[temp_id] = [int(cds_index.group("first")), int(cds_index.group("second"))]
                    else:
                        self.cds[temp_id] = [-1, -1]  # Flag that no CDS was available

            elif Transcriptome.headerType == 'ensembl':
                for i in transcriptome:
                    self.transcriptomeHeader.append(i)
                    parsedData = re.search(r'(?P<id> gene:\S*)',i)
                    temp_id = parsedData.group("id")[6:]
                    self.ids.append(temp_id)
                    self.geneNames.append('') # Gene name not provided in header
                    self.Sequences[temp_id] = transcriptome[i]
            else:
                error("[Error]: Transcriptome() - Header type setting error. It should be either 'cufflinks' or 'ensembl'")

            if Transcriptome.verbose:
                print('... completed in ',(tic(0)-st).total_seconds(),'s')

        # -------------------------------------------------------------------------
        # Set abundances
        # -------------------------------------------------------------------------
        if len(self.abundance)==0:
            if isinstance(Transcriptome.abundPath,str):
                self.AddAbundances(Transcriptome.abundPath)
            else:
                self.abundance = dict(zip(self.ids, [1] * self.ids)) # Updated from zeros to equal weighting of 1

        # Update the internal storage/indexing
        self.UpdateIDNameDict()


    ## Update ID-Name dict
    def UpdateIDNameDict(self):
        self.id2name.clear()
        self.name2id.clear()

        self.id2name = dict(zip(self.ids, self.geneNames))
        for idx in range(len(self.ids)):
            if self.geneNames[idx] not in self.name2id:
                self.name2id[self.geneNames[idx]] = [self.ids[idx]]
            else:
                self.name2id[self.geneNames[idx]].append(self.ids[idx])

        Transcriptome.numTranscripts = len(self.id2name)
        Transcriptome.numGenes = len(self.name2id)


    # -------------------------------------------------------------------------
    # Add entry
    # -------------------------------------------------------------------------
    def AddEntries(self, names, ids, seqs, abunds=[], cds=[], idVersions=[], overwrite=True):
        # Add entries to the transcriptome object after it is created
        # self.AddEntries(name[...],id[...], seq[...])
        # self.AddEntries(name[...],id[...], seq[...], abund[...])
        # self.AddEntries(name[...],id[...], seq[...], abund[...], cds[[cds1_start, cds1_end] ...])
        # self.AddEntries(name[...],id[...], seq[...], abund[...], cds[[cds1_start,cds1_end] ...], idVersion[...])

        #
        if Transcriptome.verbose:
            print('Adding sequences')

        if len(names) != len(ids): error("'Error: AddEntries() - #ids is not equal to #names.")
        if len(ids) != len(seqs): error("'Error: AddEntries() - #IDs is not equal to #sequence.")

        overlap_ids = [i for i in ids if i in self.ids]
        if len(overlap_ids) > 1:
            print("[Warning]: the following IDs are already in the Transcriptome, values will be overwrote. \n"
                  "Or setting 'overwrite=False' to keep the existing values." )
            print(overlap_ids)

        # Handle backwards compatibility for no provided idVersions
        if not idVersions:
            idVersions = [''] *  len(names)

        # Handle backwards compatibility for no provided cds
        if (not cds) and abunds:
            if len(ids) != len(abunds): error("[Error]: AddEntries() - #IDs is not equal to #abunds values.")
            cds = dict(zip(ids,[[-1,-1]] * len(abunds)))

        # Append new values to the end of the existing values

        for id_idx in range(len(ids)):
            if ids[id_idx] in self.ids:
                if overwrite:
                    existing_idx = self.ids.index(ids[id_idx])
                    self.geneNames[existing_idx] = names[id_idx]
                    self.abundance[ids[id_idx]] = abunds[id_idx]
                    self.Sequences[ids[id_idx]] = seqs[id_idx]
                    self.cds[ids[id_idx]] = cds[id_idx]
                    self.idVersion[existing_idx] = idVersions
                else:
                    pass
            else:
                self.ids.append(ids[id_idx])
                self.geneNames.append(names[id_idx])
                self.abundance[ids[id_idx]]= abunds[id_idx]
                self.Sequences[ids[id_idx]]= seqs[id_idx]
                self.cds[ids[id_idx]]= cds[id_idx]
                self.idVersion.append(idVersions)

        # Update the internal storage/indexing
        self.UpdateIDNameDict()

        if Transcriptome.verbose:
            print('Added ',len(names), ' sequences')

    # -------------------------------------------------------------------------
    # AddAbundances: set the abundance values to transcripts
    # -------------------------------------------------------------------------
    def AddAbundances(self, *varargin): ## return [notIn, notIncluded]
        # Add abundances to transcriptome
        # self.AddAbundances(pathToAbundanceFile)
        # self.AddAbundances(ids, abundVec)
        # Currently only supports cufflinks .fpkm_tracking files

        # print(varargin)

        # -------------------------------------------------------------------------
        # Check file type and path
        # -------------------------------------------------------------------------
        if len(varargin) == 1:
            abundancePath = varargin[0]
            if not os.path.exists(abundancePath):
                error('Error: AddAbundances() - Invalid path to abundance data')

            if not abundancePath.endswith('.fpkm_tracking'):
                error('Error: AddAbundances() - Only fpkm_tracking files are currently supported.')

            # -------------------------------------------------------------------------
            # Load file and extract ids and fpkm
            # -------------------------------------------------------------------------
            if Transcriptome.verbose:
                print('Loading abundances from ',abundancePath)
                print('Start at: ', tic(1))
                st = tic(0)

            fpkm = {}
            print("Reading abundances data...")
            with open(abundancePath) as fp:
                next(fp) ## skip the header
                for line in fp:
                    line_ls = line.split("\t")
                    fpkm[line_ls[0]] = float(line_ls[9])

            print(f"Read in {len(fpkm)} abundance records.")

        elif (len(varargin) == 2) and (len(varargin[0])>0) and (len(varargin[1])>0):
            # -------------------------------------------------------------------------
            # Handle the case of direct loading of abundance names
            # -------------------------------------------------------------------------
            foundIds_ls = varargin[0]
            fpkm_ls = [float(f_i) for f_i in varargin[1]]
            if len(foundIds_ls) != len(fpkm_ls):
                error("[Error] AddAbundances(): #IDs is not equal to #abundance records")
            else:
                fpkm = dict(zip(foundIds_ls,fpkm_ls))
                print(f"Read in {len(fpkm)} abundance records.")
        else:
            error('Error: AddAbundances() - Invalid arguments. Only (pathToAbundanceFile) or (idNames[], abundVec[] ).')

        # -------------------------------------------------------------------------
        # Clear abundances and load new data
        # -------------------------------------------------------------------------

        ## In Matlab:
        # obj.abundance = zeros(1, obj.numTranscripts);
        # [commonIds, ia, ib] = intersect(foundIds, obj.ids)
        # obj.abundance(ib) = fpkm(ia)
        ## ia: the index of commonIDs in foundIDs
        ## ib: the index of commonIDs in self.ids

        self.abundance = {}
        notSetAboundace = []
        commonIDs = []
        i = 0
        for id_i in self.ids:
            i += 1
            if Transcriptome.verbose and i%5000 ==0:
                print(f"... processed {i} of {len(self.ids)} records")

            if id_i in fpkm:
                self.abundance[id_i] = fpkm[id_i]
                commonIDs.append(id_i)
            else:
                self.abundance[id_i] = "NA"
                notSetAboundace.append(id_i)

        Transcriptome.abundLoaded = True
        if len(notSetAboundace) > 0:
            print(f"[Warning]: {len(notSetAboundace)} records with missing abundance values.")

        # -------------------------------------------------------------------------
        # Return ids that were not included
        # -------------------------------------------------------------------------
        # notInTranscriptome = [id_i for id_i in fpkm if id_i not in self.ids] # notIn = setdiff(foundIds, commonIds)
        notInTranscriptome = list(set(fpkm.keys()) - set(self.ids))

        if Transcriptome.verbose:
            print('... completed in ', (tic(0) - st).total_seconds(), 's')

        return [notSetAboundace, notInTranscriptome]


    # -------------------------------------------------------------------------
    # GetAbundancesById -- Return the abundance of a set of ids
    # -------------------------------------------------------------------------
    def GetAbundanceByID(self, ids): ## return abund={id:aound,....}
        # Return a list of abundances corresponding to the ids provided
        # abund = self.GetAbundanceByID(ids)
        # ids not in the transcriptome are returned as "NA"

        # Check for single entry that is not a cell
        abund = {}

        if isinstance(ids,str):
            ids = [ids]

        if isinstance(ids,list):
            for id_i in ids:
                if id_i in self.ids:
                    abund[id_i] = float(self.abundance[id_i]) if self.abundance[id_i]!="NA" else "NA"
                else:
                    abund[id_i] = "NA"
        else:
            error('Error: GetAbundanceByID() - Invalid arguments: An ID list/ID string is required.')

        return abund

    # -------------------------------------------------------------------------
    # GetAbundanceByName -- Return the abundance of a gene
    # -------------------------------------------------------------------------
    def GetAbundanceByName(self, names, func = "all"): ## return abund={name:aound,....}
        # Return a list of abundances corresponding to provided names
        # abund = self.GetAbundanceByName(names)
        # abund = self.GetAbundanceByName(names, 'isoformFunc', functionHandle)
        # The abundance of multiple isoforms is added by default, but a
        # different function, e.g. sum/mean/min/max/all, can be passed with 'func'
        # abund = self.GetAbundanceByName(names,"sum/mean/min/max/all")
        # If 'all' is specified abundances are returned as a list of isoform abundances.

        # -------------------------------------------------------------------------
        # Find isoforms and associated keys
        # -------------------------------------------------------------------------
        if isinstance(names,str): # Handle single input
            names = [names]

        abund = {}
        if isinstance(names,list):
            for name_i in names:
                if name_i in self.name2id:
                    if len(self.name2id[name_i])>1:  ## one name has multi abundance records
                        for id_i in self.name2id[name_i]:
                            abund[name_i].append(float(self.abundance[id_i]))
                    else:
                        abund[name_i] = float(self.abundance[self.name2id[name_i][0]])
                else:
                    abund[name_i] = "NA"
        else:
            error('Error: GetAbundanceByName() - Invalid arguments: An name list/name string is required.')

        # -------------------------------------------------------------------------
        # Accumulate isoform abundance information
        # -------------------------------------------------------------------------
        if func=="sum":
            for name_i in abund:
                if abund[name_i] == "NA": continue
                if len(abund[name_i]) > 1: abund[name_i] = sum(abund[name_i])
        elif func=="mean":
            for name_i in abund:
                if abund[name_i] == "NA": continue
                if len(abund[name_i]) > 1: abund[name_i] = sum(abund[name_i])/len(abund[name_i])
        elif func=="min":
            for name_i in abund:
                if abund[name_i] == "NA": continue
                if len(abund[name_i]) > 1: abund[name_i] = min(abund[name_i])
        elif func=="max":
            for name_i in abund:
                if abund[name_i] == "NA": continue
                if len(abund[name_i]) > 1: abund[name_i] = max(abund[name_i])
        elif func == "all":
            pass
        else:
            error('[Error]: GetAbundanceByName() - Invalid arguments: func can only be set to any of sum/mean/min/max/all.')

        return abund


    # -------------------------------------------------------------------------
    # GetSequencesByName: Return sequences by gene name
    # -------------------------------------------------------------------------
    def GetSequencesByName(self, names):
        # Return all sequences for the specified name
        # seqs = self.GetSequencesByName(names)

        # Handle a single input
        if isinstance(names,str):
            names = [names]

        # Create sequences list
        seqs = {}

        # Find valid names
        if isinstance(names,list):
            for name_i in names:
                if name_i in self.name2id:
                    if len(self.name2id[name_i])>1:  ## one name has multi records
                        for id_i in self.name2id[name_i]:
                            seqs[name_i].append(self.Sequences[id_i])
                    else:
                        seqs[name_i] = self.Sequences[self.name2id[name_i][0]]
                else:
                    seqs[name_i] = "NA"
        else:
            error('[Error]: GetSequencesByName() - Invalid arguments: An name list/name string is required.')

        return seqs

    # -------------------------------------------------------------------------
    # GetSequenceByID: Return sequences by gene id
    # -------------------------------------------------------------------------
    def GetSequenceByID(self, ids):
        # Return sequence for the specified id
        # seq = self.GetSequenceByID(ids)

        # Handle a single input
        if isinstance(ids,str):
            ids = [ids]

        # Create sequences list
        seqs = {}
        if isinstance(ids,list):
            for id_i in ids:
                if id_i in self.Sequences:
                    seqs[id_i] = self.Sequences[id_i]
                else:
                    seqs[id_i] = "NA"
        else:
            error('[Error]: GetSequenceByID() - Invalid arguments: An ID list/ID string is required.')

        return seqs

    # -------------------------------------------------------------------------
    # CDSByID: Return CDS values
    # -------------------------------------------------------------------------
    def GetCDSByID(self, ids):
        # Return CDS values for each id
        # cdsValues = CDSByID(ids)

        # Handle a single input
        if isinstance(ids, str):
            ids = [ids]

        # Create sequences cell array
        cdsValues = {}
        if isinstance(ids, list):
            for id_i in ids:
                if id_i in self.cds:
                    cdsValues[id_i] = self.cds[id_i]
                else:
                    cdsValues[id_i] = "NA"
        else:
            error('[Error]: GetCDSByID() - Invalid arguments: An ID list/ID string is required.')

        return cdsValues

    # -------------------------------------------------------------------------
    # GetIDVersion: Return id version values
    # -------------------------------------------------------------------------
    def GetIDVersion(self, ids):
        # Return version identifier for each id
        # versions = GetIDVersion(obj, ids)

        # Handle a single input
        if isinstance(ids, str):
            ids = [ids]

        # Create sequences cell array
        versions = {}
        if isinstance(ids, list):
            for id_i in ids:
                if id_i in self.ids:
                    versions[id_i] = self.versions[self.ids.index(id_i)]
                else:
                    versions[id_i] = "NA"
        else:
            error('[Error]: GetIDVersion() - Invalid arguments: An ID list/ID string is required.')

        return versions


    # -------------------------------------------------------------------------
    # GetIDsByName: Return gene ids by gene name(s)
    # -------------------------------------------------------------------------
    def GetIDsByName(self, names):
        # Return all transcript ids corresponding to a gene name or names
        # ids = self.GetIDsByName(names)

        # Handle a single input
        if isinstance(names, str):
            names = [names]

        # Prepare output: empty cells indicates invalid names
        id_ls = {}
        if isinstance(names, list):
            for name_i in names:
                if name_i in self.name2id:
                    id_ls[name_i] = self.name2id[name_i]
                else:
                    id_ls[name_i] = "NA"
        else:
            error('[Error]: GetIDsByName() - Invalid arguments: An name list/name string is required.')

        return id_ls

    # -------------------------------------------------------------------------
    # GetNameById
    # -------------------------------------------------------------------------
    def GetNamesById(self, ids):
        # Return gene name for given ids
        # Names = self.GetNamesByID(name)

        # Handle a single input
        if isinstance(ids, str):
            ids = [ids]

        # Prepare output: empty cells indicates invalid names
        name_ls = {}
        if isinstance(ids, list):
            for id_i in ids:
                if id_i in self.id2name:
                    name_ls[id_i] = self.id2name[id_i]
                else:
                    name_ls[id_i] = "NA"
        else:
            error('[Error]: GetIDsByName() - Invalid arguments: An ID list/ID string is required.')

        return name_ls


    # -------------------------------------------------------------------------
    # GetNameById
    # -------------------------------------------------------------------------
    def GetNames(self):
        # Return all gene names
        # names = self.GetNames()

        names = list(self.name2id.keys())
        return names

    #-------------------------------------------------------------------------
    # Slice the transcriptome
    # -------------------------------------------------------------------------
    def Slice(self, ids):  ## return a newTranscriptome
        # Generate a new transcriptome object with a subset of entries
        # newObj = self.Slice('geneID', {id1, id2, ...})

        id_exist = []
        invalid_id = []

        sub_geneNames = []
        sub_seqs = []
        sub_abundance = []
        sub_cds = []
        sub_idVersion = []
        for id_i in ids:
            if id_i in self.ids:
                id_exist.append(id_i)
                sub_geneNames.append(self.id2name[id_i])
                sub_seqs.append(self.Sequences[id_i])
                sub_abundance.append(self.abundance[id_i])
                sub_cds.append(self.cds[id_i])
                sub_idVersion.append(self.idVersion[id_i])
            else:
                invalid_id.append(id_i)
        # print("...",self.id2name[id_i],":",len(id_exist), "valid IDs were detedted.")
        if invalid_id:
            print("[Warning]: Slice - The following ", len(invalid_id), " IDs were not found and ignored:", invalid_id)
        # -------------------------------------------------------------------------
        # Build new transcriptome object
        # -------------------------------------------------------------------------
        # obj = Transcriptome([ids, geneNames, sequences, abund, cds], **kwargs)

        newTranscriptome = Transcriptome([id_exist, sub_geneNames, sub_seqs, sub_abundance, sub_cds, sub_idVersion],
                                         verbose = Transcriptome.verbose,
                                         headerType = Transcriptome.headerType,
                                         IDType = Transcriptome.IDType,
                                         abundPath = Transcriptome.abundPath
                                         )

        return newTranscriptome

    # -------------------------------------------------------------------------
    # Save Function
    # -------------------------------------------------------------------------
    def Save(self, dirPath):
        # Save the Transcriptome object in a directory specified by dirPath
        # self.Save(dirPath)

        # -------------------------------------------------------------------------
        # Check directory validity
        # -------------------------------------------------------------------------
        # print("TEST",dirPath)

        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        # -------------------------------------------------------------------------
        # Save fields
        # -------------------------------------------------------------------------
        with open(dirPath + '/numTranscripts.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.numTranscripts, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/numGenes.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.numGenes, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/verbose.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.verbose, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/headerType.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.headerType, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/IDType.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.IDType, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/abundPath.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.abundPath, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/abundLoaded.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.abundLoaded, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/transPath.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.transPath, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/version.pkl', 'wb') as fout:
            pickle.dump(Transcriptome.version, fout, pickle.HIGHEST_PROTOCOL)

        ## Obj attributes
        with open(dirPath + '/ids.pkl', 'wb') as fout:
            pickle.dump(self.ids, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/geneNames.pkl', 'wb') as fout:
            pickle.dump(self.geneNames, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/Sequences.pkl', 'wb') as fout:
            pickle.dump(self.Sequences, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/abundance.pkl', 'wb') as fout:
            pickle.dump(self.abundance, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/cds.pkl', 'wb') as fout:
            pickle.dump(self.cds, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/id2name.pkl', 'wb') as fout:
            pickle.dump(self.id2name, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/name2id.pkl', 'wb') as fout:
            pickle.dump(self.name2id, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/idVersion.pkl', 'wb') as fout:
            pickle.dump(self.idVersion, fout, pickle.HIGHEST_PROTOCOL)
        with open(dirPath + '/transcriptomeHeader.pkl', 'wb') as fout:
            pickle.dump(self.transcriptomeHeader, fout, pickle.HIGHEST_PROTOCOL)

        ## Save done!
    # -------------------------------------------------------------------------
    # Return internal index for gene names or gene ids
    # -------------------------------------------------------------------------
    def GetInternalInds(self, attr_type, attr_list): ## [idx, ids, names]
        # Return the internal order of elements in the transcriptome as
        # specified by a list of gene names or gene ids
        # idx = self.GetInternalInds('name', names)
        # idx = self.GetInternalInds('id', ids)
        # idx = self.GetInternalInds('ind', inds)
        # idx = self.GetInternalInds('all',[])

        idx = []
        names = []
        ids = []
        # -------------------------------------------------------------------------
        # Return desired properties
        # -------------------------------------------------------------------------
        if attr_type == "all":
            idx = list(range(Transcriptome.numTranscripts))
            names = self.geneNames
            ids = self.ids
        elif attr_type == "ind":
            idx = attr_list
            names = [self.geneNames[i] for i in attr_list]
            ids = [self.ids[i] for i in attr_list]
        elif attr_type == "name":
            validKeys = [v_i for v_i in attr_list if v_i in self.name2id]
            for v_i in validKeys:
                idx = [i for i,e in enumerate(self.geneNames) if e==v_i ]
                names.append(v_i)
                ids.append(self.name2id[v_i])
        elif attr_type == "id":
            validKeys =  [v_i for v_i in attr_list if v_i in self.ids]
            for v_i in validKeys:
                idx.append(self.ids.index(v_i))
                names.append(self.id2name[v_i])
                ids.append(v_i)
        else: # If nothing is requested, return everything
            idx = list(range(Transcriptome.numTranscripts))
            names = self.geneNames
            ids = self.ids

        return [idx, ids, names]


    # -------------------------------------------------------------------------
    # Static methods
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Build a Transcriptome object from a saved version
    # -------------------------------------------------------------------------
    @staticmethod
    def Load(dirPath): ## return an Obj
        # obj = Transcriptome.Load(dirPath, 'verbose', boolean)

        # -------------------------------------------------------------------------
        # Check provided path
        # -------------------------------------------------------------------------
        if not os.path.exists(dirPath):
            error('[Error]:invalidArguments Invalid directory path for loading the Transcriptonm object.')

        # -------------------------------------------------------------------------
        # Create empty object (to define fields to load)
        # -------------------------------------------------------------------------
        obj = Transcriptome()

        # -------------------------------------------------------------------------
        # Load properties/data
        # -------------------------------------------------------------------------
        if os.path.exists(dirPath + '/numTranscripts.pkl'):
            with open(dirPath + '/numTranscripts.pkl', 'rb') as fin:
                obj.numTranscripts = pickle.load(fin)
        if os.path.exists(dirPath + '/numGenes.pkl'):
            with open(dirPath + '/numGenes.pkl', 'rb') as fin:
                obj.numGenes = pickle.load(fin)
        if os.path.exists(dirPath + '/verbose.pkl'):
            with open(dirPath + '/verbose.pkl', 'rb') as fin:
                obj.verbose = pickle.load(fin)
        if os.path.exists(dirPath + '/headerType.pkl'):
            with open(dirPath + '/headerType.pkl', 'rb') as fin:
                obj.headerType = pickle.load(fin)
        if os.path.exists(dirPath + '/IDType.pkl'):
            with open(dirPath + '/IDType.pkl', 'rb') as fin:
                obj.IDType = pickle.load(fin)
        if os.path.exists(dirPath + '/abundPath.pkl'):
            with open(dirPath + '/abundPath.pkl', 'rb') as fin:
                obj.abundPath = pickle.load(fin)
        if os.path.exists(dirPath + '/abundLoaded.pkl'):
            with open(dirPath + '/abundLoaded.pkl', 'rb') as fin:
                obj.abundLoaded = pickle.load(fin)
        if os.path.exists(dirPath + '/transPath.pkl'):
            with open(dirPath + '/transPath.pkl', 'rb') as fin:
                obj.transPath = pickle.load(fin)
        if os.path.exists(dirPath + '/version.pkl'):
            with open(dirPath + '/version.pkl', 'rb') as fin:
                obj.version = pickle.load(fin)

        ## Obj attribu
        if os.path.exists(dirPath + '/ids.pkl'):
            with open(dirPath + '/ids.pkl', 'rb') as fin:
                obj.ids = pickle.load(fin)
        if os.path.exists(dirPath + '/geneNames.pkl'):
            with open(dirPath + '/geneNames.pkl', 'rb') as fin:
                obj.geneNames = pickle.load(fin)
        if os.path.exists(dirPath + '/Sequences.pkl'):
            with open(dirPath + '/Sequences.pkl', 'rb') as fin:
                obj.Sequences = pickle.load(fin)
        if os.path.exists(dirPath + '/abundance.pkl'):
            with open(dirPath + '/abundance.pkl', 'rb') as fin:
                obj.abundance = pickle.load(fin)
        if os.path.exists(dirPath + '/cds.pkl'):
            with open(dirPath + '/cds.pkl', 'rb') as fin:
                obj.cds = pickle.load(fin)
        if os.path.exists(dirPath + '/id2name.pkl'):
            with open(dirPath + '/id2name.pkl', 'rb') as fin:
                obj.id2name = pickle.load(fin)
        if os.path.exists(dirPath + '/name2id.pkl'):
            with open(dirPath + '/name2id.pkl', 'rb') as fin:
                obj.name2id = pickle.load(fin)
        if os.path.exists(dirPath + '/idVersion.pkl'):
            with open(dirPath + '/idVersion.pkl', 'rb') as fin:
                obj.idVersion = pickle.load(fin)
        if os.path.exists(dirPath + '/transcriptomeHeader.pkl'):
            with open(dirPath + '/transcriptomeHeader.pkl', 'rb') as fin:
                obj.transcriptomeHeader = pickle.load(fin)

        return obj
