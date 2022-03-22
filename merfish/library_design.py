import os
import sys
import shutil

import datetime
import re

import numpy as np

from utils.funcs import *
from utils.fileIO import *
from merfish.probe_construction.Transcritome import Transcriptome
from merfish.probe_construction.OTTable import OTTable
from merfish.probe_construction.TRDesigner import TRDesigner
from merfish.probe_construction.TargetRegions import TargetRegions
from merfish.probe_construction.PrimerDesigner import PrimerDesigner

# def library_design():

## ------------------------------------------------------------------------
# pre-Step: set up the environment variables
#   setup the inputs
#   prepare the output paths for results
## -------------------------------------------------------------------------
rawTranscriptomeFasta = '../examples/MERFISH_LibDesign_Examples/transcripts.fasta'
fpkmPath = '../examples/MERFISH_LibDesign_Examples/isoforms.fpkm_tracking'
ncRNAPath = '../examples/MERFISH_LibDesign_Examples/Homo_sapiens.GRCh38.ncrna.fa'
readoutPath = '../examples/MERFISH_LibDesign_Examples/readouts.fasta'
codebookPath = '../examples/MERFISH_LibDesign_Examples/codebook.csv'

libSavePath = '../examples/libraryDesign'
if not os.path.exists(libSavePath):
    os.makedirs(libSavePath)

rRNAtRNAPath = os.path.join(libSavePath,'rRNAtRNA.fa')
transcriptomePath = os.path.join(libSavePath,'transcriptomeObj')
specificityTablePath = os.path.join(libSavePath, 'specificityTable')
isoSpecificityTablePath = os.path.join(libSavePath, 'isoSpecificityTables')
trDesignerPath = os.path.join(libSavePath, 'trDesigner')
trRegionsPath = os.path.join(libSavePath, 'tr_GC_43_63_Tm_66_76_Len_30_30_IsoSpec_0.75_1_Spec_0.75_1')

## Parallel setting
n_worker = 1


##############################################################################################
## ------------------------------------------------------------------------
# Step 1: Construct possible target regions
#   Below a set of transcripts, transcript abundances, and non-coding RNA
#   sequences will be used to design a set of possible target regions for all
#   transcripts in the human transcriptome
## -------------------------------------------------------------------------

## ------------------------------------------------------------------------
# Load ncRNAs and cut to rRNA and tRNA: Needed to eliminate probes that
#   have homology to these abundant RNAs
## -------------------------------------------------------------------------

PageBreak()
## Load and parse ncRNAs
if not os.path.exists(rRNAtRNAPath):
    print('Loading: ',ncRNAPath)
    print('Start at: ', tic(1))
    st = tic(0)
    ncRNAs = fastaread(ncRNAPath)
    print('Found ', len(ncRNAs), ' sequences')
    print('... completed in', toc(st), "s" )

    # Parse out 'gene_biotype'
    biotypes = {}
    for i in ncRNAs:
        tempString = re.search('gene_biotype:\S+ ', i)
        strParts = re.split(':| ',tempString[0])
        biotypes[i] = strParts[1]

    ## Identify features to keep
    PageBreak()
    biotypesToKeep = ['rRNA', 'tRNA', 'Mt_rRNA', 'Mt_tRNA']
    print('Keeping the following types: ')
    print("\t"+"\n\t".join(biotypesToKeep))

    rRNAtRNA = {i:ncRNAs[i] for i in ncRNAs if biotypes[i] in biotypesToKeep}
    print('Keeping ', len(rRNAtRNA), ' ncRNAs')

    # Save ncRNA in a fasta file
    print('Writing: ', rRNAtRNAPath)
    fastawrite(rRNAtRNAPath, rRNAtRNA)
    print('Writing done! ')
else:
    ## Load existing file if already created
    print('Found and loading: ', rRNAtRNAPath)
    print('Start at: ', tic(1))
    st = tic(0)
    rRNAtRNA = fastaread(rRNAtRNAPath)
    print('Loaded ',len(rRNAtRNA), ' sequences')
    print('... completed in', toc(st), "s" )

## ------------------------------------------------------------------------
#  Build transcriptome object: This object collects information about
#   all transcripts in the transcriptome so as to facilitate access to
#   various properties.
## -------------------------------------------------------------------------

PageBreak()
## Build transcriptome object
if not os.path.exists(transcriptomePath):
    # Build transcriptome using existing abundance data
    transcriptome = Transcriptome(rawTranscriptomeFasta, abundPath = fpkmPath,verbose = True)
    transcriptome.Save(transcriptomePath)
else:
    print('Found and loading transcriptome: ', transcriptomePath)
    st = tic(99)
    # Load transcriptome if it already exists
    transcriptome = Transcriptome.Load(transcriptomePath)
    print('Loaded ', transcriptome.numTranscripts, ' sequences')
    print('... completed in', toc(st), "s")

## ------------------------------------------------------------------------
# Build specificity tables:  These tables are used to identify (and
#   penalize probes that contain) potential regions of cross-homology
#   between transcripts
##-------------------------------------------------------------------------
# Build isoform specificity table -- these tables, one per gene, calculate
#  the penalty associated with homology regions between isoforms of the
#  the same gene
#
if not os.path.exists(isoSpecificityTablePath):
    # Get isoform data
    names = transcriptome.GetNames() ## name list
    idsByName = transcriptome.GetIDsByName(names) ## Dict {name1:ids1[],...}

    # Display progress header
    PageBreak()
    print('Starting construction of isoform specificity tables...')
    st = tic(99)

    isoSpecificityTables = []
    i = 0 ## count the number of processed genes
    for name_i in names: # Loop over all gene names -- RNAs that share the same gene name are considered isoforms
        i += 1
        # Generate a local transcriptome object that contains transcripts for a single gene
        localTranscriptome = transcriptome.Slice(idsByName[name_i])

        # Generate a OTTable for isoforms for the given gene
        # 17 is the length of exact homology used to calculate penalties
        isoSpecificityTables.append(OTTable(localTranscriptome, 17, verbose=False,transferAbund=True,name = name_i,numPar=n_worker))

        # Display progress
        if i%1000 == 0:
            print('...completed ',i,' of ',len(names), ' genes')

    # Save tables
    print("...Saving the data tables...")
    for table_i in isoSpecificityTables:
        table_i.Save(isoSpecificityTablePath)

    print('...completed in ',toc(st),' s')
else:
    # Load tables if they already exist
    PageBreak()
    print("Loading isoSpecificityTables from ", isoSpecificityTablePath)
    st = tic(99)
    isoSpecificityTables = []
    for file_i in os.listdir(isoSpecificityTablePath):
        table_file_i  = os.path.join(isoSpecificityTablePath,file_i)
        isoSpecificityTables.append(OTTable.Load(table_file_i,verbose=False, mapType = "OTMap2"))

    print('Loaded ', len(isoSpecificityTables), ' tables')
    print('... completed in',toc(st), "s")

## Build total specificity table --- this table contains a penalty associated
#   with all possible sequences in the transcriptome
if not os.path.exists(specificityTablePath):
    isoSpecificityTables[0].verbose = True
    # Add isoform specificity tables to create total transcriptome
    # specificity table
    specificityTable = OTTable.sum(isoSpecificityTables)  # Composed by summing the penalties calculated for all isoforms

    PageBreak()
    print("Saving the TranscriptomeSpecificity data tables...")
    st = tic(99)
    # Name the table
    specificityTable.name = 'TranscriptomeSpecificity'
    # Save table
    specificityTable.verbose = True
    specificityTable.Save(specificityTablePath)
    print('... completed in', toc(st), "s")

else:
    # Load table if it exists
    PageBreak()
    print("Loading TranscriptomeSpecificity from ", isoSpecificityTablePath)
    st = tic(99)
    specificityTable_file = os.path.join(specificityTablePath,"TranscriptomeSpecificity_OTTable.pkl")
    specificityTable = OTTable.Load(specificityTable_file,verbose=True,mapType="OTMap2")
    print('... completed in', toc(st), "s")


## ------------------------------------------------------------------------
# Build Penalty Tables
##-------------------------------------------------------------------------
## Build rRNA/tRNA penalty table
# Any region of exact homology equal to or greater than 15 nt will be removed
OTrRNA15 = OTTable(fastaread(rRNAtRNAPath), 15,verbose = True,numPar=n_worker)

## ------------------------------------------------------------------------
# Build TRDesigner
##-------------------------------------------------------------------------
## Slice the transcriptome to the desired expression range to lower
# computational complexity by not calculating target regions for transcripts
# that are not expressed within the desired range.

PageBreak()
print('Slicing transcriptome based on expression level: >= 1e-2 FPKM')

# Find ids with abund >= 1e-2
ids = transcriptome.ids
abund = transcriptome.GetAbundanceByID(ids)
goodIDs = [id_i for id_i in ids if ((abund[id_i]!="NA") and (abund[id_i]>=1e-2))]

# Slice transcriptome
slicedTranscriptome = transcriptome.Slice(goodIDs)

## Create Target Region Designer object
if not os.path.exists(trDesignerPath):
    trDesigner = TRDesigner(transcriptome = slicedTranscriptome,
                            OTTables = [OTrRNA15],
                            OTTableNames = ['rRNA'],
                            specificityTable = specificityTable,
                            isoSpecificityTables = isoSpecificityTables,
                            numPar=n_worker)
    trDesigner.Save(trDesignerPath)
else:
    PageBreak()
    print("Loading trDesigner object from ", trDesignerPath)
    st = tic(99)
    trDesigner_file = os.path.join(trDesignerPath,"trDesigner.pkl")
    trDesigner = TRDesigner.Load(trDesigner_file)
    print('... completed in', toc(st), "s")

## ------------------------------------------------------------------------
# Create target regions for a specific set of probe properties
##-------------------------------------------------------------------------
if not os.path.exists(trRegionsPath):
    # Design target regions
    targetRegions = trDesigner.DesignTargetRegions(regionLength=[30],
                                                   GC = [0.43,0.63],
                                                   Tm =  [66,76],
                                                   isoSpecificity = [0.75, 1],
                                                   specificity=[0.75, 1],
                                                   OTTables={'rRNA':[0, 0]}
                                                   )
    # NOTE: The ranges above were determined empirically to strike
    # the proper balance between stringency (narrow ranges) and
    # sufficient probe numbers to target the desired genes. We
    # recommend scanning many different ranges to identify the
    # optimal for each application.

    # Save target regions
    for targetRegion_i in targetRegions:
        targetRegion_i.Save(trRegionsPath)

else:
    PageBreak()
    print("Loading TargetRegions object from ", trRegionsPath)
    st = tic(99)
    targetRegions = TargetRegions.Load(trRegionsPath)
    print('... completed in', toc(st), "s")

##########################################################################################
## ------------------------------------------------------------------------
# Step 2: Compile the library
#  The target regions designed above will be compiled into template
#  molecules that can be used to build the desired probe library
##-------------------------------------------------------------------------

## ------------------------------------------------------------------------
# Load readouts, target regions, codewords, and selected genes
##-------------------------------------------------------------------------
## Load readouts and strip out the 3-letter readouts
PageBreak()
print('Loading: ', readoutPath)
st = tic(99)
readouts_dict = fastaread(readoutPath)
readouts_set =[(k,v) for k,v in readouts_dict.items()]

print('Found ',len(readouts_set), ' oligos in ',toc(st),'s')

## Load codebook (which defines the readouts to use,
# the isoforms to use, and the barcodes assigned to them)

# NOTE: A codebook should be defined before the library is constructed.
# NOTE: See the code_construction example script for instructions on how
# to generate barcodes for different encoding schemes
codebook = LoadCodebook(codebookPath)[0]

## ------------------------------------------------------------------------
# Select isoforms
##-------------------------------------------------------------------------
## Identify the isoforms to keep from those requested in the codebook
finalIds = codebook["id"] # Extract isoform ids from codebook
finalGenes = codebook["name"] # Extract gene common names from codebook
barcodes = np.array(codebook["barcode"]) # Extract string barcodes and convert to logical matrix
# Extract only the desired target regions
finalTargetRegions = [targetRegion_i for targetRegion_i in targetRegions if targetRegion_i.id in finalIds]

## ------------------------------------------------------------------------
# Construct the library
##-------------------------------------------------------------------------
## Define common properties
numProbesPerGene = 92
libraryName = 'L1E1'

PageBreak()
print('Designing oligos for ', libraryName)
print('... ', numProbesPerGene, ' probes per gene')

## Record the used readout sequences
usedReadoutPath = os.path.join(libSavePath, libraryName+'_used_readouts.fasta')
if os.path.exists(usedReadoutPath):
    print('[Warning]: Found', usedReadoutPath, '.\nThe file will be overwritten.')
    os.remove(usedReadoutPath)

fastawrite(usedReadoutPath, readouts_dict)
PageBreak()
print('Wrote ', len(readouts_dict), ' readouts to ', usedReadoutPath)

## The following IF does not in the original MATLAB version, added: Ruifeng Hu, 10162021
## Check the consistency between the readouts number and the barcodes length
## readouts number === barcodes length
if len(readouts_set) != barcodes.shape[1]:
    error("[Error]: readouts number is inconsistent to the barcodes length!")

## Build possible probes
# more than are needed are constructed to allow those with homology to rRNA/tRNA to be removed
oligosPath = os.path.join(libSavePath,libraryName+'_possible_oligos.fasta')
np.random.seed(12)
if not os.path.exists(oligosPath):
    oligos = {}
    for i in range(len(finalIds)):
        # Save local gene
        localGeneName = finalGenes[i]

        # Display progress
        PageBreak()
        print('Designing probes for ', libraryName, ': ',localGeneName)

        # Determine the bits to include for each word
        possibleReadouts = [readouts_set[j] for j in range(len(barcodes[i,:])) if barcodes[i,j]=="1"]

        # Determine targetRegion sequences
        #!!!Should be here using the geneName or using the gene ID ???? !!!
        tRegion = [tRegion_i for tRegion_i in finalTargetRegions if tRegion_i.geneName == localGeneName]

        if len(tRegion) > 0: # Check to see if there are no target regions--only used for blanks
            tRegion = tRegion[0]
            seqs = {}
            headers = {}

            # Build all possible oligos
            for pp in range(tRegion.numRegions):
                # Create random orientation and selection of readouts
                randperm_ls = np.random.choice(list(range(len(possibleReadouts))),3,replace=False)
                ## Randomly pick the N probes
                # localReadouts = [possibleReadouts[j] for j in randperm_ls]
                ## testing, pick the first n possibleReadouts
                localReadouts = possibleReadouts[:3]

                # if np.random.rand(1) > 0.5:
                if 1 > 0.5:
                    # Create header
                    headers[pp] = "".join([libraryName, ' ',
                                           localReadouts[0][0], ' ',
                                           tRegion.geneName, '_',
                                           tRegion.id, '_',
                                           str(tRegion.startPos[pp]), '_',
                                           str(len(tRegion.sequence[pp])), '_',
                                           str(round(tRegion.GC[pp],3)), '_',
                                           str(round(tRegion.Tm[pp],3)), '_',
                                           str(round(tRegion.specificity[pp],3)), ' ',
                                           localReadouts[1][0], ' ',
                                           localReadouts[2][0]
                                           ])

                    # Create sequence
                    seqs[pp] = "".join(['A ',
                                        seqrcomplement(localReadouts[0][1]), ' ',
                                        seqrcomplement(tRegion.sequence[pp]), ' A ',
                                        seqrcomplement(localReadouts[1][1]), ' ',
                                        seqrcomplement(localReadouts[2][1])
                                        ])
                else:
                    # Create header
                    headers[pp] = "".join([libraryName, ' ',
                                           localReadouts[0][0], ' ',
                                           localReadouts[1][0], ' ',
                                           tRegion.geneName, '_',
                                           tRegion.id, '_',
                                           str(tRegion.startPos[pp]), '_',
                                           str(len(tRegion.sequence[pp])), '_',
                                           str(round(tRegion.GC[pp], 3)), '_',
                                           str(round(tRegion.Tm[pp], 3)), '_',
                                           str(round(tRegion.specificity[pp], 3)), ' ',
                                           localReadouts[2][0]
                                           ])

                    # Create sequence
                    seqs[pp] = "".join(['A ',
                                        seqrcomplement(localReadouts[0][1]), ' ',
                                        seqrcomplement(localReadouts[1][1]), ' ',
                                        seqrcomplement(tRegion.sequence[pp]), ' A ',
                                        seqrcomplement(localReadouts[2][1])
                                        ])

            print('... constructed ', len(seqs), ' possible probes')

            # seqsWOSpace = cellfun(@(x) x(~isspace(x)), seqs, 'UniformOutput', false)
            seqsWOSpace = [seqs[seq_i].replace(" ", "") for seq_i in seqs]

            # Identify penalties
            # hasrRNAPenalty = cellfun(@(x) sum(OTrRNA15.CalculatePenalty(seqrcomplement(x)))>0, seqsWOSpace)
            hasrRNAPenalty =[sum(OTrRNA15.CalculatePenalty(seqrcomplement(x))[0]) > 0 for x in seqsWOSpace]

            # Select probes
            indsToKeep = np.nonzero(np.logical_not(hasrRNAPenalty))[0]
            indsToRemove = np.setdiff1d(np.array(range(len(seqs))), indsToKeep)
            print('... removing ', len(indsToRemove), ' probes')
            for r in range(len(indsToRemove)):
                print('...    ', headers[indsToRemove[r]])

            ## Randomly pick the N probes
            # indsToKeep = indsToKeep[np.random.choice(range(len(indsToKeep)), min(len(indsToKeep), numProbesPerGene),replace=False)]
            ## testing, pick the first N probes
            indsToKeep = indsToKeep[: min(len(indsToKeep), numProbesPerGene)]
            print('... keeping ',len(indsToKeep), ' probes')

            # Check on number
            if len(indsToKeep) < numProbesPerGene:
                print('[Warning]: Not enough probes for ', i, ': ', tRegion.geneName)

            # Save new oligos in oligos struct
            for s in indsToKeep:
                oligos[headers[s]] = seqs[s]
        else: #tRegion == 0
            pass

    PageBreak()
    print('Writing: ', oligosPath)
    writeTimer = tic(99)
    fastawrite(oligosPath, oligos)
    print('... completed in ',toc(writeTimer),"s")
else:
    print('[Warning]: Found existing possible oligos file!')
    oligos = fastaread(oligosPath)

## Design primers -- removing those that have homology to the probes designed above
primersPath = os.path.join(libSavePath, libraryName+'_possible_primers.fasta')
if not os.path.exists(primersPath):
    # Display progress
    PageBreak()
    print('Designing primers for ', libraryName)

    # Build Off-Target Table for existing sequences and their reverse
    # complements
    # seqRcomplement = cellfun(@(x)seqrcomplement(x(~isspace(x))), {oligos.Sequence}, 'UniformOutput', false)
    seqRcomplement = [seqrcomplement(oligos[x].replace(" ", "")) for x in oligos]
    # allSeqs = cellfun(@(x) x(~isspace(x)), {oligos.Sequence}, 'UniformOutput', false)
    allSeqs = [oligos[x].replace(" ", "") for x in oligos]
    allSeqs += seqRcomplement

    encodingProbeOTTable = OTTable(targetSequences = allSeqs, seedLength = 15, verbose = True, numPar=1)

    # Build primer designer
    prDesigner = PrimerDesigner(numPrimersToGenerate = 1e3,
                                primerLength=20,
                                OTTables = [encodingProbeOTTable],
                                OTTableNames = ['encoding'],
                                numPar = 1)

    # Cut primers
    prDesigner.CutPrimers(Tm=[70, 72],GC=[0.501, 0.651], OTTables = {'encoding':[0,0]})
    prDesigner.RemoveForbiddenSeqs()
    prDesigner.RemoveSelfCompPrimers(homologyMax=6)
    prDesigner.RemoveHomologousPrimers(homologyMax=8)

    # Write fasta file
    prDesigner.WriteFasta(primersPath)
else:
    error('[Error]: Found existing primers!')

## Add primers to the possible encoding probes designed above to generate template molecules
primers = fastaread(primersPath)
# Select the first two of the valid primers generated above
# primers is a Dict {header:seq,...}
usedPrimers = {k: primers[k] for k in list(primers.keys())[:2]}

# Add primers to encoding probes
PageBreak()
print('Adding primers')
finalPrimersPath = os.path.join(libSavePath, libraryName+'_primers.fasta')
if not os.path.exists(finalPrimersPath):
    # Record the used primers
    fastawrite(finalPrimersPath, usedPrimers)
    print('Wrote: ', finalPrimersPath)

    usedPrimers = list(usedPrimers.items())

    # Build the final oligos
    finalOligos = {}
    for header_i in oligos:
        stringParts = header_i.split(' ')
        name1 = usedPrimers[0][0].split(" ")
        name1 = name1[0]
        name2 = usedPrimers[1][0].split(' ')
        name2 = name2[0]
        temp_Header = stringParts[0]+' '+name1+' '
        for j in stringParts[1:]:
            temp_Header = temp_Header+j+' '
        temp_Header = temp_Header+name2

        finalOligos[temp_Header] = usedPrimers[0][1]+' '+oligos[header_i]+' '+seqrcomplement(usedPrimers[1][1])

else:
    error('[Error]: Found existing final primers path!')

## Select the final template molecules -- Remove any template molecules with homology to noncoding RNAs and select only the desired number to keep
PageBreak()
print('Running final cross checks and building final fasta file')
# Write final fasta
oligosPath = os.path.join(libSavePath, libraryName+'_oligos.fasta')
if not os.path.exists(oligosPath):
    # Screen against the original tables
    st = tic(99)
    print('Searching oligos for homology')
    # hasrRNAPenalty = cellfun(@(x) sum(OTrRNA15.CalculatePenalty(seqrcomplement(x(~isspace(x)))))>0, {finalOligos.Sequence})
    hasrRNAPenalty = [sum(OTrRNA15.CalculatePenalty(seqrcomplement(x.replace(" ", "")))[0]) > 0 for x in finalOligos.values()]
    print('... completed in ',toc(st), 's')

    indsToKeep = np.logical_not(hasrRNAPenalty)
    print('... found ',sum(np.logical_not(indsToKeep)),' oligos to remove')
    indsToRemove = np.nonzero(np.logical_not(indsToKeep))[0]
    for r in indsToRemove:
        print('...   ', list(finalOligos.keys())[r])

    # Remove bad oligos
    key_kept = [list(finalOligos.keys())[k] for k,v in enumerate(indsToKeep) if v]
    finalOligos = {k:finalOligos[k] for k in key_kept}

    # Write final oligos
    fastawrite(oligosPath, finalOligos)
    print('Wrote: ', oligosPath)

else:
    print('[Warning]: Found existing oligos')

## Beep
print('\nLibrary design is DONE !!!')
print('!!!======>>>>>>>>^_^<<<<<<<<======!!!')












