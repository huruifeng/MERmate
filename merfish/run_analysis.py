# -------------------------------------------------------------------------
# Purpose: analysis of MERFISH data.
# -------------------------------------------------------------------------

## Running this script
# First, make sure all of the paths are properly configured
# Second, change the paths to the example data below.
# Third, download example data from
# http://zhuang.harvard.edu/merfish/MERFISHData/MERFISH_Examples.zip.
# Place the unzipped folder in the same directory as the MERFISH_analysis software.

import pickle
import random
import pandas as pd

from utils.funcs import *
from utils.misc import StripWords
from merfish.codes.codebook_process import CodebookToMap,SECDEDCorrectableWords
from merfish.analysis.analyzeMERFISH import AnalyzeMERFISH

from merfish.reports.reports import GenerateFPKMReport
from merfish.reports.reports import GenerateBitFlipReport
from merfish.reports.reports import GenerateOnBitHistograms
from merfish.reports.reports import GenerateMoleculeStatsReport
from merfish.reports.reports import GenerateHammingSphereReport

#####################################################################################
## Setup Paths
# UPDATE these paths to point to your data (or downloaded example data).
# And to a folder where you wish to save the output of the analysis.

PageBreak()
print("Initalizing job settings ...")
# This path can be changed if these data were saved elsewhere
dataPath = '../examples/MERFISH_Analysis_Examples'
storm_data = os.path.join(dataPath,"example_data")

# This path can be changed to change the location where analysis will be saved
analysisSavePath = '../examples/analysisResults'
warning_flag = 1
createdPath = analysisSavePath
while True:
    if not os.path.exists(createdPath):
        os.makedirs(createdPath)
        print("Result saving folder:", createdPath)
        break
    else:
        if warning_flag:
            print("[Warning]: Results saving folder exists. A random 4-digit suffix were added to make a new folder name")
            warning_flag = 0
        random_number = random.randint(1000, 9999)
        createdPath = analysisSavePath + "_" + str(random_number)
analysisSavePath = createdPath

## Setup parameters
# Setup parameters for parsing the name of image files
parameters = {}
parameters["imageTag"]= 'STORM'          # Initial name--typically describes the scope
parameters["imageMListType"] = 'alist'    # Tag for molecule list file for found RNA spots
parameters["fiducialMListType"] = 'list'  # Tag for molecule list file for found fiducial spots

# Setup parameters of the run
parameters["numHybs"] = 16                # The number of hybridization/imaging rounds
parameters["bitOrder"] = list(range(16))[::-1]   # The order in which bits are imaged. The example was run in reverse

# Setup parameters for constructing words
parameters["wordConstMethod"] = 'perLocalization' # A flag to specify the word construction method. This is the preferred method.

# Setup parameters for decoding data
parameters["codebookPath"] = os.path.join(dataPath, 'codebook/codebook.fasta')         # Insert path to the codebook
parameters["exactMap"] = CodebookToMap(parameters["codebookPath"], keyType = 'binStr')
parameters["errCorrFunc"] = SECDEDCorrectableWords   # The function used to generate a map for correctable words

FPKM_df = pd.read_csv(os.path.join(dataPath, 'FPKM_data/FPKMData.csv'),engine="python",index_col=0,header=0)  # Insert path to the FPKMdata file
FPKM_dict = FPKM_df.to_dict(orient="dict")
parameters["FPKMData"] = FPKM_dict[list(FPKM_dict.keys())[0]]

# Setup FOV/cells to analyze
parameters["cellsToAnalyze"] = []         # Analyze all cells if empty

# Setup parameters for saving results
parameters["savePath"] = analysisSavePath

# Configure parameters for generating different reports
parameters["reportsToGenerate"] = {}
parameters["reportsToGenerate"]['fiducialReport1']=True
parameters["reportsToGenerate"]['fiducialReport2']=True  # {report name, 'True'/'False' do/do not display figure}
parameters["reportsToGenerate"]['numOnBitsHistByCell']=True
parameters["reportsToGenerate"]['numOnBitsHistAllCells']=True
parameters["reportsToGenerate"]['focusLockReportByCell']=True
parameters["reportsToGenerate"]['totalFPKMReport']=True
parameters["reportsToGenerate"]['cellByCellFPKMReport']=True #  Not included as a default report
parameters["reportsToGenerate"]['cellWithWordsImage']=True
parameters["reportsToGenerate"]['molStats']=True
parameters["reportsToGenerate"]['molDistStats']=True
parameters["reportsToGenerate"]['molHist']=True
parameters["reportsToGenerate"]['compositeHybImage']=True
parameters["reportsToGenerate"]['hamming1DReportAllGenes']=True
parameters["reportsToGenerate"]['bitFlipProbabilitiesAverage']=True
parameters["reportsToGenerate"]['bitFlipProbabilitiesAllGenes']=True
parameters["reportsToGenerate"]['confidenceRatioReport']=True

parameters["overwrite"] = True                # Overwrite existing files
parameters["figFormats"] = 'png'     # Output formats
parameters["useSubFolderForCellReport"] = True
parameters["saveAndClose"] = True # Save figure once created, then close it

## Run analysis!
[words, imageData, fiducialData, parameters] = AnalyzeMERFISH(dataPath=storm_data,**parameters)

PageBreak()
print("Generating reports...")
## Generate different summary reports
GenerateFPKMReport(words, **parameters,FPKMReportExactMatchOnly=False, showNames = True)

GenerateOnBitHistograms(words,**parameters,numOnBitsHistAllCells=True,numOnBitsHistByCell=False)

[moleculeStats,_] = GenerateMoleculeStatsReport(words, **parameters,)

[bitFlipReport,_] = GenerateBitFlipReport(words,**parameters)

GenerateHammingSphereReport(words, **parameters)


## Save output
PageBreak()
print("Saving data to results folder ...")
pickle.dump(imageData, open(os.path.join(parameters["savePath"],'imageData.pkl'),"wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(fiducialData, open(os.path.join(parameters["savePath"],'fiducialData.pkl'),"wb"), pickle.HIGHEST_PROTOCOL)

try:
    pickle.dump(fiducialData, open(os.path.join(parameters["savePath"], 'parameters.pkl'), "wb"),
                      pickle.HIGHEST_PROTOCOL)
except:
    print('[Warning]:Corrupt parameters file.....')

pickle.dump(bitFlipReport, open(os.path.join(parameters["savePath"],'bitFlipReport.pkl'),"wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(moleculeStats, open(os.path.join(parameters["savePath"],'moleculeStats.pkl'),"wb"), pickle.HIGHEST_PROTOCOL)

# Save words
pickle.dump(words, open(os.path.join(parameters["savePath"],'words.pkl'),"wb"), pickle.HIGHEST_PROTOCOL)

# Save Stripped Words
strippedWords = StripWords(words) # Remove fields that are not typically used for most analysis
pickle.dump(strippedWords, open(os.path.join(parameters["savePath"],'strippedWords.pkl'),"wb"), pickle.HIGHEST_PROTOCOL)

PageBreak()
print('Saved words, imageData, parameters, and reports to ',parameters["savePath"])

##
print ('Running complete!!!\nFAREWELL & GOOD LUCK!!!~@^_^@~ !!!')

