##-------------------------------------------------------------------------
## Script for running MERFISH analysis
##-------------------------------------------------------------------------
import os

from merfish.decoder.MERFISHScheduler import MERFISHScheduler
from utils.funcs import PageBreak, tic, toc

parameters = {}
## initialize data paths
parameters["experimentPath"] = 'F:/Harvard_BWH/projects/1001_MERFISH/MERmate/examples/2021_08_02_U2OS_groupA_6fovs'

## initialize merfish analysis paths. This variables are harcoded, variable names should not be changed
# raw data path
parameters["rawDataPath"] = os.path.join(parameters["experimentPath"], 'dave60x')
parameters["normalizedDataPath"] =os.path.join(parameters["experimentPath"], 'normalized_data')     # normalize data path,

parameters["settingsPath"] = os.path.join(parameters["rawDataPath"], 'settings')
parameters["abundDataPath"] = os.path.join(parameters["settingsPath"],'FPKMDataPublished.csv')   # U2OS Bulk RNAseq data, from Moffit et al, PNAS, 2016
parameters["dataOrganizationPath"] = os.path.join(parameters["settingsPath"], 'data_organization_good.csv') # csv file describing the organization of image files (filenames, number of frames, etc)
parameters["codebookPath"] =os.path.join(parameters["settingsPath"], 'L26E1_codebook.csv')                  # mapping between gene names and barcodes

## Initialize parameters for the decoder
parameters["pixelSize"]            = 107.4
parameters["imageSize"]            = (2048, 2048)
parameters["lowPassKernelSize"]    = 1
parameters["crop"]                 = 40
parameters["minBrightness"]        = 1                            # No threshold applied
parameters["minArea"]              = 1                            # Minimum area for saving barcodes
parameters["areaThresh"]           = 4                            # Area threshold for optimization
parameters["minBrightness"]        = 1                            # All decoded barcodes will be saved...no initial cuts
parameters["stageOrientation"]     = [-1, 1]                      # This value may not be correct....
parameters["overwrite"]            = True
parameters["hal_version"]          = 'hal2'
parameters["imageExt"]             = 'tif'
parameters["n_jobs"] = 1
parameters["verbose"] = False
parameters["keepInterFiles"] = True
parameters['aControl'] = "mwodscpfnrilbu"   ## All: mwodscpfnrilbu'
## Run the merfish scheduler
# define which analysis to perform:
# m = decoder;   w = warp/preprocess;    o = optimize;             d = decode;
# s = segment;   c = combine boundaries; p = parse;                f = perFormance;
# n = calc Nums; r = sum Raw data;       i = combIne raw sum data; l =low resolution mosaics
# b = barcode metadata; u = doUblet score

## Initialize file to save the output from merfish run
PageBreak()
print('Creating MERFISHDecoder for',parameters["rawDataPath"])
print('...Normalized data will be saved in', parameters["normalizedDataPath"])
scriptTimer = tic()

MERFISHScheduler(**parameters)

## Archive analysis
PageBreak()
print('...completed in', toc(scriptTimer), 's')
print('Completed at',tic(1))
##
print ('Running complete!!!\nFAREWELL & GOOD LUCK!!!~@^_^@~ !!!')


