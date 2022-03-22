# A master script for coordinating the many aspects of MERFISH analysis
# Jeffrey Moffitt
# lmoffitt@mcb.harvard.edu
# September 21, 2017
#--------------------------------------------------------------------------
# Copyright Presidents and Fellows of Harvard College, 2018.
# -------------------------------------------------------------------------
# Purpose: 1) To act as a metascheduler, interfacing matlab with SLURM
# 2) To automate the many steps associated with analyzing a MERFISH dataset
# 3) To better handle the rare, unexplained, cluster failure
# -------------------------------------------------------------------------

import os
import shutil

import numpy as np

from merfish.analysis.image_data import BuildFileStructure
from merfish.decoder.CalculateDoubletScore import CalculateDoubletScore
from merfish.decoder.MERFISHDecoder import MERFISHDecoder
from merfish.decoder.MERFISHPerformanceMetrics import MERFISHPerformanceMetrics
from utils.funcs import error, PageBreak, tic, toc
from multiprocessing.dummy import Pool

import logging
## https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings/11233293
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# def setup_logger(name, log_file, level=logging.INFO):
#     """To setup as many loggers as you want"""
#
#     handler = logging.FileHandler(log_file)
#     handler.setFormatter(formatter)
#
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
#     return logger
#
# # first file logger
# logger = setup_logger('first_logger', 'first_logfile.log')
# logger.info('This is just info message')
#
# # second file logger
# super_logger = setup_logger('second_logger', 'second_logfile.log')
# super_logger.error('This is an error message')
#
# def another_method():
#    # using logger defined above also works here
#    logger.info('Inside method')

##################################################
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')
# logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

def do_func(*args):
    # Create separate key/value matrix for each worker
    funcs = args[0]
    funcs(args[1])
    return 1

def MERFISHScheduler(**kwargs):
    ## ------------------------------------------------------------------------
    # Prepare workspace
    # -------------------------------------------------------------------------
    ## Check for information on what analysis to perform
    if 'aControl' not in kwargs:
        # m = decoder; w = warp/preprocess; o = optimize; d = decode;
        # s = segment; c = combine boundaries; p = parse;
        # f= perFormance; n = calculate Numbers;
        # r=sum Raw data; i = combIne raw sum data
        # l=low resolution mosaics
        # b=barcode metadata
        # u=doUblet score
        aControl = 'mwodscpfnrilbu'
    else:
        aControl = kwargs['aControl']

    ## Check for required paths
    if 'normalizedDataPath' not in kwargs:
        error('A normalized data path must be provided')
    else:
        normalizedDataPath = kwargs["normalizedDataPath"]

    if 'm' not in aControl:
        rawDataPath = '[Skipped]'
    elif 'm' in aControl:
        if 'rawDataPath' not in kwargs:
            error('A raw data path must be provided to create a MERFISHDecoder instance')
        else:
            rawDataPath = kwargs["rawDataPath"]

    if 'f' in aControl and 'abundDataPath' not in kwargs:
        error('A path to abundance data must be provided to calculate MERFISH performance metrics')

    if 'abundDataPath' in kwargs:
        abundDataPath = kwargs["abundDataPath"]

    ## Create normalized data path if it does not exist
    if not os.path.exists(normalizedDataPath):
        os.mkdir(normalizedDataPath)

    parameters = kwargs

    ## Start
    PageBreak()
    print('Running MERFISH Scheduler: ')
    print('Raw data path:',rawDataPath)
    print('Analyzed data path:', normalizedDataPath)
    print('Requested analysis:',aControl)
    print('Provided parameters:')
    for kw_i in parameters:
        print(f"\t{kw_i:20s} --> {parameters[kw_i]}")

    ## ------------------------------------------------------------------------
    # Build decoder
    ## ------------------------------------------------------------------------
    if 'm' in aControl:
        # Confirm that the decoder does not yet exist
        if os.path.exists(os.path.join(normalizedDataPath,'mDecoder')):
            try:
                shutil.rmtree(os.path.join(normalizedDataPath, 'mDecoder'))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            print('[Warning]: A decoder already exists.  Removed.')
            # error('A decoder already exists. Please remove it.')

        # Mark the start of decoder construction
        PageBreak()
        print('Creating MERFISHDecoder')
        localTimer = tic()

        # Build the decoder
        new_mDecoder = MERFISHDecoder(**parameters)

        # Save the decoder
        new_mDecoder.Save() # Assume the default location

        # Mark completion
        print('...completed MERFISHDecoder construction in ',toc(localTimer), 's')
    else:
        print('Create MERFISHDecoder...[Skipped]')
    ## Load Decoder: to determine the fov ids
    mDecoder = MERFISHDecoder.Load(normalizedDataPath,"mDecoder")

    if "n_jobs" in parameters:
        n_jobs = parameters["n_jobs"]
        n_jobs = 1
    else:
        n_jobs = 1
    ## ------------------------------------------------------------------------
    # Create jobs and job arrays
    # -------------------------------------------------------------------------
    PageBreak()
    print('Running environment checks are Done!')
    print('Data decoding starts...')

    ## 1. Warp and preprocess data
    if 'w' in aControl:
        # Create job array
        PageBreak()
        print('[1/13] Warp and preprocess data...')
        # -------------------------------------------------------------------------
        # Make directories if they do not exist
        # -------------------------------------------------------------------------
        # Directory for molecule lists
        if not os.path.exists(os.path.join(mDecoder.normalizedDataPath,mDecoder.fiducialDataPath)):
            os.makedirs(os.path.join(mDecoder.normalizedDataPath,mDecoder.fiducialDataPath),exist_ok=True)
            print("...Create fiducialData path:", os.path.join(mDecoder.normalizedDataPath,mDecoder.fiducialDataPath))
        # Directory for warped data
        if not os.path.exists(os.path.join(mDecoder.normalizedDataPath,mDecoder.warpedDataPath)):
            os.makedirs(os.path.join(mDecoder.normalizedDataPath,mDecoder.warpedDataPath),exist_ok=True)
            print("...Create warpedData path:", os.path.join(mDecoder.normalizedDataPath, mDecoder.warpedDataPath))
        # Directory for molecule lists
        if not os.path.exists(os.path.join(mDecoder.normalizedDataPath, mDecoder.processedDataPath)):
            os.makedirs(os.path.join(mDecoder.normalizedDataPath, mDecoder.processedDataPath), exist_ok=True)
            print("...Create processedData path:", os.path.join(mDecoder.normalizedDataPath, mDecoder.warpedDataPath))

        if n_jobs==1:
            for f in mDecoder.fovIDs:
                mDecoder.WarpFOV([f])
                mDecoder.PreprocessFOV([f])
        else:
            targetArgs_splited = np.array_split(mDecoder.fovIDs, n_jobs)
            ## ***********************************************
            ## multiple processing [1，2，...,i,...,numPar]
            pool_args = [(mDecoder.WarpFOV,list(targetArgs_splited[i])) for i in range(n_jobs)]
            pool = Pool(processes=n_jobs)
            pool.starmap(do_func, pool_args)
            pool.close()
            pool.join()

            pool_args = [(mDecoder.PreprocessFOV, list(targetArgs_splited[i])) for i in range(n_jobs)]
            pool = Pool(processes=n_jobs)
            pool.starmap(do_func, pool_args)
            pool.close()
            pool.join()
    else:
        PageBreak()
        print('[1/13] Warp and preprocess data...[Skipped]')

    ## 2. Optimize thresholds
    if 'o' in aControl or "m" in aControl:  ## if created a new mDecoder, Optimize process is required for getting the scaleFactor!
        # Create job for the optimization process
        PageBreak()
        print('[2/13] Optimize data...')
        if not os.path.exists(os.path.join(mDecoder.normalizedDataPath, mDecoder.reportPath)):
            os.makedirs(os.path.join(mDecoder.normalizedDataPath, mDecoder.reportPath))

        ## Aggregate warp data and generate warp report
        mDecoder.GenerateWarpReport()
        ## Combine pixel histograms and prepare initial scale factors
        mDecoder.InitializeScaleFactors()
        mDecoder.OptimizeScaleFactors(25, overwrite = mDecoder.overwrite, useBlanks=False) # Normally 50 CHANGE ME BACK
        ## Archive analysis
        mDecoder.Save()
    else:
        PageBreak()
        print('[2/13] Optimize data...[Skipped]')

    ## 3. Create decoding jobs
    if 'd' in aControl:
        PageBreak()
        print('[3/13] Decoding data...')
        ## Load MERFISH Decoder
        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")
        mDecoder.overwrite = True  # Allow graceful restart of processes

        # -------------------------------------------------------------------------
        # Make directories if they do not exist
        # -------------------------------------------------------------------------
        # Base directory for barcodes
        if not os.path.exists(os.path.join(mDecoder.normalizedDataPath,mDecoder.barcodePath)):
            os.makedirs(os.path.join(mDecoder.normalizedDataPath,mDecoder.barcodePath),exist_ok=True)

        # Directory for barcodes by fov
        barcodeByFovPath = os.path.join(mDecoder.normalizedDataPath,mDecoder.barcodePath, 'barcode_fov')
        if not os.path.exists(barcodeByFovPath):
            os.makedirs(barcodeByFovPath, exist_ok=True)

        # -------------------------------------------------------------------------
        # Generate decoding matrices
        # -------------------------------------------------------------------------
        [exactBarcodes, singleBitErrorBarcodes] = mDecoder.GenerateDecodingMatrices()

        for f in mDecoder.fovIDs:
            ## Decode FOV
            mDecoder.DecodeFOV([f],exactBarcodes,singleBitErrorBarcodes)
            print('Completed decoding of fov',f,'at',tic(1))
    else:
        PageBreak()
        print('[3/13] Decoding data...[Skipped]')

    ## 4. Create segmentation jobs
    if 's' in aControl:
        PageBreak()
        print('[4/13] Image segmentation...')
        ## Load MERFISH Decoder
        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")
        mDecoder.overwrite = False  # Allow graceful restart of processes

        # -------------------------------------------------------------------------
        # Create segmentation directory if needed
        # -------------------------------------------------------------------------
        if not os.path.exists(os.path.join(mDecoder.normalizedDataPath,mDecoder.segmentationPath)):
            os.makedirs(os.path.join(mDecoder.normalizedDataPath,mDecoder.segmentationPath),exist_ok=True)

        localSavePath = os.path.join(mDecoder.normalizedDataPath, mDecoder.segmentationPath, 'fov_images')
        if not os.path.exists(localSavePath):
            os.makedirs(localSavePath, exist_ok=True)
            print('Created', localSavePath)

        for f in mDecoder.fovIDs:
            mDecoder.SegmentFOV([f])
            print('Completed segmentation of fov', f, 'at', tic(1))
    else:
        PageBreak()
        print('[4/13] Image segmentation...[Skipped]')

    ## 5. Create combine features job
    if 'c' in  aControl:
        # Create job for the combination process
        PageBreak()
        print('[5/13] Combine features...')
        ## Load MERFISH Decoder
        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")
        mDecoder.overwrite = False  # Allow graceful restart of processes

        scriptTimer = tic()

        mDecoder.CombineFeatures()

        ## Generate report
        mDecoder.GenerateFoundFeaturesReport()

        ## Export the found features to a csv file
        mDecoder.FoundFeaturesToCSV(downSampleFactor =10, zIndex=3)

        ## Archive analysis
        PageBreak()
        print('...completed in',toc(scriptTimer),'s')
        print('Completed at',tic(1))

    else:
        PageBreak()
        print('[5/13] Combine features...[Skipped]')


    ## 6. Create low resolution mosaic job
    if 'l' in  aControl:
        # Create low resolution mosaic job
        PageBreak()
        print('[6/13] Generate low resolution mosaic...')
        ## Load MERFISH Decoder
        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")
        mDecoder.overwrite = False  # Allow graceful restart of processes

        scriptTimer = tic()

        ##Create the  summation report
        mDecoder.GenerateLowResolutionMosaic()

        PageBreak()
        print('...completed in', toc(scriptTimer), 's')
        print('Completed at', tic(1))

    else:
        PageBreak()
        print('[6/13] Generate low resolution mosaic...[Skipped]')

    ## 7. Create parse jobs
    if 'p' in aControl:
        # Create job array
        PageBreak()
        print('[7/13] Parse  FOV...')
        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")
        mDecoder.overwrite = False  # Allow graceful restart of processes

        # -------------------------------------------------------------------------
        # Make directories if they do not exist
        # -------------------------------------------------------------------------
        barcodeByFovPath = os.path.join(mDecoder.normalizedDataPath, mDecoder.barcodePath, 'barcode_fov')
        # Check for existing barcodes
        if not os.path.exists(barcodeByFovPath):
            error('[Error]:missingData - No barcodes could be found.')

        # Make directory for the parsed barcodes
        parsedBarcodePath = os.path.join(mDecoder.normalizedDataPath, mDecoder.barcodePath, 'parsed_fov')
        # Directory for barcodes by fov
        if not os.path.exists(parsedBarcodePath):
            os.makedirs(parsedBarcodePath, exist_ok=True)

        for f in mDecoder.fovIDs:
            # Run job for each fov
            mDecoder.ParseFOV([f])
            print('Completed decoding of fov', f, 'at', tic(1))

    else:
        PageBreak()
        print('[7/13] Parse FOV... [Skipped]')

    ## 8. Calculate performance
    if 'f' in aControl:
        PageBreak()
        print('[8/13] Calculate performance...')
        scriptTimer = tic()
        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")
        mDecoder.overwrite = False  # Allow graceful restart of processes

        # Confirm validity of all barcode files
        bFiles,_ = BuildFileStructure(os.path.join(normalizedDataPath,'barcodes','barcode_fov'),
                                    regExp='fov_(?P<fov>[0-9]+)_blist',fileExt='pkl',
                                    fieldNames=['fov'], fieldConv=[int])

        print('Found',len(bFiles),'barcode files')

        # Determine if some barcode files are missing
        missingFovIds = np.setdiff1d(mDecoder.fovIDs, [b["fov"] for b in bFiles])
        if len(missingFovIds) > 0:
            print('[Warning]: Discovered missing fov ids!')
            print(missingFovIds)

        #Create parameters object
        performanceParameters = {}
        performanceParameters["brightnessThreshold"] = mDecoder.parameters["quantification"]["minimumBarcodeBrightness"]
        performanceParameters["areaThreshold"] = mDecoder.parameters["quantification"]["minimumBarcodeArea"]
        performanceParameters["stageOrientation"] = mDecoder.parameters["decoding"]["stageOrientation"]
        performanceParameters["abundDataPath"] = abundDataPath
        performanceParameters["codebookPath"] = mDecoder.codebookPath
        performanceParameters["verbose"] = True
        performanceParameters["outputPath"] = os.path.join(mDecoder.normalizedDataPath,mDecoder.reportPath,'performance')

        # Archive parameters in log
        PageBreak()
        print('Using the following parameters')
        for f in performanceParameters:
            print('   ',f,':',performanceParameters[f])

        # Calculate peformance
        MERFISHPerformanceMetrics(normalizedDataPath,mDecoder, **performanceParameters)

        # Archive analysis
        PageBreak()
        print('...completed in',toc(scriptTimer),'s')
        print('Completed at',tic(1))
    else:
        PageBreak()
        print('[8/13] Calculate performance...[Skipped]')

    ## 9. Calculate numbers
    if 'n' in aControl:
        PageBreak()
        print('[9/13] Calculate numbers...')
        scriptTimer = tic()

        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")

        ## Calculate feature counts
        mDecoder.CalculateFeatureCounts()

        ## Generate feature counts report
        mDecoder.GenerateFeatureCountsReport()

        # Archive analysis
        PageBreak()
        print('...completed in', toc(scriptTimer), 's')
        print('Completed at', tic(1))
    else:
        PageBreak()
        print('[9/13] Calculate numbers...[Skipped]')

    ## 10. Export barcode metadata
    if 'b' in aControl:
        PageBreak()
        print('[10/13] Export barcode metadata...')
        scriptTimer = tic()

        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")

        ## Calculate feature counts
        mDecoder.BarcodesToCSV()

        ## Calculate the doublet score metadata features
        CalculateDoubletScore(mDecoder)

        # Archive analysis
        PageBreak()
        print('...completed in', toc(scriptTimer), 's')
        print('Completed at', tic(1))
    else:
        PageBreak()
        print('[10/13] Export barcode metadata...[Skipped]')

    ## 11. Calculate doublet score values
    if 'u' in aControl:
        PageBreak()
        print('[11/13] Calculate the doublet score...')
        scriptTimer = tic()

        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")

        ## Calculate the doublet score
        CalculateDoubletScore(mDecoder)

        # Archive analysis
        PageBreak()
        print('...completed in', toc(scriptTimer), 's')
        print('Completed at', tic(1))
    else:
        PageBreak()
        print('[11/13] Calculate the doublet score...[Skipped]')

    ## 12. Sum raw data
    if 'r' in aControl:
        PageBreak()
        print('[12/13] Sum raw data...')
        scriptTimer = tic()

        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")

        for f in mDecoder.fovIDs:
            # Run job for each fov
            mDecoder.SumRawSignalFOV([f])
            print('Completed summation of raw data in fov', f, 'at', tic(1))

        # Archive analysis
        PageBreak()
        print('...completed in', toc(scriptTimer), 's')
        print('Completed at', tic(1))
    else:
        PageBreak()
        print('[12/13] Sum raw data...[Skipped]')

    ## 13. Combine raw sum
    if 'i' in aControl:
        PageBreak()
        print('[13/13]Combine raw sum...')
        scriptTimer = tic()

        mDecoder = MERFISHDecoder.Load(normalizedDataPath, "mDecoder")

        # Control for possible overwrite
        mDecoder.overwrite = False

        # Run the raw sum combination
        mDecoder.CombineRawSum()

        # Create the summation report
        mDecoder.GenerateSummationReport()

        # Archive analysis
        PageBreak()
        print('...completed in', toc(scriptTimer), 's')
        print('Completed at', tic(1))
    else:
        PageBreak()
        print('[13/13] Combine raw sum...[Skipped]')

