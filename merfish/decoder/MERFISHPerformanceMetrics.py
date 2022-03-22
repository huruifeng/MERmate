import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from merfish.analysis.image_data import BuildFileStructure
from utils.fileIO import LoadCodebook, ReadBinaryFile
from utils.funcs import error, PageBreak, tic, toc
from utils.misc import bi2de, parula_map

np.seterr(invalid='ignore')

def MERFISHPerformanceMetrics(normalizedDataPath,mDecoder, **kwargs):
    # ------------------------------------------------------------------------
    # MERFISHPerformanceMetrics(normalizedDataPath, varargin)
    # This function generates a series of important quantifications of the
    # MERFISH data as well as several useful performance metrics.
    
    #--------------------------------------------------------------------------
    # Necessary Inputs:
    #   normalizedDataPath -- A valid path to a normalized MERFISH data path
    #--------------------------------------------------------------------------
    # Outputs:
    #   --None
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # June 5, 2016
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    
    
    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    
    # Parameters for displaying progress
    parameters['verbose']= False      # Display progress?
    parameters['logProgress']= True   # Create log file?
    parameters['archive']= True      # Create copies of the utilized functions

    # Parameters for location of barcodes
    parameters['barcodePath']=""         # The path to the barcodes

    # Parameters for location of saved reports
    parameters['outputPath']=""          # Path to save generated reports and metrics

    # Parameters for analyzing barcodes
    parameters['codebookPath'] =""                # Path to the codebook
    parameters['abundDataPath'] = ""               # Path to FPKM data
    parameters['cellBoundariesPath']=""          # Path to a file containing cell boundaries


    # Parameters for handling cells vs fov
    parameters['cellIDMethod']="fov"               # The method for assigning RNAs to cell: {'fov', 'cellID'}
    parameters['blockSize']=1e5                   # The number of barcodes to load at a time

    # Parameters for selecting barcodes
    parameters['brightnessThreshold'] = 10**0.75         # The minimum brightness to save a barcode
    parameters['areaThreshold'] = 2               # The minimum area to save a barcode
    parameters['stageOrientation'] = np.array([1,-1])            # The orientation of the different stage axis wrt to the camera

    # Parameters for generating reports
    parameters['visibleOption']=True               # Display figures or just generate, save, and close
    parameters['brightnessBins'] = np.append(np.arange(0,4,0.025),4.0)              # The histogram bins for log10 brightnes
    parameters['areaBins'] = np.arange(1,17)                    # The histogram bins for area

    # Parameters for finding blanks
    parameters['blankFnc'] = lambda x: ('Blank-' in x) or ('once' in x)       # Function to identify blank controls in codebook

    # Parameters for barcode density report                 # The size of 2D histogram bins in pixels
    parameters['barcodeDensityBinSize'] = 20

    # Parameters for SaveFigure
    parameters['overwrite']= True    # Options for SaveFigure
    parameters['formats']='png'
    parameters['useExportFig']= False

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if  not os.path.exists(normalizedDataPath):
        error('[Error]:invalidArguments - A valid normalized data path must be provided')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    # -------------------------------------------------------------------------
    # Define paths to various items
    # -------------------------------------------------------------------------
    # Define default path and (alternate default) for codebook
    if len(parameters["codebookPath"])==0:
        # Look for the a codebook file
        foundFiles,_ = BuildFileStructure(normalizedDataPath, regExp='(?P<codebookName>\w+)_codebook',
                                        fileExt='csv', fieldNames= ['codebookName'], fieldConv=[str])
        if len(foundFiles)==0:
            error('[Error]:missingItem-Could not find a valid codebook')
        elif len(foundFiles) > 2:
            error('[Error]:missingItem-Found two many files that look like codebooks')

        PageBreak()
        print('Utilizing a found codebook for',foundFiles[0].codebookName)
        print('...',foundFiles[0].filePath)
        parameters["codebookPath"] = foundFiles[0].filePath

    # Define path to barcodes
    if len(parameters["barcodePath"]) ==0:
        parameters["barcodePath"] = os.path.join(normalizedDataPath,'barcodes')

    # Define default output path
    if len(parameters["outputPath"])==0:
        parameters["outputPath"] = os.path.join(parameters["barcodePath"],'performance')

    # Create this path if it does not exist
    if not os.path.exists(parameters["outputPath"]):
        os.makedirs(parameters["outputPath"],exist_ok=True)

    # ------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    PageBreak()
    print('Calculating performance metrics for',normalizedDataPath)
    tic()

    # ------------------------------------------------------------------------
    # Load codebook and create decoding maps
    #-------------------------------------------------------------------------
    # Load codebook
    [codebook, codebookHeader,_] = LoadCodebook(parameters["codebookPath"], verbose=True, barcodeConvFunc=lambda x: bi2de(x))
    bits = codebookHeader["bit_names"]

    # Archive numbers
    numBarcodes = len(codebook["barcode"])
    numBits = len(bits)

    # Prepare variables for abundance correlation correlation
    if os.path.exists(parameters["abundDataPath"]):
        PageBreak()
        print('Loading abundance data for correlation from:',parameters["abundDataPath"])
        abundData = pd.read_csv(parameters["abundDataPath"],index_col=None,header=0)
        geneNames = codebook["name"]

        commonNames = [gN_i for gN_i in geneNames if gN_i in abundData["geneName"].tolist()]

        print('...found',len(commonNames),'values that overlap with codebook entries')
    else:
        PageBreak()
        print('No abundance data provided.')


    # ------------------------------------------------------------------------
    # Load and compile data either via different methods for handling different
    # 'cells'
    #-------------------------------------------------------------------------
    # Archive progress
    PageBreak()
    print('Compiling properties by cell using the <',parameters["cellIDMethod"],'> method')
    totalTimer = tic()

    # Define the path based on the method, define how these will be indexed as
    # well
    if parameters["cellIDMethod"]== 'cellID': # Handle the case that cells were parsed and barcodes contain cellIDs
        # Define path to parsed barcodes
        bListPath = os.path.join(parameters["barcodePath"],'parsed','assigned_blist.pkl')

        # Read flat header of assigned barcode list to determine number of entries
        flatHeader = pickle.load(open(bListPath,"rb"))

        # Break indices into blocks of desired size
        inds = np.arange(0,len(flatHeader),parameters["blockSize"])
        inds = inds.append(len(flatHeader))
        numObjects = len(inds)-1

        # Define cell boundaries path
        if len(parameters["cellBoundariesPath"])==0:
            parameters.cellBoundariesPath = os.path.join(normalizedDataPath,'segmentation','rawCellBoundaries.pkl')

        # Load cell boundaries
        cellBoundaries = pickle.load(open(parameters["cellBoundariesPath"],"rb"))

        # Define the number of cells
        numCells = len(cellBoundaries)

        # Define the y-labels for plot below
        FPKMCorrpltylabel = 'Counts/Cell'

    elif  parameters["cellIDMethod"]== 'fov':
        # Define path to barcodes for each fov
        bListPath = os.path.join(parameters["barcodePath"],'barcode_fov')

        # Build the file structure for these barcodes
        foundFiles,_ = BuildFileStructure(bListPath,fileExt='pkl', regExp='fov_(?P<fov>[0-9]+)',
                                        fieldNames=['fov'],fieldConv =[int])

        # Display progress
        print('Found',len(foundFiles), 'barcode files')

        # Define properties for iteration
        numObjects = len(foundFiles)
        numCells = len(foundFiles)

        # Define the y-labels for plot below
        FPKMCorrpltylabel = 'Counts/FOV'

    # Define local variables for accumulation per worker
    countsPerCellExact = np.zeros((numBarcodes, numCells))
    countsPerCellCorrected = np.zeros((numBarcodes, numCells))

    numZero2One = np.zeros((numBarcodes, numBits+1)) # 0 is also included to signify no error
    numOne2Zero = np.zeros((numBarcodes, numBits+1))

    brightnessAreaHist = np.zeros((len(parameters["brightnessBins"])-1, len(parameters["areaBins"])-1))

    distToNucleusOutNucleus = np.zeros((numBarcodes, numCells)) # Average distance per RNA to nucleus if the RNA is outside of the nucleus
    distToNucleusInNucleus = np.zeros((numBarcodes, numCells)) # Average distance per RNA to nucleus if the RNA is inside the nucleus
    fractionInNucleus = np.zeros((numBarcodes, numCells)) # Fraction of RNAs inside the nucleus

    # Histogram of barcode density per fov
    barcodeDensity = np.zeros((int(mDecoder.imageSize[0]/parameters["barcodeDensityBinSize"]),
                               int(mDecoder.imageSize[1]/parameters["barcodeDensityBinSize"]),
                               int(mDecoder.numZPos)))  # The histogram will be calculated for all z positions individually

    # Loop over a set of blocks per worker
    for b in range(numObjects):
        # Load barcodes
        loadTimer = tic(0)
        if parameters["cellIDMethod"]=='cellID':
            PageBreak()
            print('Loading block',(b+1),'of',len(inds)-1)
            aList = ReadBinaryFile(bListPath, first = inds[b]+1, last=inds[b+1])
        elif parameters["cellIDMethod"] == 'fov':
            PageBreak()
            print('Loading fov',foundFiles[b]["fov"],"-",(b+1),'of',numObjects)
            aList = pickle.load(open(foundFiles[b]["filePath"],"rb"))

        # Check for empty barcodes
        if len(aList)==0:
            continue


        print('...Loaded',len(aList),'barcodes in',toc(loadTimer),'s')

        # Compute area/brightness histogram prior to cuts
        histogramTimer = tic(0)
        brightness_hist,xedge,yedge =  np.histogram2d(x = np.log10(aList.total_magnitude/aList.area),
                                        y=aList.area,
                                        bins=[parameters["brightnessBins"], parameters["areaBins"]])
        brightnessAreaHist = brightnessAreaHist + brightness_hist

        print('...Computed brightness and area histogram in',toc(histogramTimer), 's')

        # Cut barcodes
        cutTimer = tic(0)
        aList = aList.loc[(aList.area >= parameters["areaThreshold"]) & (aList.total_magnitude/aList.area >= parameters["brightnessThreshold"]),:]

        print('...Cut to',len(aList),'barcodes in',toc(cutTimer),'s')

        # Check for empty barcodes
        if len(aList)==0:
            continue

        # Extract cellID (or fov_id)
        if parameters["cellIDMethod"] == 'cellID':
            cellID = aList.cellID
        elif parameters["cellIDMethod"] == 'fov':
            cellID = aList.fov_id.values

        # Compute histograms for counts per cell: exact and corrected
        histogramTimer = tic(0)
        data = np.stack((aList.loc[aList.is_exact==1,"barcode_id"],cellID[aList.is_exact==1]),1)
        if len(data) > 0:
            countsPerCellExact = countsPerCellExact + np.histogram2d(x=data[:,0],y=data[:,1], bins=[np.arange(numBarcodes+1), np.arange(numCells+1)])[0]

        data = np.stack((aList.loc[aList.is_exact==0,"barcode_id"], cellID[aList.is_exact==0]),1)
        if len(data) >0:
            countsPerCellCorrected = countsPerCellCorrected + np.histogram2d(x=data[:,0],y=data[:,1], bins=[np.arange(numBarcodes+1), np.arange(numCells+1)])[0]
        print('...Computed counts per cell histograms in',toc(histogramTimer),'s')

        # Accumulate errors at each bit for each barcode
        histogramTimer = tic(0)
        hist_x =aList.loc[aList.error_dir == 0, "barcode_id"]
        hist_y =aList.loc[aList.error_dir == 0, "error_bit"]
        if len(data) >0:
            numZero2One = numZero2One + np.histogram2d(x=hist_x,y=hist_y,  bins=[np.arange(numBarcodes+1), np.arange(numBits+2)])[0]

        hist_x = aList.loc[aList.error_dir == 1, "barcode_id"]
        hist_y = aList.loc[aList.error_dir == 1, "error_bit"]
        if len(data)>0:
            numOne2Zero = numOne2Zero +  np.histogram2d(x=hist_x,y=hist_y,  bins=[np.arange(numBarcodes+1), np.arange(numBits+2)])[0]

        print('...Computed errors in',toc(histogramTimer),'s')

        # Accumulate the barcode density as a function of location
        densityTimer = tic(0)

        barcodeCenter = np.stack(aList.weighted_pixel_centroid,0) # Extract the centers of all barcodes
        # Build a list of z indices for all barcodes
        if barcodeCenter.shape[1] == 3:
            zInds = np.digitize(barcodeCenter[:,2], list(range(mDecoder.numZPos))+[np.inf])
        else:
            zInds = np.ones((barcodeCenter.shape[0],))

        for z in range(mDecoder.numZPos):
            barcode_hist =  np.histogram2d(x = barcodeCenter[zInds == (z + 1), 1],
                                           y = barcodeCenter[zInds == (z + 1), 0],
                                           bins=[np.arange(0,mDecoder.imageSize[1],parameters["barcodeDensityBinSize"]),
                                                 np.arange(0,mDecoder.imageSize[0],parameters["barcodeDensityBinSize"])])
            barcodeDensity[:,:,z] = barcodeDensity[:,:,z] +barcode_hist[0]



        print('...Computed barcode density in',toc(densityTimer),'s')

    print('Completed compiling performance statistics at',tic(1),'in',toc(totalTimer), 's')

    # Normalize the distances
    # Add all counts
    totalCounts = countsPerCellExact + countsPerCellCorrected

    # Compute number out nucleus
    numOutNucleus = totalCounts - fractionInNucleus

    # Normalize distances
    distToNucleusInNucleus = distToNucleusInNucleus/fractionInNucleus # fractionInNucleus contains the number of counts in nucleus at this point
    distToNucleusOutNucleus = distToNucleusOutNucleus/numOutNucleus

    # Normalize fraction
    fractionInNucleus = fractionInNucleus/totalCounts

    # Handle division by zero
    distToNucleusInNucleus[np.isnan(distToNucleusInNucleus)] = 0
    distToNucleusOutNucleus[np.isnan(distToNucleusOutNucleus)] = 0
    fractionInNucleus[np.isnan(fractionInNucleus)] = 0

    # ------------------------------------------------------------------------
    # Save the calculated data
    #-------------------------------------------------------------------------
    if not os.path.exists(parameters["outputPath"]):
        os.makedirs(parameters["outputPath"],exist_ok=True)

    PageBreak()
    print('Writing data')
    np.savetxt(os.path.join(parameters["outputPath"],'countsPerCellExact.csv'),countsPerCellExact,delimiter=",")
    print('...wrote',os.path.join(parameters["outputPath"],'countsPerCellExact.csv'))

    np.savetxt(os.path.join(parameters["outputPath"], 'countsPerCellCorrected.csv'),countsPerCellCorrected,delimiter=",")
    print('...wrote', os.path.join(parameters["outputPath"], 'countsPerCellCorrected.csv'))

    np.savetxt(os.path.join(parameters["outputPath"], 'numZero2One.csv'),numZero2One,delimiter=",")
    print('...wrote', os.path.join(parameters["outputPath"], 'numZero2One.csv'))

    np.savetxt(os.path.join(parameters["outputPath"], 'numOne2Zero.csv'),numOne2Zero,delimiter=",")
    print('...wrote', os.path.join(parameters["outputPath"], 'numOne2Zero.csv'))

    np.savetxt(os.path.join(parameters["outputPath"], 'brightnessAreaHist.csv'),brightnessAreaHist,delimiter=",")
    print('...wrote', os.path.join(parameters["outputPath"], 'brightnessAreaHist.csv'))

    for z in range(barcodeDensity.shape[2]): # Write all of the z position densities separately
        np.savetxt(os.path.join(parameters["outputPath"], 'barcodeDensity-'+str(z)+'.csv'),barcodeDensity[:,:,z],delimiter=",")
        print('...wrote', os.path.join(parameters["outputPath"], 'barcodeDensity-'+str(z)+'.csv'))

    if parameters["cellIDMethod"]=='cellID':
        np.savetxt(os.path.join(parameters["outputPath"], 'distToNucleusInNucleus.csv'),distToNucleusInNucleus,delimiter=",")
        print('...wrote', os.path.join(parameters["outputPath"], 'distToNucleusInNucleus.csv'))

        np.savetxt(os.path.join(parameters["outputPath"], 'distToNucleusOutNucleus.csv'),distToNucleusOutNucleus,delimiter=",")
        print('...wrote', os.path.join(parameters["outputPath"], 'distToNucleusOutNucleus.csv'))

        np.savetxt(os.path.join(parameters["outputPath"], 'fractionInNucleus.csv'),fractionInNucleus,delimiter=",")
        print('...wrote', os.path.join(parameters["outputPath"], 'fractionInNucleus.csv'))

    # -------------------------------------------------------------------------
    # Create area-brightness report
    #--------------------------------------------------------------------------
    # Create figure handle
    file_name = 'Area and brightness histograms'
    fig = plt.figure(file_name,figsize=(18,6))

    # Area distribution
    fig.add_subplot(1,3,1)
    plt.bar(x = parameters["areaBins"][:-1], height=np.sum(brightnessAreaHist,0),width=1.0)
    plt.xlabel('Area (pixels)')
    plt.ylabel('Counts')
    plt.axvline(x=parameters["areaThreshold"],ls="--",c="gray",linewidth=0.5)
    plt.xlim(parameters["areaBins"][0]-1,parameters["areaBins"][-1])

    # Brightness distribution (log10)
    fig.add_subplot(1,3,2)
    plt.bar(x = parameters["brightnessBins"][:-1], height=np.sum(brightnessAreaHist,1),width=0.025)
    plt.xlabel('Brightness (log$_{10}$)')
    plt.ylabel('Counts')
    plt.axvline(x=np.log10(parameters["brightnessThreshold"]), ls="--", c="gray")
    plt.xlim(parameters["brightnessBins"][0]+np.mean(np.diff(parameters["brightnessBins"])),
             parameters["brightnessBins"][-1]-np.mean(np.diff(parameters["brightnessBins"])))

    # Area/brightness distributions
    fig.add_subplot(1,3,3)
    for i in range(len(parameters["areaBins"])-1):
        p = brightnessAreaHist[:,i]/np.max(brightnessAreaHist[:,i]) # Normalize to 1
        i_x = np.concatenate((-p,p[::-1],[-p[0]]),0)/2.5+parameters["areaBins"][i]
        i_y = np.concatenate((parameters["brightnessBins"][:-1], parameters["brightnessBins"][:-1][::-1],[parameters["brightnessBins"][0]]),0)
        plt.plot(i_x,i_y,c='blue',linewidth=0.8)
    # Add thresholds
    plt.xlabel('Area (pixels)')
    plt.ylabel('Brightness (log$_{10}$)')
    plt.xlim([parameters["areaBins"][0]-1,parameters["areaBins"][-1]])
    plt.axhline(y=np.log10(parameters["brightnessThreshold"]), ls="--", c="gray",linewidth=0.5)
    plt.axvline(x=parameters["areaThreshold"]-0.5, ls="--", c="gray",linewidth=0.5)

    plt.tight_layout()
    # Save figure
    plt.savefig(os.path.join(parameters["outputPath"],file_name+"."+parameters["formats"]))
    plt.close()


    # -------------------------------------------------------------------------
    # Create barcode density report
    #--------------------------------------------------------------------------
    # Create separate reports for all z planes
    for z in range(barcodeDensity.shape[2]):
        file_name = 'Barcode density z '+str(z)
        fig = plt.figure(file_name,figsize=(18,15))
        # Plot the density
        plt.pcolor(np.arange(0,mDecoder.imageSize[1],parameters["barcodeDensityBinSize"]),
                   np.arange(0,mDecoder.imageSize[0],parameters["barcodeDensityBinSize"]),
                   barcodeDensity[:,:,z],cmap=parula_map)
        plt.xlim([1,mDecoder.imageSize[1]])
        plt.ylim([1,mDecoder.imageSize[0]])
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.colorbar(label="Number")

        # Save figure
        plt.savefig(os.path.join(parameters["outputPath"], file_name + "." + parameters["formats"]))
        plt.close()

    # Create a combined report
    file_name = "Barcode density z-combined"
    fig = plt.figure(file_name,figsize=(18,15))

    # Plot the density
    plt.pcolor(np.arange(0, mDecoder.imageSize[1], parameters["barcodeDensityBinSize"]),
               np.arange(0, mDecoder.imageSize[0], parameters["barcodeDensityBinSize"]),
               np.sum(barcodeDensity,2),cmap=parula_map)

    plt.xlim([1, mDecoder.imageSize[1]])
    plt.ylim([1, mDecoder.imageSize[0]])
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.colorbar(label="Number")

    # Save figure
    plt.savefig(os.path.join(parameters["outputPath"], file_name + "." + parameters["formats"]))
    plt.close()

    # -------------------------------------------------------------------------
    # Create FPKM correlation plot
    #--------------------------------------------------------------------------
    if len(parameters["abundDataPath"]) > 0: # Check for FPKM data
        # Create figure handle
        file_name = "FPKM correlation"
        fig = plt.figure(file_name,figsize=(18,12))

        # Sort codebook and abundance data to match names
        [_, sortIndBarcodes, sindB] = np.intersect1d(geneNames, abundData.geneName,return_indices=True)
        sortedFPKM = abundData.FPKM[sindB].values

        # Exact FPKM Correlation
        fig.add_subplot(2,3,1)
        nA = np.mean(countsPerCellExact,1)
        nA1 = nA[sortIndBarcodes] # Sort to FPKM data
        goodInds = (nA1 > 0) & (sortedFPKM > 0)
        if np.sum(goodInds) > 2:
            plt.loglog(sortedFPKM[goodInds], nA1[goodInds], '.')
            R,P = stats.pearsonr(np.log10(sortedFPKM[goodInds]), np.log10(nA1[goodInds]))
            plt.title('Exact: $\\rho_{10}:' + f"{R:.3f} (P= {P:.2f})$" )
            plt.ylim([np.min(nA)*0.8,np.max(nA)*1.2])
            plt.xlim([np.min(sortedFPKM)*0.8,np.max(sortedFPKM)*1.2])

        plt.xlabel('FPKM')
        plt.ylabel(FPKMCorrpltylabel)

        # Enrichment/deenrichment
        fig.add_subplot(2,3,4)
        if np.sum(goodInds) > 2:
            ratio = np.log2(nA1[goodInds]/sortedFPKM[goodInds])
            sortInd = np.argsort(nA1[goodInds])[::-1]
            plt.plot(ratio[sortInd] - np.mean(ratio), '.')
            plt.title(f"STD:{np.std(ratio):.2f}")

        plt.xlabel('Barcode')
        plt.ylabel('Ratio (log$_{2}$)')

        # Corrected FPKM Correlation
        fig.add_subplot(2, 3, 2)
        nA = np.mean(countsPerCellCorrected, 1)
        nA1 = nA[sortIndBarcodes]  # Sort to FPKM data
        goodInds = (nA1 > 0) & (sortedFPKM > 0)
        if np.sum(goodInds) > 2:
            plt.loglog(sortedFPKM[goodInds], nA1[goodInds], '.')
            R, P = stats.pearsonr(np.log10(sortedFPKM[goodInds]), np.log10(nA1[goodInds]))
            plt.title('Exact: $\\rho_{10}:' + f"{R:.3f} (P= {P:.2f})$")
            plt.ylim([np.min(nA) * 0.8, np.max(nA) * 1.2])
            plt.xlim([np.min(sortedFPKM) * 0.8, np.max(sortedFPKM) * 1.2])

        plt.xlabel('FPKM')
        plt.ylabel(FPKMCorrpltylabel)

        # Enrichment/deenrichment
        fig.add_subplot(2, 3, 5)
        if np.sum(goodInds) > 2:
            ratio = np.log2(nA1[goodInds] / sortedFPKM[goodInds])
            sortInd = np.argsort(nA1[goodInds])[::-1]
            plt.plot(ratio[sortInd] - np.mean(ratio), '.')
            plt.title(f"STD:{np.std(ratio):.2f}")

        plt.xlabel('Barcode')
        plt.ylabel('Ratio (log$_{2}$)')


        # Total FPKM Correlation
        fig.add_subplot(2, 3, 3)
        nA = np.mean(countsPerCellCorrected+countsPerCellExact, 1)
        nA1 = nA[sortIndBarcodes]  # Sort to FPKM data
        goodInds = (nA1 > 0) & (sortedFPKM > 0)
        if np.sum(goodInds) > 2:
            plt.loglog(sortedFPKM[goodInds], nA1[goodInds], '.')
            R, P = stats.pearsonr(np.log10(sortedFPKM[goodInds]), np.log10(nA1[goodInds]))
            plt.title('Exact: $\\rho_{10}:' + f"{R:.1f} (P= {P:.2f})$")
            plt.ylim([np.min(nA) * 0.8, np.max(nA) * 1.2])
            plt.xlim([np.min(sortedFPKM) * 0.8, np.max(sortedFPKM) * 1.2])

        plt.xlabel('FPKM')
        plt.ylabel(FPKMCorrpltylabel)

        # Enrichment/deenrichment
        # Enrichment/deenrichment
        fig.add_subplot(2, 3, 6)
        if np.sum(goodInds) > 2:
            ratio = np.log2(nA1[goodInds] / sortedFPKM[goodInds])
            sortInd = np.argsort(nA1[goodInds])[::-1]
            plt.plot(ratio[sortInd] - np.mean(ratio), '.')
            plt.title(f"STD:{np.std(ratio):.2f}")

        plt.xlabel('Barcode')
        plt.ylabel('Ratio (log$_{2}$)')

        plt.tight_layout()
        # Save figure
        plt.savefig(os.path.join(parameters["outputPath"], file_name + "." + parameters["formats"]))
        plt.close()

    # -------------------------------------------------------------------------
    # Per-bit error report
    #--------------------------------------------------------------------------
    # Create figure handle
    file_name = "Error report"
    fig = plt.figure(file_name,figsize=(24,12))

    # Plot barcode numbers
    fig.add_subplot(2,4,1)
    plt.bar(x=[1,2],height=[np.sum(countsPerCellCorrected),np.sum(countsPerCellExact)])
    plt.xticks([1,2],['Corrected', 'Exact'],rotation= 45)
    plt.ylabel('Counts')
    plt.title(f"{np.sum(countsPerCellCorrected)/(np.sum(countsPerCellCorrected+countsPerCellExact))*100:.2f}%")

    # Plot num above blanks for different data sets
    dataSets = [countsPerCellCorrected, countsPerCellExact, countsPerCellCorrected + countsPerCellExact]
    labels = ['Corrected', 'Exact', 'All']

    # Determine blank/non-blank inds
    blankInds = np.nonzero([parameters["blankFnc"](name_i) for name_i in codebook["name"]])[0]
    nonBlankInds = np.nonzero([1-parameters["blankFnc"](name_i) for name_i in codebook["name"]])[0]

    for i in range(len(dataSets)): # Loop over the different data combinations
        fig.add_subplot(2,4,i+2)
        nA = np.sum(dataSets[i],1)

        sortednA = np.sort(nA)[::-1]
        sind = np.argsort(nA)[::-1]
        x = np.arange(len(codebook["name"]))
        x = x[sind]
        localBlankInds = np.nonzero(np.isin(x, blankInds))[0]
        plt.bar(np.arange(len(codebook["name"])), sortednA,width=1,color="blue")
        plt.bar(localBlankInds, sortednA[localBlankInds],color="red")
        plt.yscale('log')
        plt.xlabel('Barcode ID')
        plt.ylabel('Counts '+labels[i])
        plt.xlim([0,len(codebook["name"])+1])

        maxBlank = np.max(nA[blankInds])
        plt.title('Number above:'+str(np.sum(nA[nonBlankInds] > maxBlank)))


    # Plot confidence ratio
    confidenceRatio = np.sum(countsPerCellExact,1)/(np.sum(countsPerCellCorrected,1)+np.sum(countsPerCellExact,1))

    fig.add_subplot(2,4,5)
    sortedCR = np.sort(confidenceRatio)[::-1]
    sind = np.argsort(confidenceRatio)[::-1]
    x = np.arange(len(codebook["name"]))
    x = x[sind]
    localBlankInds = np.nonzero(np.isin(x, blankInds))[0]
    plt.bar(np.arange(len(codebook["name"])), sortedCR,width=1,color="blue")
    plt.bar(localBlankInds, sortedCR[localBlankInds],color="red")
    plt.yscale('log')
    plt.xlabel('Barcode ID')
    plt.ylabel('Confidence ratio')
    plt.xlim([0,len(codebook["name"])+1])

    maxBlank = np.max(confidenceRatio[blankInds])
    plt.title('Number above:'+str(np.sum(confidenceRatio[nonBlankInds] > maxBlank)))

    # Plot per bit error rates
    normalizedOne2Zero = numOne2Zero[:,1:]/np.tile(np.sum(countsPerCellExact+countsPerCellCorrected,1).reshape(-1,1),(1,numBits))
    normalizedZero2One = numZero2One[:,1:]/np.tile(np.sum(countsPerCellExact+countsPerCellCorrected,1).reshape(-1,1),(1,numBits))

    # 1->0
    fig.add_subplot(2,4,6)
    One2Zero_mean = np.nanmean(normalizedOne2Zero,0)
    One2Zero_mean[~np.isfinite(One2Zero_mean)] = np.nan
    plt.bar(np.arange(len(One2Zero_mean)),height=One2Zero_mean, color='blue')
    plt.ylabel('Error rate')
    plt.xlabel('Bit')
    plt.title(f'1 $\\rightarrow$ 0: {np.nanmean(One2Zero_mean):.2f}')
    plt.xlim([0,len(bits)+1])

    # 0->1
    fig.add_subplot(2,4,7)
    Zero2One_mean = np.nanmean(normalizedZero2One, 0)
    Zero2One_mean[~np.isfinite(Zero2One_mean)] = np.nan
    plt.bar(np.arange(len(Zero2One_mean)), height=Zero2One_mean, color='blue')
    plt.ylabel('Error rate')
    plt.xlabel('Bit')
    plt.title(f'0 $\\rightarrow$ 1: {np.nanmean(Zero2One_mean):.2f}')
    plt.xlim([0,len(bits)+1])

    plt.tight_layout()
    # Save figure
    plt.savefig(os.path.join(parameters["outputPath"], file_name + "." + parameters["formats"]))
    plt.close()

    # ------------------------------------------------------------------------
    # Archival
    # -------------------------------------------------------------------------
    # Record completion
    PageBreak()
    print('Completed decoding of',normalizedDataPath,'\n...at',tic(1))



