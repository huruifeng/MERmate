###############################
## Start: 01152022
## End:
## Ruifeng Hu
## Lexington, MA
################################
import os
import pickle
import shutil
import sys
from os.path import basename
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pyclustering.cluster import kmedoids

from bs4 import BeautifulSoup as BS
from matplotlib import cm
from scipy import ndimage
from skimage import restoration, measure, morphology
from skimage.color import label2rgb
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage.morphology import disk, ball, binary_erosion, erosion, dilation, binary_dilation, diamond
from skimage.segmentation import watershed
from skimage.transform import AffineTransform, warp, resize
from tifffile import TiffWriter, imread, TIFF

from merfish.analysis.image_data import BuildFileStructure, ReadInfoFile, ReadDax, ReadMasterMoleculeList
from merfish.decoder.FoundFeature import FoundFeature
from merfish.reports.generate_report import GenerateGeoTransformReport
from merfish.scripts.mList import MLists2Transform
from merfish.scripts.daoSTORM import WriteDaoSTORMParameters, daoSTORM
from merfish.scripts.deconvolve import deconvlucy_x

from utils.funcs import error, PageBreak, tic, toc, tiffimginfo, knnsearch2d, bitFlip, knnsearch_ckdtree
from utils.fileIO import LoadCodebook, LoadDataOrganization, ReadBinaryFileHeader, WriteBinaryFile
from utils.misc import bi2de, fgauss2D, conndef, de2bi, sub2ind, parula_map, ind2sub, ind2sub3d
from utils.misc import imadjust, imimposemin

np.seterr(all='ignore')

class MERFISHDecoder:
    # ------------------------------------------------------------------------
    # [mdObject] = MERFISHDecoder(varargin)
    # This class provides a wrapper for a MERFISH data set, the basic functions
    # for data normalization, data analysis, and data visualization.
    #--------------------------------------------------------------------------
    # Original version: Writing in MATLAB
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # jeffrey.moffitt@childrens.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    #--------------------------------------------------------------------------
    # This class is a wrapper around a MERFISH data set. It coordinates all
    # aspects of analysis of MERFISH data, including decoding, segmentation, and
    # parsing. It also provides basic functionality to access different aspects
    # of a MERFISH data set.

    ## class properties with default values
    version = '0.1'        # Version number--used to identify how to load saved classes

    # -------------------------------------------------------------------------
    # Define constructor
    # -------------------------------------------------------------------------
    def __init__(self,**kwargs):
        # This method is a wrapper for a MERFISH dataset.
        # It provides an internal mapping of the raw data, normalized data,
        # and methods for normalizing, decoding, and visualizing data
        #
        # Examples
        # mDecoder = MERFISHDecoder(rawDataPath) # Create a MERFISH decoder object
        # mDecoder = MERFISHDecoder() # Create an empty decoder containing all default values for parameters
        # mDecoder = MERFISHDecoder(rawDataPath, 'variableName', value) #
        #    Pass a named variable to the decoder to overwrite a default.
        #    See property summary by typing doc MERFISHDecoder

        self.name = kwargs["name"] if ("name" in kwargs) and kwargs["name"] else ""
        self.hal_version =  kwargs["hal_version"] if ("hal_version" in kwargs) and kwargs["hal_version"] else "hal1"   # The microscope control software version,{'hal1', 'hal2'}
        self.verbose =  kwargs["verbose"] if ("verbose" in kwargs) and kwargs["verbose"] else True
        self.overwrite =  kwargs["overwrite"] if ("overwrite" in kwargs) and kwargs["overwrite"] else False
        self.keepInterFiles = kwargs["keepInterFiles"] if ("keepInterFiles" in kwargs) and kwargs["keepInterFiles"] else False

        # Metadata associated with raw data
        self.rawDataPath = kwargs["rawDataPath"] if ("rawDataPath" in kwargs) and kwargs["rawDataPath"] else ""            # The path to the raw data
        self.dataOrganization = kwargs["dataOrganization"] if ("dataOrganization" in kwargs) and kwargs["dataOrganization"] else ""        # The data organization file
        self.dataOrganizationPath = kwargs["dataOrganizationPath"] if ("dataOrganizationPath" in kwargs) and kwargs["dataOrganizationPath"] else ""    # Path to the data organization file

        # Properties for parallel processing
        self.numPar = kwargs["numPar"] if ("numPar" in kwargs) and kwargs["numPar"] else 0                  # The number of parallel workers

        # Metadata associated with codebook
        self.codebookPath = kwargs["codebookPath"] if ("codebookPath" in kwargs) and kwargs["codebookPath"] else ""   # Path to the codebook
        self.codebook = kwargs["codebook"] if ("codebook" in kwargs) and kwargs["codebook"] else ""                # The codebook
        self.codebookHeader = kwargs["codebookHeader"] if ("codebookHeader" in kwargs) and kwargs["codebookHeader"] else ""          # Metadata associated with the codebook

        # Metadata associated with normalized data
        self.normalizedDataPath =  kwargs["normalizedDataPath"] if ("normalizedDataPath" in kwargs) and kwargs["normalizedDataPath"] else ''   # Base path to the data set
        self.mDecoderPath = kwargs["mDecoderPath"] if ("mDecoderPath" in kwargs) and kwargs["mDecoderPath"] else 'mDecoder'                # Relative path to the MERFISHDecoder instance
        self.fiducialDataPath = kwargs["fiducialDataPath"] if ("fiducialDataPath" in kwargs) and kwargs["fiducialDataPath"] else 'fiducial_data'       # Relative path to information on the fiducial warping process
        self.warpedDataPath = kwargs["warpedDataPath"] if ("warpedDataPath" in kwargs) and kwargs["warpedDataPath"] else 'warped_data'           # Relative path to warped tiff stacks
        self.processedDataPath = kwargs["processedDataPath"] if ("processedDataPath" in kwargs) and kwargs["processedDataPath"] else 'processed_data'     # Relative path to pre-processed tiff stacks
        self.barcodePath = kwargs["barcodePath"] if ("barcodePath" in kwargs) and kwargs["barcodePath"] else 'barcodes'  # Relative path to barcode data
        self.reportPath = kwargs["reportPath"] if ("reportPath" in kwargs) and kwargs["reportPath"] else 'reports'      # Relative path to various reports/perfomance metrics
        self.segmentationPath = kwargs["segmentationPath"] if ("segmentationPath" in kwargs) and kwargs["name"] else 'segmentation'        # Relative path to segmentation results
        self.summationPath = kwargs["summationPath"] if ("summationPath" in kwargs) and kwargs["summationPath"] else 'summation'              # Relative path to summation results
        self.mosaicPath = kwargs["mosaicPath"] if ("mosaicPath" in kwargs) and kwargs["mosaicPath"] else 'mosaics'      # Relative path to low resolution mosaic images

        # Metadata associated with the MERFISH run
        self.numDataChannels = kwargs["numDataChannels"] if ("numDataChannels" in kwargs) and kwargs["numDataChannels"] else 0         # Number of data channels, i.e. specific image channels

        self.numBits = kwargs["numBits"] if ("numBits" in kwargs) and kwargs["numBits"] else ""                 # The number of bits
        self.bitNames = kwargs["bitNames"] if ("bitNames" in kwargs) and kwargs["bitNames"] else ""               # The names of the individual readout probes for each bit
        self.numBarcodes = kwargs["numBarcodes"] if ("numBarcodes" in kwargs) and kwargs["numBarcodes"] else 0    # The number of barcodes

        self.numFov = kwargs["numFov"] if ("numFov" in kwargs) and kwargs["numFov"] else 0                  # The number of FOV
        self.fovIDs = kwargs["fovIDs"] if ("fovIDs" in kwargs) and kwargs["fovIDs"] else []                 # The fov ids
        self.fovPos = kwargs["fovPos"] if ("fovPos" in kwargs) and kwargs["fovPos"] else []                 # The position of each fov in microns (Nx2)

        self.numZPos = kwargs["numZPos"] if ("numZPos" in kwargs) and kwargs["numZPos"] else 0                # The number of z stacks for each data frame
        self.zPos = kwargs["zPos"] if ("zPos" in kwargs) and kwargs["zPos"] else []                    # The position of each z-stack in microns

        self.imageSize = kwargs["imageSize"] if ("imageSize" in kwargs) and kwargs["imageSize"] else 0    # The number of pixels (HxW)
        self.imageExt  = kwargs["imageExt"] if ("imageExt" in kwargs) and kwargs["imageExt"] else 'dax'     # Type of the raw data, {'dax', 'tif', 'tiff'}
        self.pixelSize = kwargs["pixelSize"] if ("pixelSize" in kwargs) and kwargs["pixelSize"] else 109     # The size of the pixel

        self.cameraIDs =kwargs["cameraIDs"] if ("cameraIDs" in kwargs) and kwargs["cameraIDs"] else []        # The ids associated with cameras for imaging
        self.numCameraIDs = kwargs["numCameraIDs"] if ("numCameraIDs" in kwargs) and kwargs["numCameraIDs"] else 1   # The number of cameras used to collect data

        self.numImagingRounds = kwargs["numImagingRounds"] if ("numImagingRounds" in kwargs) and kwargs["numImagingRounds"] else 0       # The number of imaging rounds
        self.imageRoundIDs = kwargs["imageRoundIDs"] if ("imageRoundIDs" in kwargs) and kwargs["imageRoundIDs"] else 0     # The ids of the different imaging rounds

        # Parameters associated with decoding
        self.scaleFactors = kwargs["scaleFactors"] if ("scaleFactors" in kwargs) and kwargs["scaleFactors"] else []    # The relative scaling for each imaging round
        self.initScaleFactors = kwargs["initScaleFactors"] if ("initScaleFactors" in kwargs) and kwargs["initScaleFactors"] else []       # The initial scale factors determined from all fov
        self.pixelHistograms = kwargs["pixelHistograms"] if ("pixelHistograms" in kwargs) and kwargs["pixelHistograms"] else []         # The pixel histograms for the processed data
        self.optFovIDs = kwargs["optFovIDs"] if ("optFovIDs" in kwargs) and kwargs["optFovIDs"] else []               # The FOV ids used for optimization

        # Parameters associated with the raw data and data IO
        self.rawDataFiles = kwargs["rawDataFiles"] if ("rawDataFiles" in kwargs) and kwargs["rawDataFiles"] else  []   # List of raw data files
        self.fov2str = kwargs["fov2str"] if ("fov2str" in kwargs) and kwargs["fov2str"] else ""     # A function for converting fov ids to strings

        # Properties associated with warping
        self.affineTransforms = kwargs["affineTransforms"] if ("affineTransforms" in kwargs) and kwargs["affineTransforms"] else []        # All affine transformations
        self.residuals = kwargs["residuals"] if ("residuals" in kwargs) and kwargs["residuals"] else []               # All residuals
        self.geoTransformReport = kwargs["geoTransformReport"] if ("geoTransformReport" in kwargs) and kwargs["geoTransformReport"] else []    # The geometric transform report

        # Properties associated with sliced decoders
        self.originalMaxFovID = kwargs["originalMaxFovID"] if ("originalMaxFovID" in kwargs) and kwargs["originalMaxFovID"] else []   # The original maximum fov id number from downsampled data

        self.sliceIDs =kwargs["sliceIDs"] if ("sliceIDs" in kwargs) and kwargs["sliceIDs"] else  []  # A cell array containing fovIDs that should be associated within individual slices

        self.parameters = self.InitializeParameters(**kwargs)[0]

        # -------------------------------------------------------------------------
        # Return empty decoder if no arguments were provided
        # -------------------------------------------------------------------------
        if len(kwargs) == 0:
            return

        # -------------------------------------------------------------------------
        # Parse required input
        # -------------------------------------------------------------------------
        if (not isinstance(self.rawDataPath, str)) or self.rawDataPath == "":
            error('[Error]:invalidArguments - A valid raw data path must be provided')

        if (not isinstance(self.normalizedDataPath, str)) or self.normalizedDataPath == "":
            error('[Error]:invalidArguments - A valid normalized data path path must be provided')

        # -------------------------------------------------------------------------
        # Set internal paths
        # -------------------------------------------------------------------------
        if self.verbose:
            PageBreak()
            print('Generating MERFISHDecoder for data in ',self.rawDataPath)

        # -------------------------------------------------------------------------
        # Check validity of normalized data path
        # -------------------------------------------------------------------------
        if not os.path.exists(self.normalizedDataPath):
            os.makedirs(self.normalizedDataPath,exist_ok=True)
        print('Saving results in ',self.normalizedDataPath)

        # -------------------------------------------------------------------------
        # Check data organization file
        # -------------------------------------------------------------------------
        if (not isinstance(self.dataOrganizationPath, str)) or self.dataOrganizationPath == "":
            error('[Error]:invalidArguments - Cannot load file:' + self.dataOrganizationPath)

        # -------------------------------------------------------------------------
        # Check codebook file
        # -------------------------------------------------------------------------
        if (not isinstance(self.codebookPath, str)) or self.codebookPath == "":
            error('[Error]:invalidArguments - Cannot load file:' + self.codebookPath)

        # -------------------------------------------------------------------------
        # Load data organization file
        # -------------------------------------------------------------------------
        [self.dataOrganization, metaData,_] = LoadDataOrganization(self.dataOrganizationPath, verbose = self.verbose)
        shutil.copyfile(self.dataOrganizationPath,os.path.join(self.normalizedDataPath,"data_organization.csv"))

        # -------------------------------------------------------------------------
        # Load codebook
        # -------------------------------------------------------------------------
        [codebook, codebookHeader,_] = LoadCodebook(self.codebookPath, verbose =  self.verbose, barcodeConvFunc = bi2de)
        self.bitNames = codebookHeader['bit_names']
        self.numBits = len(codebookHeader['bit_names'])
        self.numBarcodes = len(codebook['barcode'])
        self.codebook = codebook
        self.codebookHeader = codebookHeader
        shutil.copyfile(self.codebookPath, os.path.join(self.normalizedDataPath,"the_codebook.csv"))

        # -------------------------------------------------------------------------
        # Extract properties of data from data organization file
        # -------------------------------------------------------------------------
        self.numDataChannels = metaData["numDataChannels"]
        self.numZPos = metaData["numZPos"]
        self.zPos = metaData["zPos"]

        # -------------------------------------------------------------------------
        # Sort data organization file to match the order of the bits in the
        # codebook
        # -------------------------------------------------------------------------
        isInCodebook = np.isin(self.dataOrganization["bitName"], self.bitNames) # Sort data organization to the order of bit names
        sind = np.nonzero(isInCodebook)[0]
        tempDataOrg = pd.concat([self.dataOrganization.iloc[sind,:],self.dataOrganization[~isInCodebook]])   # Order data organization: bits in codebook, then all other data channels in order of the data org file


        self.dataOrganization = tempDataOrg # Assign temporary data organization to the object property to set the proper order

        # Cross check
        if np.sum(isInCodebook) != len(self.bitNames):
            error('[Error]:invalidArguments - Not all bits in the codebook are present in the data organization file')

        # -------------------------------------------------------------------------
        # Organize the raw data files
        # -------------------------------------------------------------------------
        self.rawDataFiles = self.MapRawData() # Map the raw data

        # -------------------------------------------------------------------------
        # Load image metadata
        # -------------------------------------------------------------------------
        self.LoadImageMetaData()


    # -------------------------------------------------------------------------
    # Calculate barcode counts per feature
    # -------------------------------------------------------------------------
    def FoundFeaturesToCSV(self, **kwargs):
        # Export a single z plane of boundaries to a csv file with
        # additional found features metadata
        #
        # self.FoundFeaturesToCSV()
        # self.FoundFeaturesToCSV('downSampleFactor',N) # Down sample
        # boundaries by N-fold
        # self.FoundFeaturesToCSV('zIndex', M) Export the boundaries found
        # in the M-th z plane

        # Handle the variable input parameters
        parameters = {}

        parameters['downSampleFactor']=  10  # The degree to which the boundaries are downsampled
        parameters['zIndex']= 4             # The index of the z plane for the downsample boundaries

        for k_i in kwargs:
            parameters[k_i] = kwargs[k_i]

        # Display progress
        if self.verbose:
            PageBreak()
            print('Exporting found features')
            print('   Down sampling:',parameters["downSampleFactor"])
            print('   z Index:',parameters["zIndex"])
            localTimer = tic(0)

        # Load the found features
        foundFeatures = self.GetFoundFeatures()

        # Extract metadata for the table
        featureUIDs = [fF.uID for fF in foundFeatures]
        featureIDs = [fF.feature_id for fF in foundFeatures]
        primaryFovIDs = [fF.fovID[0] for fF in foundFeatures]
        featureVolume = [fF.abs_volume for fF in foundFeatures]

        # Handle the case of a improper requested z bondary
        if parameters["zIndex"] > foundFeatures[0].num_zPos:
            print('[Warning]: The requested z index is not within the found features. Defaulting to the first plane')
            parameters["zIndex"] = 0

        # Extract the downsample boundaries
        boundaryX = ['']*len(foundFeatures)
        boundaryY = ['']*len(foundFeatures)
        centroids = np.zeros((len(foundFeatures),3))

        # Loop over the found features to extract feature-specific info
        for f in range(len(foundFeatures)):
            # Extract centroid
            centroids[f,:]= foundFeatures[f].CalculateCentroid()
            # Extract Downsampled boundary
            lBoundary = foundFeatures[f].abs_boundaries[parameters["zIndex"]]
            if len(lBoundary)==0:
                boundaryX[f] = ''
                boundaryY[f] = ''
                continue

            # Downsample the boundary

            lBoundary = lBoundary[::parameters["downSampleFactor"],:]
            lBoundary =  np.vstack([lBoundary, lBoundary[0,:]])   # Close the boundary for display purposes

            # Convert to delimited string for simple export
            boundaryString_x = ";".join([str(x_i) for x_i in lBoundary[:,0]])
            boundaryString_y = ";".join([str(x_i) for x_i in lBoundary[:, 1]])


            boundaryX[f] =boundaryString_x
            boundaryY[f] =boundaryString_y

            # Display progress
            if self.verbose and not (f+1)%100:
                print('...completed parsing',f+1,'of',len(foundFeatures),'features')

        # Create a table to which the information is going to be added
        T = pd.DataFrame({"feature_uID":featureUIDs, 'feature_ID':featureIDs, 'primary_fovID':primaryFovIDs,
                          'abs_volume':featureVolume, 'centroid_x':centroids[:,0],'centroid_y':centroids[:,1],
                          'centroid_z': centroids[:, 2],'boundaryX':boundaryX, 'boundaryY':boundaryY})

        # Write table
        if os.path.exists(os.path.join(self.normalizedDataPath,self.reportPath)):
            os.makedirs(os.path.join(self.normalizedDataPath,self.reportPath),exist_ok=True)

        tablePath = os.path.join(self.normalizedDataPath,self.reportPath,'feature_metadata.csv')
        T.to_csv(tablePath,index=False)

        # Display progress
        if self.verbose:
            print('...wrote to',tablePath)
            print('...completed export in ',toc(localTimer),'s')

        return T

    # -------------------------------------------------------------------------
    # Calculate barcode counts per feature
    # -------------------------------------------------------------------------
    def BarcodesToCSV(self, **kwargs):
        # Load, parse, and save properties of barcodes
        #
        # self.BarcodesToCSV()
        # self.BarcodesToCSV('fieldsToExport',
        # {'fieldName1',....,'fieldNameN'}) # Export only the provided
        # fields
        # self.BarcodesToCSV('parsedBarcodes', true) Export the parsed
        # barcodes

        # -------------------------------------------------------------------------
        # Handle defaults for varariable arguments
        # -------------------------------------------------------------------------
        # Create defaults cell
        parameters={}

        # Paths to various metadata
        parameters['fieldsToExport']= ['barcode_id', 'fov_id', 'total_magnitude','area', 'abs_position',
                                       'error_bit', 'error_dir', 'feature_id', 'in_feature'] # Default barcode metadata to export

        parameters['parsedBarcodes']= True             # Export parsed or not parsed barcodes?

        # Parse variable input
        for k_i in kwargs:
            parameters[k_i] = kwargs[k_i]

        # -------------------------------------------------------------------------
        # Prepare for loading barcode list files
        # -------------------------------------------------------------------------
        # Display progress
        if self.verbose:
            PageBreak()
            print('Loading barcodes to export metadata to csv file')
            totalTimer = tic(0)

        # Determine the requested barcode type
        if parameters["parsedBarcodes"]:
            barcodePath = os.path.join(self.normalizedDataPath, self.barcodePath,'parsed_fov')
        else:
            barcodePath = os.path.join(self.normalizedDataPath, self.barcodePath,'barcode_fov')

        # Transfer some information to local variables
        fovIDs = self.fovIDs
        numFov = len(fovIDs)

        # Initialize storage for barcodes
        combinedList = pd.DataFrame()

        # Loop over a set of blocks per worker
        for b in range(numFov):
            # Determine local fov
            localFovID = fovIDs[b]

            # Display progress
            if self.verbose:
                PageBreak()
                print('Exporting metadata for fov',self.fov2str(localFovID))
                loadTimer = tic(0)

            # Load barcodes
            aList = pickle.load(open(os.path.join(barcodePath,'fov_'+self.fov2str(localFovID)+'_blist.pkl'),"rb"))

            # Check for empty barcodes
            if len(aList)==0:
                continue

            # Cut List
            aList = aList.loc[(aList.area >= self.parameters["quantification"]["minimumBarcodeArea"]) &
                              (aList.total_magnitude / aList.area >= self.parameters["quantification"]["minimumBarcodeBrightness"]), :]

            # Display progress
            if self.verbose:
               print('Loaded and cut ',len(aList),'barcodes')

            # Check for empty barcodes
            if len(aList)==0:
                continue

            # Cut list to only the requested fields
            column_ls = [c_i for c_i in parameters['fieldsToExport'] if c_i in aList.columns]
            aList =aList.loc[:,column_ls]

            # Append list
            combinedList =combinedList.append(aList)

            # Display progress
            if self.verbose:
                print('..completed in ',toc(loadTimer),'s')

        # ------------------------------------------------------------------------
        # Save the calculated data
        #-------------------------------------------------------------------------
        # Define reports path
        reportsPath = os.path.join(self.normalizedDataPath,'reports')
        if not os.path.exists(reportsPath):
            os.makedirs(reportsPath,exist_ok=True)

        # Define path to barcode_metadata
        barcodeMetadataPath = os.path.join(reportsPath,'barcode_metadata.csv')

        # Display progress
        if self.verbose:
            PageBreak()
            print('Converting to table')
            conversionTimer = tic(0)
        # Convert to table
        T = combinedList.copy()

        # Display progress
        if self.verbose:
            print('...completed in ',toc(conversionTimer),'s')
            print('Writing barcode metadata:',barcodeMetadataPath)
            writeTimer = tic(0)

        # Write table to file
        T.to_csv(barcodeMetadataPath)

        # Display progress
        if self.verbose:
            print('..completed write in ',toc(writeTimer),'s')
            print('Completed feature counts in ',toc(totalTimer),'s')

    # -------------------------------------------------------------------------
    # Calculate barcode counts per feature
    # -------------------------------------------------------------------------
    def CalculateFeatureCounts(self, **kwargs):
        # Calculate the number of barcodes per feature
        # self.CalculateFeatureCounts()

        # Display progress
        if self.verbose:
            PageBreak()
            print('Compute barcode counts for all features')
            localTimer = tic(0)

        parsedBarcodePath = os.path.join(self.normalizedDataPath, self.barcodePath,'parsed_fov')

        # -------------------------------------------------------------------------
        # Load feature boundaries
        # -------------------------------------------------------------------------
        if self.verbose:
            print('Loading found features')

        foundFeatures = self.GetFoundFeatures()

        # Determine number of unique feature ids
        featureIDs = np.unique([fF.feature_id for fF in foundFeatures])
        numFeatures = len(featureIDs)

        if self.verbose:
            print('...found ',len(featureIDs),'features')

        # Transfer some information to local variables
        numBarcodes = self.numBarcodes
        numFov = self.numFov
        fovIDs = self.fovIDs
        minDist = self.parameters["quantification"]["minimumDistanceToFeature"]

        # Define local variables for accumulation per worker
        countsPerCellExactIn = np.zeros((numBarcodes, numFeatures),dtype=np.int)
        countsPerCellCorrectedIn = np.zeros((numBarcodes, numFeatures),dtype=np.int)
        countsPerCellExactOut = np.zeros((numBarcodes, numFeatures),dtype=np.int)
        countsPerCellCorrectedOut = np.zeros((numBarcodes, numFeatures),dtype=np.int)

        # Loop over a set of blocks per worker
        for b in range(numFov):
            # Determine local fov
            localFovID = fovIDs[b]

            # Display progress
            if self.verbose:
                PageBreak()
                print('Counting for fov',self.fov2str(localFovID))
                loadTimer = tic(0)

            # Load barcodes
            aList = pickle.load(open(os.path.join(parsedBarcodePath,'fov_'+self.fov2str(localFovID)+'_blist.pkl'),"rb"))

            # Check for empty barcodes
            if len(aList)==0:
                continue

            # Cut List
            aList = aList.loc[(aList.area >= self.parameters["quantification"]["minimumBarcodeArea"]) &
                              (aList.total_magnitude/aList.area >= self.parameters["quantification"]["minimumBarcodeBrightness"]),:]

            # Cut list to a specific Z range if requested
            if len(self.parameters["quantification"]["zSliceRange"])>0:
                pos = np.stack(aList.abs_position,0)
                zPos = pos[:,2]
                aList = aList.loc[(zPos >= self.parameters["quantification"]["zSliceRange"][0]) & (zPos <= self.parameters["quantification"]["zSliceRange"][1]),:]

            if self.verbose:
                print('Loaded and cut',len(aList),'barcodes in',toc(loadTimer),'s')
                countTimer = tic(0)

            # Check for empty barcodes
            if len(aList)==0:
                continue

            # Extract needed information for exact information
            isExact = (aList.is_exact==1).values
            barcodeID = aList.barcode_id.values
            inFeature = aList.in_feature.values
            featureDist = aList.feature_dist.values

            # Convert featureID to the corresponding index in the count
            # matrix
            featureInd = np.digitize(aList.feature_id.values, list(featureIDs) + [np.inf]) # The lower edge is <= the upper is <

            #--------------------------------------------------------------------------
            featureInd = np.uint32(featureInd)
            barcodeID = np.uint32(barcodeID)
            #--------------------------------------------------------------------------

            # Compute the various quantities
            # Exact/in feature
            hist_x = barcodeID[isExact & inFeature]
            hist_y = featureInd[isExact & inFeature]
            if len(hist_x) > 0:
                countsPerCellExactIn = countsPerCellExactIn + \
                                       np.histogram2d(x= hist_x,y=hist_y, bins =[np.arange(numBarcodes+1), np.arange(numFeatures+1)])[0]

            # Corrected/in features
            hist_x = barcodeID[~isExact & inFeature]
            hist_y =  featureInd[~isExact & inFeature]
            if len(hist_x) >0:
                countsPerCellCorrectedIn = countsPerCellCorrectedIn + \
                                           np.histogram2d(x=hist_x,y=hist_y, bins =[np.arange(numBarcodes+1), np.arange(numFeatures+1)])[0]

            # Exact/out of feature
            hist_x = barcodeID[isExact & ~inFeature & (featureDist <= minDist)]
            hist_y = featureInd[isExact & ~inFeature & (featureDist <= minDist)]
            if len(hist_x)>0:
                countsPerCellExactOut = countsPerCellExactOut +\
                                        np.histogram2d(x=hist_x,y=hist_y, bins =[np.arange(numBarcodes+1), np.arange(numFeatures+1)])[0]

            # Exact/out of feature
            hist_x = barcodeID[~isExact & ~inFeature & (featureDist <= minDist)]
            hist_y = featureInd[~isExact & ~inFeature & (featureDist <= minDist)]
            if len(hist_x)>0:
                countsPerCellCorrectedOut = countsPerCellCorrectedOut + \
                                            np.histogram2d(x=hist_x,y=hist_y, bins =[np.arange(numBarcodes+1), np.arange(numFeatures+1)])[0]


            if self.verbose:
               print('..completed in ',toc(countTimer),'s')

        # ------------------------------------------------------------------------
        # Save the calculated data
        #-------------------------------------------------------------------------
        reportsPath = os.path.join(self.normalizedDataPath,'reports')
        if not os.path.exists(reportsPath):
            os.makedirs(reportsPath,exist_ok=True)

        if self.verbose:
            PageBreak()
            print('Writing data')

        # Write

        np.savetxt(os.path.join(reportsPath,'countsPerCellExactIn.csv'), countsPerCellExactIn.astype(np.int),delimiter=",")
        np.savetxt(os.path.join(reportsPath, 'countsPerCellCorrectedIn.csv'), countsPerCellCorrectedIn.astype(np.int),delimiter=",")
        np.savetxt(os.path.join(reportsPath, 'countsPerCellExactOut.csv'), countsPerCellExactOut.astype(np.int),delimiter=",")
        np.savetxt(os.path.join(reportsPath, 'countsPerCellCorrectedOut.csv'), countsPerCellCorrectedOut.astype(np.int),delimiter=",")

        if self.verbose:
            print('...wrote', os.path.join(reportsPath,'countsPerCellExactIn.csv'))
            print('...wrote', os.path.join(reportsPath,'countsPerCellCorrectedIn.csv'))
            print('...wrote', os.path.join(reportsPath,'countsPerCellExactOut.csv'))
            print('...wrote', os.path.join(reportsPath,'countsPerCellCorrectedOut.csv'))
            print('Completed compiling performance statistics at',tic(1))

        # ------------------------------------------------------------------------
        # Write metadata
        #-------------------------------------------------------------------------
        # Define the feature name file
        featuresNameFilePath = os.path.join(reportsPath,'featureNames.csv')

        # Generate the contents of the feature names file
        featureUIDs = [fF_i.uID for fF_i in foundFeatures] # Feature ids

        # Open and write file
        fid = open(featuresNameFilePath, 'w')
        fid.write("\n".join(featureUIDs) + "\n")
        fid.close()
        # Display progress
        if self.verbose:
            print('...wrote', featuresNameFilePath)

        # Define the gene name file
        geneNamesFilePath = os.path.join(reportsPath,'geneNames.csv')
        geneNames = self.codebook["name"]

        # Open and write file
        fid = open(geneNamesFilePath, 'w')
        fid.write("\n".join(geneNames) + "\n")
        fid.close()

        # Display progress
        if self.verbose:
            print('...wrote', featuresNameFilePath)

        # Display progress
        if self.verbose:
            print('Completed feature counts in ',toc(localTimer),'s')

    # -------------------------------------------------------------------------
    # Create found features report
    # -------------------------------------------------------------------------
    def GenerateFoundFeaturesReport(self):
        # Generate a report on the segmented and joined features
        #
        # self.GenerateFoundFeaturesReport()

        # Display progress
        if self.verbose:
            PageBreak()
            print('Generating the found features reports')

        # Define paths to found features
        allFoundFeaturesPath = os.path.join(self.normalizedDataPath,self.segmentationPath,'all_found_features.pkl')
        finalFoundFeaturesPath = os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl')

        # Load the original found features to determine some basic stats
        if self.verbose:
            print('...loading original found features:', allFoundFeaturesPath)
            loadTimer = tic(0)

        allFoundFeatures = pickle.load(open(allFoundFeaturesPath, 'rb'))

        if self.verbose:
            print('...loaded ',len(allFoundFeatures),'features in ',toc(loadTimer),'s')
            loadTimer = tic(0)

        # Save some basic statistics
        numOriginalFeatures = len(allFoundFeatures)
        numBroken = sum([aFF.is_broken == 1 for aFF in allFoundFeatures])
        numUnbroken = sum([aFF.is_broken == 0 for aFF in allFoundFeatures])
        zPos = allFoundFeatures[0].abs_zPos

        # Extract area
        area = np.stack([aFF.boundary_area for aFF in allFoundFeatures],axis=0)
        abs_area = np.stack([aFF.abs_boundary_area for aFF in allFoundFeatures],axis=0)

        # Extract volume
        volume = [aFF.volume for aFF in allFoundFeatures]

        # Create figure handle
        file_name = "Found feature statistics"
        fig = plt.figure(file_name,figsize=(18,12))

        # Generate subplot on feature numbers
        fig.add_subplot(2,3,1)
        plt.bar([1,2,3],[numOriginalFeatures,numBroken,numUnbroken])
        plt.xticks([1,2,3],['All', 'Broken', 'Unbroken'],rotation=90)
        plt.ylabel('Number of features')

        # Generate subplot on feature area by z position
        fig.add_subplot(2,3,2)
        avArea = np.mean(area,0)
        errArea = np.std(area,0)/np.sqrt(len(avArea))
        plt.bar(zPos, avArea, color='blue',width=0.5)
        plt.errorbar(zPos, avArea, errArea, fmt ='k.')
        plt.xlabel('Z Position (micron)')
        plt.ylabel('Average number of voxels')
        plt.title(str(round(np.mean(area[:]),2)) +'+/-'+str(round(np.std(area[:])/np.sqrt(len(area)),2))+'(SEM)')
        if len(zPos) > 1:
            zDiff = np.mean(np.diff(zPos))
            plt.xlim((np.min(zPos),np.max(zPos)) + zDiff*np.array([-1,1]))

        # Plot distribution of volume
        fig.add_subplot(2,3,3)
        plt.hist(sorted(volume)[1:-1], bins=1000)  ## remove the extreme values for batter display
        plt.xlabel('Volume (voxels)')
        plt.ylabel('Count')
        plt.title(str(round(np.mean(volume),2))+'+/-'+str(round(np.std(volume)/np.sqrt(len(volume)),2))+'(SEM)')


        # Load the final found features
        if self.verbose:
            print('...loading final found features:',finalFoundFeaturesPath)
            loadTimer = tic(0)

        finalFoundFeatures = pickle.load(open(finalFoundFeaturesPath, 'rb'))

        if self.verbose:
            print('...loaded ',len(finalFoundFeatures),'features in ',toc(loadTimer),'s')
            loadTimer = tic(0)

        # Save some basic statistics
        numOriginalFeatures = len(finalFoundFeatures)
        numSelfJoin = sum([fFF.is_broken == 1 for fFF in finalFoundFeatures])
        numDoubleJoin = sum([fFF.is_broken == 2 for fFF in finalFoundFeatures])

        # Extract area
        area = np.stack([fFF.abs_boundary_area for fFF in finalFoundFeatures],axis=0)

        # Extract volume
        volume = [fFF.abs_volume for fFF in finalFoundFeatures]

        # Generate subplot on feature numbers
        fig.add_subplot(2,3,4)
        plt.bar([1,2,3],[numOriginalFeatures,numSelfJoin,numDoubleJoin])
        plt.xticks([1,2,3],['All', 'Self-Join', 'Double-Join'],rotation=90)
        plt.ylabel('Number of features')

        # Generate subplot on feature area by z position
        fig.add_subplot(2,3,5)
        avArea = np.mean(area,0)
        errArea = np.std(area,0)/np.sqrt(len(avArea))
        plt.bar(zPos, avArea, color='blue')
        plt.errorbar(zPos, avArea, errArea, fmt='k.')
        plt.xlabel('Z Position (micron)')
        plt.ylabel('Average area (microns$^{2}$)')
        plt.title(str(round(np.mean(area),2))+ '+/-'+str(round(np.std(area)/np.sqrt(len(area)),2))+'(SEM)')
        if len(zPos) > 1:
            zDiff = np.mean(np.diff(zPos))
            plt.xlim((np.min(zPos), np.max(zPos)) + zDiff * np.array([-1, 1]))

        # Plot distribution of volume
        fig.add_subplot(2,3,6)
        plt.hist(sorted(volume)[1:-1],1000) ## remove the extreme values for batter display
        plt.xlabel('Volume (microns$^{3}$)')
        plt.ylabel('Count')
        plt.title(str(round(np.mean(volume), 2)) + '+/-' + str(round(np.std(volume) / np.sqrt(len(volume)), 2)) + '(SEM)')

        plt.tight_layout()
        # Save report figure
        plt.savefig(os.path.join(self.normalizedDataPath,self.reportPath,file_name+self.parameters["display"]["formats"]))
        plt.close()



    # -------------------------------------------------------------------------
    # Generate Summation Report
    # -------------------------------------------------------------------------
    def GenerateSummationReport(self):
        # Generate a report on the summation of features
        #
        # self.GenerateSummationReport()

        # Display progress
        if self.verbose:
            PageBreak()
            print('Generating a report on the summation of raw data channels')

        # Check status for mosaics
        useMosaic = False
        if os.path.exists(os.path.join(self.normalizedDataPath,self.mosaicPath)):
            mosaic_fils = os.listdir(os.path.join(self.normalizedDataPath,self.mosaicPath))
            if len(mosaic_fils)>0:
                useMosaic = True
                if self.verbose:
                    print('Use mosaic: ',useMosaic)


        # Load found features
        foundFeatures = self.GetFoundFeatures()
        foundFeatureFovIDs = [fF.fovID[0] for fF in foundFeatures]

        # Load the summation data
        [normalizedSignal,_,_,dataChannelNames,_] = self.GetSummedSignal()

        # Create figure handle
        fig_name = "Summation statistics"
        fig = plt.figure(fig_name,figsize=(12,12))

        # Create a master approach
        for d in range(len(dataChannelNames)):
            # Extract local data
            localData = normalizedSignal[d,:]

            # Create kernel estimate
            # [p, x] = ksdensity(localData, np.arange(0, 250, 2 * np.quantile(localData, 0.95)), 'support', [0, np.inf])
            kde = scipy.stats.gaussian_kde(localData)
            x = np.linspace(0, 2*np.quantile(localData, 0.95), 250)
            p = kde(x)


            # Create scatter plot
            plt.plot(d +1 + 0.75*(np.random.rand(len(localData))-0.5), localData, '.', c="#999",markersize=0.5)

            # Create violin
            v_x = d + 1 + 0.75*np.concatenate((p, -p[::-1]), 0)/np.max(p)/2
            v_y = np.concatenate((x,x[::-1]), 0)
            plt.plot(v_x, v_y,lw=0.8)


        # Set axis labels
        plt.xticks(np.arange(1,len(dataChannelNames)+1),dataChannelNames,rotation=90)
        plt.ylabel('Normalized Signal')

        # Save report figure

        plt.savefig(os.path.join(self.normalizedDataPath,self.reportPath, fig_name + "." +self.parameters["display"]["formats"]))
        # Close figure handle
        plt.close()

        # Generate distribution reports for all cell types
        numColors = 25
        cMap = cm.get_cmap('jet')(np.linspace(0, 1, numColors))

        # Determine properties of slices
        numSlices = len(self.sliceIDs)

        # Loop over slices
        for s in range(numSlices):
            # Check to confirm that the dataset contains the fovIDs for
            # this slice
            if len(np.setdiff1d(self.sliceIDs[s], self.fovIDs)) > 0:
                continue # Not all fovIDs are present, skip analysis

            # Extract the proper features
            goodFeatureInds = np.isin(foundFeatureFovIDs, self.sliceIDs[s])
            goodFeatures = [foundFeatures[e_i] for e_i,x_i in enumerate(goodFeatureInds) if x_i]

            # Load the slice mosaic (if requested)
            if useMosaic:
                [mosaicImageStack, coordinates] = self.GetMosaic(s)

            # Loop over data channels
            for d in range(len(dataChannelNames)):
                # Extract local data
                localData = normalizedSignal[d,goodFeatureInds]

                # Discretize the data into specific bins
                edges = [-np.inf]+ list(np.linspace(np.quantile(localData, 0.05), np.quantile(localData, 0.95), numColors-1))+[np.inf]

                colorID = np.digitize(localData, edges)

                # Create figure
                fig_name = 'Cell distributions for Channel-'+dataChannelNames[d]
                fig = plt.figure(fig_name,figsize=(8,8))

                # Add the mosaic
                if useMosaic:
                    plt.pcolor(coordinates["xLimits"],
                               coordinates["yLimits"],
                               mosaicImageStack[:,:,self.parameters["summation"]["dcIndsForSummation"][d]],
                               cmap=parula_map)


                # Loop over the colorIDs
                for c in range(numColors):
                    localFeatures = goodFeatures[colorID == c]

                    pos = np.zeros((0,2))

                    for g in range(len(localFeatures)):
                        localPos = localFeatures[g].abs_boundaries[np.ceil(localFeatures[g].num_zPos/2)]
                        pos = np.stack((pos, localPos[::self.parameters["display"]["downSample"],:],[np.nan,np.nan]),0)

                    plt.plot(pos[:,0], pos[:,1], c=cMap[c])

                # Format figure
                plt.xlabel('X Position (microns)')
                plt.ylabel('Y Position (microns)')

                # Save report figure
                if not os.path.exists(os.path.join(self.normalizedDataPath, self.reportPath,'summation_distributions','slice_'+str(s))):
                    os.makedirs(os.path.join(self.normalizedDataPath, self.reportPath,'summation_distributions','slice_'+str(s)),exist_ok=True)
                plt.savefig(os.path.join(self.normalizedDataPath, self.reportPath,'summation_distributions','slice_'+str(s),
                                         fig_name + "." + self.parameters["display"]["formats"]))
                plt.close()


    # -------------------------------------------------------------------------
    # Create feature counts report
    # -------------------------------------------------------------------------
    def GenerateFeatureCountsReport(self):
        # Generate a report on the counts per feature
        #
        # self.GenerateFeatureCountsReport()

        # Display progress
        if self.verbose:
            PageBreak()
            print('Generating the feature count reports')

        # Check for the existance of the feature reports
        reportsPath = os.path.join(self.normalizedDataPath,'reports')
        if not os.path.exists(os.path.join(reportsPath,'countsPerCellExactIn.csv')) or \
                not os.path.exists(os.path.join(reportsPath, 'countsPerCellCorrectedIn.csv')) or \
                not os.path.exists(os.path.join(reportsPath, 'countsPerCellExactOut.csv')) or \
                not os.path.exists(os.path.join(reportsPath, 'countsPerCellCorrectedOut.csv')):
            error('[Error]:missingData - The feature counts must be calculated first.')

        # Load feature counts
        cInE = np.loadtxt(os.path.join(reportsPath,'countsPerCellExactIn.csv'),delimiter=",")
        cInC = np.loadtxt(os.path.join(reportsPath,'countsPerCellCorrectedIn.csv'),delimiter=",")
        cOutE = np.loadtxt(os.path.join(reportsPath,'countsPerCellExactOut.csv'),delimiter=",")
        cOutC = np.loadtxt(os.path.join(reportsPath,'countsPerCellCorrectedOut.csv'),delimiter=",")

        # Load features
        finalFoundFeaturesPath = os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl')

        if self.verbose:
            print('...loading final found features:', finalFoundFeaturesPath)
            loadTimer = tic(0)

        finalFoundFeatures = pickle.load(open(finalFoundFeaturesPath,"rb"))

        if self.verbose:
            print('...loaded ',len(finalFoundFeatures),'features in ',toc(loadTimer),'s')
            loadTimer = tic(0)

        # Save basic counts/feature statistics
        fig_name = "Counts per feature"
        fig = plt.figure(fig_name,figsize=(18,12))

        fig.add_subplot(2,3,1)
        localData = np.sum(cInC,0)
        plt.hist(localData, np.arange(500))
        plt.xlabel('Counts per cell')
        plt.ylabel('Number of cells')
        plt.title(f'CI: {np.mean(localData):.2f} $\\pm$ {np.std(localData):.2f}')
        plt.xlim([0,500])
        # plt.yscale('log')

        fig.add_subplot(2,3,2)
        localData = np.sum(cInE,0)
        plt.hist(localData,  np.arange(1000))
        plt.xlabel('Counts per cell')
        plt.ylabel('Number of cells')
        plt.title(f'EI: {np.mean(localData):.2f} $\\pm$ {np.std(localData):.2f}')
        plt.xlim([0,500])
        # plt.yscale('log')

        fig.add_subplot(2,3,3)
        localData = np.sum(cInE+cInC,0)
        plt.hist(localData,  np.arange(1000))
        plt.xlabel('Counts per cell')
        plt.ylabel('Number of cells')
        plt.title(f'All in: {np.mean(localData):.2f} $\\pm$ {np.std(localData):.2f}')
        plt.xlim([0,500])
        # plt.yscale('log')

        fig.add_subplot(2,3,4)
        localData = np.sum(cOutC,0)
        plt.hist(localData,   np.arange(1000))
        plt.xlabel('Counts per cell')
        plt.ylabel('Number of cells')
        plt.title(f'CO: {np.mean(localData):.2f} $\\pm$ {np.std(localData):.2f}')
        plt.xlim([0,500])
        # plt.yscale('log')

        fig.add_subplot(2,3,5)
        localData = np.sum(cOutE,0)
        plt.hist(localData,  np.arange(1000))
        plt.xlabel('Counts per cell')
        plt.ylabel('Number of cells')
        plt.title(f'EO: {np.mean(localData):.2f} $\\pm$ {np.std(localData):.2f}')
        plt.xlim([0,1000])
        # plt.yscale('log')

        fig.add_subplot(2,3,6)
        localData = np.sum(cOutE,0)
        plt.hist(localData,   np.arange(2000))
        plt.xlabel('Counts per cell')
        plt.ylabel('Number of cells')
        plt.title(f'All out: {np.mean(localData):.2f} $\\pm$ {np.std(localData):.2f}')
        plt.xlim([0,1000])
        # plt.yscale('log')

        plt.tight_layout()
        # Save figure
        plt.savefig(os.path.join(reportsPath, fig_name + "." + self.parameters["display"]["formats"]))
        plt.close()


        # Save basic counts/volume stats
        fig_name = 'Counts per volume'
        fig = plt.figure(fig_name,figsize=(12,6))

        # Extract the volume values for each feature
        volume = [fFF.volume for fFF in finalFoundFeatures]

        # Volume versus all counts
        fig.add_subplot(1,2,1)
        localData = np.sum(cInC+cInE,0)
        plt.plot(volume, localData, '.')
        plt.xlabel('Volume (microns^3)')
        plt.ylabel('Total counts within feature')

        fig.add_subplot(1,2,2)
        localData = np.sum(cInC+cInE,0)/volume
        localData = localData[~np.isnan(localData)]
        plt.hist(localData, np.linspace(0,0.025, 100),rwidth=0.00025)
        plt.xlabel('Density (counts/microns^3)')
        plt.ylabel('Counts')
        plt.xlim([-0.0001,0.03])
        plt.title(f'Median: {np.median(localData):.4f} +/- {scipy.stats.iqr(localData):.4f}(iqr)')

        plt.tight_layout()
        # Save figure
        plt.savefig(os.path.join(reportsPath, fig_name + "." + self.parameters["display"]["formats"]))
        plt.close()


    # -------------------------------------------------------------------------
    # Load image metadata
    # -------------------------------------------------------------------------
    def LoadImageMetaData(self):
        # fovPos = MapFOVPositions()
        # Extract fov positions from raw data meta data and set image size

        # -------------------------------------------------------------------------
        # Check that raw file metadata has been loaded
        # -------------------------------------------------------------------------
        if len(self.rawDataFiles)==0:
            error('[Error]:nonexistantData: Raw data files have not yet been loaded')

        # -------------------------------------------------------------------------
        # Loop over fov in order of fovIDs and load stage pos
        # -------------------------------------------------------------------------
        for f in range(self.numFov):
            # Select the first data channel image file to determine
            # location of meta data

            # Handle camera
            if 'imagingCameraID' not in self.dataOrganization:
                fileInd = np.nonzero((self.rawDataFiles["imageType"].values==self.dataOrganization["imageType"][0]) & \
                                     (self.rawDataFiles["imagingRound"].values == self.dataOrganization["imagingRound"][0]) & \
                                     (self.rawDataFiles["fov"].values == self.fovIDs[f]))[0]
            else:
                fileInd = np.nonzero((self.rawDataFiles["imageType"].values==self.dataOrganization["imageType"][0]) & \
                                     (self.rawDataFiles["imagingRound"].values == self.dataOrganization["imagingRound"][0]) & \
                                     (self.rawDataFiles["cameraID"].values==self.dataOrganization["imagingCameraID"][0]) & \
                                     (self.rawDataFiles["fov"].values == self.fovIDs[f]))[0]

            # Check for existance
            if len(fileInd)==0:
                print('Error: missing fov id:',self.fov2str(self.fovIDs[f]))
                error('[Error]:missingFile: Could not find a raw data file')
            fileInd = fileInd[0]
            # Switch on raw data type to determine how load metadata
            if self.imageExt in ['dax', 'tiff', 'tif']:
                #Switch on the version of hal
                if self.hal_version == 'hal1':
                    if self.imageExt in ['dax']:
                        # Load image meta data
                        infoFile = ReadInfoFile(self.rawDataFiles["filePath"][fileInd], verbose=False)
                    else:
                        # THIS CASE IS RESERVED FOR FUTURE USE
                        error('[Error]:unsupportedExt - Tiff data are not yet for hal1 info file structure')

                elif self.hal_version == 'hal2':
                    # UNDER CONSTRUCTION: THIS NEEDS TO BE UDPDATED
                    # TO HAVE A ROBUST FUNCTION FOR LOADING THE XML
                    # FILE ASSOCIATED WITH THE IMAGE

                    infoFile = {}

                    # Determine the filename and strip off the
                    # camera tag if necessary
                    baseFile = self.rawDataFiles["filePath"][fileInd]
                    baseFile = baseFile[:-4] # Strip off the dax extension
                    if baseFile[-3:] in ['_c1', '_c2']:
                        baseFile = baseFile[:-3]

                    # Load the xml file
                    with open(baseFile+'.xml',"r") as f_xml:
                        xml_doc = f_xml.read()
                    xDoc = BS(xml_doc, 'html.parser')

                    # Find the stage position node and get its data
                    allPosItems = xDoc.find_all('stage_position')
                    thisListItem = allPosItems[0]
                    posString = thisListItem.text

                    pos = posString.split(',')

                    infoFile["Stage_X"] = float(pos[0])
                    infoFile["Stage_Y"] = float(pos[1])

                    # Find the height and width of the image to
                    # compute image size
                    xSizeList = xDoc.find_all('x_pixels')
                    xSizeItem = xSizeList[0]
                    xSizeString = xSizeItem.text

                    infoFile["frame_dimensions"]= [int(xSizeString)]

                    ySizeList = xDoc.find_all('y_pixels')
                    ySizeItem = ySizeList[0]
                    ySizeString = ySizeItem.text

                    infoFile["frame_dimensions"].append(int(ySizeString))

                else:
                    error('[Error]:unsupportedExt: An unrecognized hal version was provided')

                # Archive stage position
                self.fovPos[f] = [infoFile["Stage_X"],infoFile["Stage_Y"]]  # Record stage x, y

                # Archive image size (only the first fov)
                if f==1:
                    self.imageSize = infoFile["frame_dimensions"]
            else:
                error('[Error]:unsupportedExt: This file extension is not supported')


    # -------------------------------------------------------------------------
    # Map raw data file
    # -------------------------------------------------------------------------
    def MapRawData(self):
        # foundFiles = mDecoder.MapRawData()
        # Map the organization of the raw data files in the raw data path
        # according to the data organization file

        # Display progress
        if self.verbose:
            PageBreak()
            print('Finding all',self.imageExt,'files')
        parseTimer = tic()

        # Extract unique image types
        uniqueImageType,ia = np.unique(self.dataOrganization["imageType"],return_index=True)
        print('Found',len(uniqueImageType),'image types')

        for i in range(len(uniqueImageType)):
            print('...Parsing',uniqueImageType[i],'with:',self.dataOrganization["imageRegExp"][ia[i]])

        # Loop over all file name patterns
        foundFiles = pd.DataFrame()
        for i in range(len(uniqueImageType)):
            # Build file data structure
            newFiles,_ = BuildFileStructure(self.rawDataPath,
                                          fileExt=self.imageExt,
                                          fieldNames=['imageType', 'fov', 'imagingRound', 'cameraID'], # Required fields
                                          fieldConv = [str, int, int, str],
                                          regExp = self.dataOrganization["imageRegExp"][ia[i]],
                                          requireFlag = uniqueImageType[i]
                                          )
            newFiles = pd.DataFrame(newFiles)
            # Coerce empty imageRound fields
            if "imagingRound" in newFiles and np.any(newFiles["imagingRound"]==""):
                newFiles["imagingRound"] = -1 # Flag that no image round was specified

            # Coerce empty cameraID fields
            if "cameraID" in newFiles and np.any(newFiles["cameraID"].isnull()):
                newFiles["cameraID"] = '' # Flag that no image round was specified

            # Combine file structures
            if len(newFiles) > 0:
                foundFiles = foundFiles.append(newFiles)
        print('...completed parse in',toc(parseTimer),'s')

        foundFiles.reset_index(drop=True,inplace=True)

        # Check to see if any files were found
        if len(foundFiles)==0:
            error('[Error]:noFoundFiles: No files matching the patterns in the data organization file were found!')

        # Handle the occasional parsing error
        indsToKeep = foundFiles["imageType"].isin(uniqueImageType).to_numpy()
        if np.any(~indsToKeep) and self.verbose:
            print('...found parsing errors')
            print('...removing',np.sum(~indsToKeep),'files')

        foundFiles = foundFiles.loc[indsToKeep,:]

        # Remove replicates
        _, ia = np.unique(foundFiles["name"],return_index=True) # Find all unique file names
        if self.verbose:
            print('...removing',len(foundFiles) - len(ia),'replicated files')
        foundFiles = foundFiles.loc[ia,:]

        # Compile properties
        self.fovIDs = np.unique(foundFiles["fov"])
        self.imageRoundIDs = np.unique(foundFiles["imagingRound"])
        self.numFov = len(self.fovIDs)
        self.numImagingRounds = len(self.imageRoundIDs)
        self.cameraIDs = np.unique(foundFiles["cameraID"])
        self.numCameraIDs = len(self.cameraIDs)

        # Allocate memory for stage positions
        self.fovPos = np.empty((self.numFov, 2)) # These entries are in the same order as the fovIDs.
        self.fovPos[:] = np.nan

        # Display progress
        if self.verbose:
            print('Found',len(foundFiles),'files')
            print('...',self.numFov,'fov')
            print('...',self.numImagingRounds,'imaging rounds')
            PageBreak()
            print('Checking consistency of raw data files and expected data organization')
            localTimer = tic()

        # Define useful fov to string conversion function (uniform padding)
        padNum2str = lambda x,y: str(x).zfill(int(np.ceil(np.log10(y+1))))
        self.fov2str = lambda x: padNum2str(x, np.max(self.fovIDs))

        #Run a cross check
        # Loop over each data channcel for each fov
        for c in range(self.numDataChannels):
            # Extract local properties of the channel
            localImageType = self.dataOrganization["imageType"][c]
            localImagingRound = self.dataOrganization["imagingRound"][c]
            if 'imagingCameraID' in self.dataOrganization: # Handle backwards compatibility or single camera systems
                localCameraID = self.dataOrganization["imagingCameraID"][c]
            else:
                localCameraID = ''

            # Extract the fovIDs for files with these parameters
            condation = (foundFiles["imageType"]==localImageType) & \
                        (foundFiles["imagingRound"] == localImagingRound) &\
                        (foundFiles["cameraID"]==localCameraID)
            foundFovIDs = foundFiles.loc[condation,"fov"].values

            # Find the fovIDs that are missing
            missingFovIDs = np.setdiff1d(foundFovIDs, self.fovIDs)

            for f in range(len(missingFovIDs)):
                print('An error has been found with the following file!')
                print('   FOV:',missingFovIDs)
                print('   imageType:',self.dataOrganization["imageType"][c])
                print('   imagingRound:',self.dataOrganization["imagingRound"][c])
                print('   imagingCameraID:',localCameraID)
                error('[Error]:invalidFileInformation: Either a file is missing or there are multiple files that match an expected pattern.')

            #Display Progress
            print('...completed,',c+1,'channel of',self.numDataChannels)

        # Display progress
        if self.verbose:
            print('...completed in ',toc(localTimer),' s')
            print('...no problems were found.')

        return foundFiles

    # -------------------------------------------------------------------------
    # Segment features (cells) within individual fov
    # -------------------------------------------------------------------------
    def SegmentFOV(self, fovIDs):
        # Segment features, e.g. cells, in the specified fov
        # SegmentFOV([])       # Segment all fov
        # SegmentFOV(fovIDs)   # Segment the fov that match the specified fovids

        # -------------------------------------------------------------------------
        # Determine properties of the requested fov ids
        # -------------------------------------------------------------------------
        if fovIDs == [] or fovIDs == "":
            fovIDs = self.fovIDs
        elif not np.all([f_i in self.fovIDs for f_i in fovIDs]):
            error('[Error]:invalidArguments - An invalid fov id has been requested')

        # -------------------------------------------------------------------------
        # Extract local copy of parameters
        # -------------------------------------------------------------------------
        parameters = self.parameters["segmentation"]

        # -------------------------------------------------------------------------
        # Extract necessary parameters for segmentation
        # -------------------------------------------------------------------------
        if parameters["segmentationMethod"] == 'seededWatershed':
            # Identify seed and watershed frames based on data
            # organization file
            dataChannelNames = self.dataOrganization["bitName"]

            ID =  np.nonzero(dataChannelNames.to_numpy()==parameters["watershedSeedChannel"])[0]
            if ID.size > 1:
                error('[Error]:invalidDataType - The requested seed channel has problems')
            seedFrames = np.arange(ID*self.numZPos,(ID+1)*self.numZPos)

            ID = np.nonzero(dataChannelNames.to_numpy()==parameters["watershedChannel"])[0]
            if ID.size > 1:
                error('[Error]:invalidDataType - The requested watershed channel has problems')
            watershedFrames = np.arange(ID*self.numZPos,(ID+1)*self.numZPos)

        else:
            error('[Error]:invalidArguments - The provided segmentation method is not currently supported.')

        # Handle the request to ignore z
        if parameters["ignoreZ"]:
            seedFrames = [seedFrames[0]]
            watershedFrames = [watershedFrames[0]]

        # -------------------------------------------------------------------------
        # Run processing on individual fov in parallel (if requested)
        # -------------------------------------------------------------------------
        # Loop over requested fov
        for f in fovIDs:
            # Determine local fov id
            localFovID = f

            # Create display strings
            if self.verbose:
                PageBreak()
                print('Started segmentation for fov',localFovID)
                fovTimer = tic(0)
                localTimer = tic(0)

            # Create file path and check for existance

            foundFeaturesPath = os.path.join(self.normalizedDataPath,
                                             self.segmentationPath,
                                             'found_features_fov_'+self.fov2str(localFovID)+'.pkl')

            if os.path.exists(foundFeaturesPath):
                if self.overwrite: # If overwrite, then delete and overwrite
                    os.remove(foundFeaturesPath)
                    if self.verbose:
                        print('...overwriting existing analysis')
                else:
                    if self.verbose:
                        print('...found existing analysis. Skipping.')
                    continue
            # Display progress
            if self.verbose:
                print('...using the',parameters["segmentationMethod"],'method')


            # Define tiff file name
            tiffFileName = os.path.join(self.normalizedDataPath,self.warpedDataPath,'fov_'+self.fov2str(localFovID)+'.tif')

            # Clear any previous figure handles
            figHandles = []

            # Allocate memory for the necessary frames
            localWatershedFrames = np.array([np.zeros(self.imageSize)]*len(watershedFrames))
            localSeedFrames = np.array([np.zeros(self.imageSize)]*len(seedFrames))

            # Load and preprocess the necessary frames
            for z in range(len(watershedFrames)):
                # Load frames for watershed and seed generation
                localWatershedFrames[z] = imread(tiffFileName, key = watershedFrames[z])
                localSeedFrames[z] = imread(tiffFileName, key=seedFrames[z])


                # Filter seed frame (if requested)
                if parameters["seedFrameFilterSize"] > 0:
                    # localSeedFrames[:, :, z] = imgaussfilt(localSeedFrames[:, :, z], parameters["seedFrameFilterSize"])
                    localSeedFrames[z] = gaussian(localSeedFrames[z],
                                                        sigma=parameters["seedFrameFilterSize"], preserve_range=True,
                                                        truncate=2)

                # Filter watershed image
                if parameters["watershedFrameFilterSize"] > 0:
                    # localWatershedFrames[:,:,z] = imgaussfilt(localWatershedFrames[:,:,z], parameters["watershedFrameFilterSize"])
                    localWatershedFrames[z] = gaussian(localWatershedFrames[z],
                                                             sigma=parameters["watershedFrameFilterSize"],
                                                             preserve_range=True, truncate=2)
                # Generate report
                if parameters["saveSegmentationReports"]:
                    # Generate a figure handle for this z slice
                    fig_name = 'Segmentation report fov_'+self.fov2str(localFovID)+'_z_'+str(z)
                    fig = plt.figure(fig_name,figsize=(15,10))
                    # Update segmentation report
                    fig.add_subplot(2,3,1)
                    plt.imshow(np.rint(localSeedFrames[z]),cmap="gray")
                    plt.title('Filtered seed frame')

                    figHandles.append(fig)

            # Erode seed frame (if requested)
            if len(parameters["seedFrameErosionKernel"]) > 0:
                localSeedFrames = ndimage.grey_erosion(localSeedFrames,footprint=ball(1))
                localSeedFrames = np.array([cv2.erode(x,parameters["seedFrameErosionKernel"],borderType=cv2.BORDER_REFLECT)
                                            for x in localSeedFrames])

            # Create a mask image
            if parameters["seedThreshold"] == None: # Use Otsu's method to determine threshold
                mask = localSeedFrames < threshold_otsu(localSeedFrames)
            elif isinstance(parameters["seedThreshold"],str) and parameters["seedThreshold"] == 'adaptive':  # Use an adaptive method
                thresholdFilterSize = int(2 * np.floor(localSeedFrames.shape[1] / 16) + 1)
                mask = np.array([x < 1.1 * threshold_local(x, thresholdFilterSize, method='mean', mode='nearest')
                                     for x in localSeedFrames])

            else: # Use a constant user specified value
                mask = localSeedFrames < parameters["seedThreshold"]

            # Create a 3D mask
            localSeedFrames[mask] = 0

            # Find regional max
            # seeds = imregionalmax(localSeedFrames)
            # https://stackoverflow.com/questions/27598103/what-is-the-difference-between-imregionalmax-of-matlab-and-scipy-ndimage-filte
            seeds = morphology.local_maxima(localSeedFrames, allow_borders=True)

            # Connect seeds that are close
            if len(parameters["seedConnectionKernel"]) > 0:
                seeds = ndimage.morphology.binary_dilation(seeds, structure=ndimage.morphology.generate_binary_structure(3, 1))
                seeds = np.array([ndimage.morphology.binary_dilation( x, structure=parameters["seedConnectionKernel"]) for x in seeds])

            # Update the segmentation reports
            if parameters["saveSegmentationReports"]:
                for z in range(len(watershedFrames)):
                    # Set the active figure
                    fig_z = figHandles[z]

                    # Update debug display
                    fig_z.add_subplot(2,3,2)
                    #imshow(imoverlay(imadjust(localSeedFrames(:,:,z)), ~mask(:,:,z), 'red'))
                    plt.imshow(localSeedFrames[z], cmap="gray")
                    plt.title('Eroded, masked seed frame')

                    fig_z.add_subplot(2,3,3)
                    plt.imshow(imadjust(localSeedFrames[z]),cmap="gray")
                    plt.imshow(seeds[z], 'Reds',alpha=0.5)
                    plt.title('Seed frame with seed locations')

                    # Update debug display
                    fig_z.add_subplot(2,3,4)
                    plt.imshow(localWatershedFrames[z], cmap="gray")
                    plt.title('Watershed frame')


            # Parse the seeds to find seeds that are independent in one
            # frame and joined in another

            def create_region_image(shape, c):
                region = np.zeros(shape)
                for x in c.coords:
                    region[x[0], x[1], x[2]] = 1
                return region

            components = measure.regionprops(measure.label(seeds))

            # Clear the previous seeds (they will be added back below)
            seeds = np.zeros(seeds.shape)

            # Loop over all found 3D seeds and examine them in 2D
            for c in components:
                # Prepare an image of just this seed
                seedImage = create_region_image(seeds.shape, c)

                # Determine the properties of this seed in each slice
                localProps = [measure.regionprops(measure.label(x)) for x in seedImage]

                # Determine how many unique regions are in each slice
                numSeeds = [len(localProp_i) for localProp_i in localProps]

                # And build a consensus centroid for each of them
                if np.all([x < 2 for x in numSeeds]): # A single seed in all frames (or no seed)
                    # Extract seed centroids: the median position in
                    # the consensus position
                    goodFrames =[i for i, x in enumerate(numSeeds) if x == 1]
                    allGoodProps =  [y for x in goodFrames for y in localProps[x]]
                    seedPos = np.round([np.median([x.centroid for x in allGoodProps], axis=0)]).astype(int)
                else: # In at least one frame there is a double seed that has been combined in another frame
                    # Find the mediod using kmediods and the maximum
                    # number of observed seeds in any frame
                    goodFrames = [i for i, x in enumerate(numSeeds) if x > 1]
                    allGoodProps = [y for x in goodFrames for y in localProps[x]]
                    allGoodCentroid = [aGP.centroid for aGP in allGoodProps]
                    km = kmedoids.kmedoids(allGoodCentroid, np.random.choice(np.arange(len(allGoodCentroid)),size=np.max(numSeeds)))
                    km.process()
                    seedPos = np.round([allGoodCentroid[x] for x in km.get_medoids()]).astype(int)

                # Add back the seedPos in all z slices
                for s in seedPos:
                    for f in goodFrames:
                        seeds[f,s[0],s[1]] = 1

            # Dilate seed centroids
            if len(parameters["seedDilationKernel"]) > 0:
                seeds = ndimage.morphology.binary_dilation(seeds, structure=ball(1))
                seeds = np.array([ndimage.morphology.binary_dilation(x, structure=parameters["seedDilationKernel"]) for x in seeds])

            # Create in cell mask
            if len(parameters["watershedFrameThreshold"]) == None:
                # Calculate threshold using Otsu's method
                level = threshold_otsu(localWatershedFrames)
                # Calculate mask
                inCellMask = localWatershedFrames >= level
            elif isinstance(parameters["watershedFrameThreshold"],str) and parameters["watershedFrameThreshold"]=='adaptive':
                thresholdFilterSize = int(2 * np.floor(localWatershedFrames.shape[1] / 16) + 1)
                inCellMask = np.array([x >= 1.1 * threshold_local(x, thresholdFilterSize, method='mean', mode='nearest')
                                       for x in localWatershedFrames])
            else:
                inCellMask = localWatershedFrames >= parameters["watershedFrameThreshold"]

            # Fill in holes in the cell mask
            for z in range(len(inCellMask)):
                # inCellMask[:,:,z] = imfill(inCellMask[:,:,z], 'holes')
                inCellMask[z] = ndimage.binary_fill_holes(inCellMask[z])

            # Convert to double
            localWatershedFrames = np.double(localWatershedFrames)

            # Set uniform range to watershed image
            localWatershedFrames = (localWatershedFrames - np.min(localWatershedFrames)) / \
                (np.max(localWatershedFrames) - np.min(localWatershedFrames))

            # Create image for watershed
            localWatershedFrames = 1- localWatershedFrames # Invert image
            localWatershedFrames[~inCellMask] = 0# Mask with in cell mask
            localWatershedFrames[seeds] = 0 # Insert seeds for watershed

            # Remove local minima, forcing to only define watersheds via
            # imposed catch basins
            localWatershedFrames = imimposemin(localWatershedFrames, ~inCellMask | seeds)

            # Watershed
            L = watershed(localWatershedFrames, measure.label(seeds), mask=inCellMask, connectivity=np.ones((3, 3, 3)), watershed_line=True)

            # Compute label matrix color map
            cMap = cm.get_cmap('jet')(np.linspace(0, 1, np.max(L)))
            if len(cMap)==0:
                cMap=["blue","green","red", 'yellow', 'magenta', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen']

            # Update the segmentation reports
            if parameters["saveSegmentationReports"]:
                for z in range(len(watershedFrames)):
                    # Set the active figure
                    fig_z = figHandles[z]

                    fig_z.add_subplot(2,3,5)
                    plt.imshow(imadjust(localWatershedFrames[z]),cmap="gray")
                    plt.imshow( ~inCellMask[z], 'Reds', alpha=0.5)
                    plt.title('Watershed frame')

                    fig_z.add_subplot(2,3,6)
                    plt.imshow(label2rgb(L[z], colors = cMap,bg_label = -1))
                    plt.title('Label matrix')

            # Display progress
            if self.verbose:
                print('...completed watershed in',toc(localTimer),'s')
                localTimer = tic(0)

            # Find the label(s) that corresponds to the regions not in
            # cells
            uniqueLabels = np.unique(L)
            isBadLabel = np.zeros((len(uniqueLabels),),dtype=bool)
            for J in range(len(isBadLabel)):
                isBadLabel[J] = bool(np.any((L== uniqueLabels[J]) & ~inCellMask))

            badLabels = uniqueLabels[isBadLabel]

            # Remove this label from feature construction
            L[np.isin(L, badLabels)] = 0 # Label for no feature

            # Determine the number of features
            uniqueLabels = np.unique(L)
            uniqueLabels = np.setdiff1d(uniqueLabels, 0) # Remove a marker of no features
            numFeatures = len(uniqueLabels)

            # Loop over all features creating a found feature
            foundFeatures = []
            for n in range(numFeatures):
                # Convert label matrix into FoundFeatures
                localFeature = FoundFeature()
                localFeature.createFoundFeature(L == uniqueLabels[n],    # label matrix
                    localFovID,                                     # fovID
                    self.fovPos[self.fovIDs == localFovID,:],         # fov center position
                    self.pixelSize,                                   # pixelSize
                    self.parameters["decoding"]["stageOrientation"],        # stage orientation
                    self.parameters["segmentation"]["boundingBox"],        # bounding box
                    self.zPos,                                      # z positions for each stack
                    uniqueLabels[n])                                   # the label associated with this feature

                # Only keep features if they were not completely
                # removed upon crop
                if localFeature.abs_volume > 0:
                    foundFeatures.append(localFeature)
            # Display progress
            if self.verbose:
                print('...found',numFeatures,' features in',toc(localTimer),'s')
                print('...keeping',len(foundFeatures),'features after cuts by bounding box')
                localTimer = tic(0)

            # Identify features that are contained within another feature
            # (These arise due to failure of a seed to drive the
            # creation of a watershed region larger than itself)
            isGoodFeature = np.ones((len(foundFeatures),),dtype=bool)
            for i in range(len(foundFeatures)):
                doesOverlap = np.zeros((len(foundFeatures),),dtype=bool)
                for j in range(len(foundFeatures)):
                    # Skip the identity case
                    if i==j:
                        continue
                    # Check if feature j contains feature i
                    # doesOverlap[j] = foundFeatures[j].DoesFeatureOverlap(foundFeatures[i]) ## DoesFeatureOverlap ~~ doesContainsFeature
                    doesOverlap[j] = foundFeatures[j].doesContainsFeature(foundFeatures[i])  ## this line is ALMOST equal to the upper line
                isGoodFeature[i]= ~np.any(doesOverlap) # If no feature j contains feature i, then feature i is good

            # Slice away features that are contained (at least
            # partially) within another feature
            foundFeatures = [foundFeatures[i] for i,iG_i in enumerate(isGoodFeature) if iG_i]

            # Display progress
            if self.verbose:
                print('...completed cross checks for overlapping features in',toc(localTimer), 's')
                print('...keeping',len(foundFeatures),'non-overlapping features')
                localTimer = tic(0)

            # Create the path for the progress figures
            localSavePath = os.path.join(self.normalizedDataPath,self.segmentationPath, 'fov_images')

            # Update, save, and close the segmentation reports
            if parameters["saveSegmentationReports"]:
                for z in range(len(watershedFrames)):
                    # Set figure
                    fig_z = figHandles[z]

                    # Plot all features
                    plt.subplot(2,3,4)
                    for F in foundFeatures:
                        plt.plot(F.boundaries[0][z][:,0],F.boundaries[0][z][:,1], 'r',linewidth=0.7)

                    # Save figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.normalizedDataPath, self.segmentationPath, 'fov_images',fig_z._label + ".png"))
                    plt.close()

            # Save the found features
            with open(foundFeaturesPath, "wb") as fout:
                pickle.dump(foundFeatures, fout, pickle.HIGHEST_PROTOCOL)

            # Flush display buffer
            if self.verbose:
                print('...saved',foundFeaturesPath,'and figures in',toc(localTimer),'s')
                print('...completed fov', self.fov2str(localFovID),'in',toc(fovTimer),'s')


    # -------------------------------------------------------------------------
    # Generate a low resolution mosaic stacks of the image data
    # -------------------------------------------------------------------------
    def GenerateLowResolutionMosaic(self):
        # Generate a low resolution mosaic of the data using sliceIDs to
        # define different slices (if provided)
        #
        # self.GenerateLowResolutionMosaic()

        # Handle the case that zInd was not provided
        zInd = self.parameters["display"]["mosaicZInd"]
        if zInd < 0 or zInd > self.numZPos:
            print('[Warning]: invalidParameter - The provided mosaic z index is not within the z range. Assuming the first z position')
            zInd = 0

        # Determine the slice ids
        if len(self.sliceIDs) == 0:
            sliceIDs = [self.fovIDs]
        else:
            sliceIDs = self.sliceIDs

        numSlices = len(sliceIDs)

        # Determine the crop properties
        boundingBox = np.round(self.parameters["segmentation"]["boundingBox"]/(self.pixelSize/1000)) # Scale bounding box from microns to pixels
        startPixels = (boundingBox[0:2] + np.array([self.imageSize[0]/2,self.imageSize[1]/2])).astype(int)
        endPixels = (startPixels + boundingBox[2:4]).astype(int)

        # Display progress
        if self.verbose:
            PageBreak()
            print('Downsampling and saving ',len(sliceIDs),'mosaics')
            totalTimer = tic(0)

        # Make the downsample folder if necessary
        if not os.path.exists(os.path.join(self.normalizedDataPath,self.mosaicPath)):
            os.makedirs(os.path.join(self.normalizedDataPath,self.mosaicPath),exist_ok=True)

        # Loop over the individual slices
        for s in range(len(sliceIDs)):
            # Display progress
            if self.verbose:
                print('...rendering channels for slice',s)
                print('...preparing files')
                localTimer = tic(0)

            # Determine the local fovs
            localFovIDs = self.fovIDs[np.isin(self.fovIDs,sliceIDs[s])]

            # Check for existing slices and skip if not present
            if len(localFovIDs)==0:
                if self.verbose:
                    print('Could not find the requested fov IDs')
                continue

            # Compute locations to determine ordering of files
            fovPositions = self.fovPos[np.isin(self.fovIDs, localFovIDs),:] # Extract positions
            [_, _, xInd] = np.unique(fovPositions[:,0],return_index=True,return_inverse=True) # Map positions to row/column indices (1st, 2nd, etc...)
            [_, _, yInd] = np.unique(fovPositions[:,1],return_index=True,return_inverse=True)

            numXFrames = np.max(xInd)+1  ## +1 : convert index to count number
            numYFrames = np.max(yInd)+1

            # Determine the size of the final image by a test downsample
            testImage = np.zeros(self.imageSize) # Generate empty image
            testImage = testImage[startPixels[0]:(endPixels[0]+1),  startPixels[1]:(endPixels[1]+1)] # Crop
            width = int(testImage.shape[1] / self.parameters["display"]["downSample"]) + 1
            height = int(testImage.shape[0] /self.parameters["display"]["downSample"]) + 1
            dim = (width, height)
            testImage = cv2.resize(testImage, dim,interpolation = cv2.INTER_AREA) # Resize
            [numXPixels, numYPixels] = testImage.shape # Determine final downsampled size

            numFinalPixelsX = numXPixels*numXFrames
            numFinalPixelsY = numYPixels*numYFrames

            # Determine the coordinate system of the final image
            coordinates = {}
            coordinates["xLimits"] = np.array([np.min(fovPositions[:,0]),np.max(fovPositions[:,0])]) + \
                                     self.parameters["segmentation"]["boundingBox"][0] + \
                                     np.array([0, self.parameters["segmentation"]["boundingBox"][2]]) + \
                                     self.parameters["display"]["downSample"]/2*self.pixelSize/1000*np.array([1,-1])

            coordinates["yLimits"] = np.array([np.min(fovPositions[:,1]),np.max(fovPositions[:,1])])+ \
                                     self.parameters["segmentation"]["boundingBox"][1] + \
                                     np.array([0,self.parameters["segmentation"]["boundingBox"][3]]) + \
                                     self.parameters["display"]["downSample"]/2*self.pixelSize/1000*np.array([1,-1])

            # Save the coordinate system
            with open(os.path.join(self.normalizedDataPath,self.mosaicPath,'coordinates_slice_'+str(s)+'.pkl'),"wb") as fout:
                pickle.dump(coordinates,fout, protocol=pickle.HIGHEST_PROTOCOL)


            # Create tiff file to write
            tiffFileName = os.path.join(self.normalizedDataPath,self.mosaicPath,'slice_'+str(s)+'.tif')
            tiffFile = TiffWriter(tiffFileName, bigtiff=True,append=True)

            # Create tiff tags
            # Create tiff Tag structure
            tiffTagStruct = {}
            # tiffTagStruct["shape"] = self.imageSize
            # tiffTagStruct["ImageLength"] = numFinalPixelsX ## tag id: 257
            # tiffTagStruct["ImageWidth"] = numFinalPixelsY ## tag id: 256
            tiffTagStruct["photometric"] = TIFF.PHOTOMETRIC.MINISBLACK
            # tiffTagStruct["bitspersample"] = 16
            # tiffTagStruct["samplesperpixel"] = 1 ## tag id: 277
            # tiffTagStruct["sampleformat"] = TIFF.SAMPLEFORMAT.IEEEFP  ## tag id: 339
            tiffTagStruct["rowsperstrip"] = 16
            tiffTagStruct["planarconfig"] = TIFF.PLANARCONFIG.CONTIG
            tiffTagStruct["software"] = 'Ruifeng'
            tiffTagStruct["description"] = 'images=' + str(self.numDataChannels) + '\n' + \
                                           'channels=1\n' + \
                                           'slices=1\n' + \
                                           'frames='+str(self.numDataChannels)+'\n' + \
                                           'hyperstack=True\n' + \
                                           'loop=False\n'
            tiffExtrTags = [(339, "i", 1, TIFF.SAMPLEFORMAT.IEEEFP, False), (277, "i", 1, 1, False),
                            (256, "i", 1, numFinalPixelsY, False),(257, "i", 1, numFinalPixelsX, False)]



            # Loop over data channels
            for c in range(self.numDataChannels):
                # Reset frame (unnecessary)
                dsFrame = np.empty((numFinalPixelsX, numFinalPixelsY),dtype="uint16")
                dsFrame[:] = 65535

                # Loop over fields of view
                for v in range(len(localFovIDs)):

                    # Read image from this file
                    localFileName = os.path.join(self.normalizedDataPath,self.warpedDataPath,'fov_'+self.fov2str(localFovIDs[v])+'.tif')
                    localImage = imread(localFileName, key=self.numZPos*c + zInd)

                    # Crop
                    localImage = localImage[startPixels[0]:(endPixels[0]+1),startPixels[1]:(endPixels[1]+1)] # Crop

                    # Downsample
                    width = int(localImage.shape[1] / self.parameters["display"]["downSample"]) + 1
                    height = int(localImage.shape[0] / self.parameters["display"]["downSample"]) + 1
                    dim = (width, height)
                    localImage = cv2.resize(localImage, dim, interpolation=cv2.INTER_AREA)

                    [dsNumPixelsX, dsNumPixelsY] = localImage.shape

                    # Invert/rotation (kludge)
                    localImage = localImage.T

                    # Place into frame
                    dsFrame[xInd[v]*dsNumPixelsX:(xInd[v]+1)*dsNumPixelsX,
                            yInd[v]*dsNumPixelsY:(yInd[v]+1)*dsNumPixelsY] = localImage

                # Write tiff frame
                tiffFile.write(dsFrame.T, **tiffTagStruct, extratags=tiffExtrTags)  # The matrix transpose is kludge

                # Display progress
                print('...completed channel',c,'of',self.numDataChannels,'in',toc(localTimer),'s')
                localTimer = tic(0)
            # Close tiff file
            tiffFile.close()
        # Display progress
        if self.verbose:
            print('...completed all low resolution mosaics in ',toc(totalTimer),'s')


    # -------------------------------------------------------------------------
    # Sum raw fluorescence signals within individual boudnaries for each FOV
    # -------------------------------------------------------------------------
    def SumRawSignalFOV(self,fovIDs):
        # Sum the raw signal from each channel within boundaries
        #
        # self.SumRawSignalFOV(fovIDs) # Analyze the specified fov
        # self.SumRawSignalFOV([])     # Analyze all barcodes

        # -------------------------------------------------------------------------
        # Determine properties of the requested fov ids
        # -------------------------------------------------------------------------
        if fovIDs == [] or fovIDs == "":
            fovIDs = self.fovIDs
        elif not np.all([f_i in self.fovIDs for f_i in fovIDs]):
            error('[Error]:invalidArguments - An invalid fov id has been requested')

        # -------------------------------------------------------------------------
        # Make directories if they do not exist
        # -------------------------------------------------------------------------
        localSummationPath = os.path.join(self.normalizedDataPath,self.summationPath)
        # Directory for barcodes by fov
        if not os.path.exists(localSummationPath):
            os.makedirs(localSummationPath,exist_ok=True)

        # -------------------------------------------------------------------------
        # Load features
        # -------------------------------------------------------------------------
        # Check for existence
        foundFeaturesPath = os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl')
        if not  os.path.exists(foundFeaturesPath):
            error('[Error]:missingData - Final segmentation boundaries could not be found.')

        # Load boundaries
        foundFeatures = pickle.load(open(foundFeaturesPath, 'rb'))

        # Copy (to preserve a direct link between parsed barcodes and the
        # boundaries
        if not os.path.exists(os.path.join(localSummationPath,'final_found_features.pkl')):
            shutil.copyfile(foundFeaturesPath, os.path.join(localSummationPath,'final_found_features.pkl'))


        # Make local copy of parameters for segmentation
        self.parameters["summation"]["dcIndsForSummation"] = []  ## manually set it to [], need to discuss
        parameters = self.parameters["segmentation"]

        # -------------------------------------------------------------------------
        # Determine which data channels and z slices to sum
        # -------------------------------------------------------------------------
        if len(self.parameters["summation"]["dcIndsForSummation"])==0:
            dataChannelInds = np.arange(self.numDataChannels)
        else:
            dataChannelInds = self.parameters["summation"]["dcIndsForSummation"]


        # Check validity of provided data channel inds
        if len(np.setdiff1d(dataChannelInds, np.arange(self.numDataChannels)))>0:
            error('[Error]:invalidArguments - The data channel indices provided do not match the known data channels')

        # -------------------------------------------------------------------------
        # Prepare mask to remove pixels outside of the bounding box
        # -------------------------------------------------------------------------
        # Create pixel coordinate system
        [X,Y] = np.meshgrid(np.arange(self.imageSize[0]), np.arange(self.imageSize[1]))

        # Create real-world scale coordinate system (but not yet
        # centered on fov position)
        x = self.pixelSize/1000*self.parameters["decoding"]["stageOrientation"][0]*(X-self.imageSize[0]/2) # x in microns
        y = self.pixelSize/1000*self.parameters["decoding"]["stageOrientation"][1]*(Y-self.imageSize[1]/2) # y in microns

        # Create ROI mask
        ROIMask = (x >= parameters["boundingBox"][0]) & \
                  (x <= (parameters["boundingBox"][0] + parameters["boundingBox"][2])) & \
                  (y >= parameters["boundingBox"][1]) & \
                  (y <= (parameters["boundingBox"][1] + parameters["boundingBox"][3]))

        # -------------------------------------------------------------------------
        # Loop over FOV and sum raw signals
        # -------------------------------------------------------------------------
        # Clear memory for accumulation registers
        totalSignal = np.zeros((len(dataChannelInds), np.max([fF.feature_id for fF in foundFeatures])+1))
        numberOfPixels = np.zeros((len(dataChannelInds), np.max([fF.feature_id for fF in foundFeatures])+1))

        # Loop over individual fov
        for f in fovIDs:
            # Determine local fovID
            localFovID = f

            # Create display strings
            if self.verbose:
                PageBreak()
                print('Started summation of raw signal in < fov',self.fov2str(localFovID),'> at',tic(1))
                print('...extracting boundaries for this fov')
                fovTimer = tic(0)
                localTimer = tic(0)

            # Identify the features the correspond to this fov
            isInFov = np.zeros((len(foundFeatures),))
            for F in range(len(foundFeatures)):
                isInFov[F] = foundFeatures[F].InFov([localFovID])

            # Crop these features
            localFeatures = [foundFeatures[e_i] for e_i,v_i in enumerate(isInFov) if v_i]

            # Create display strings
            if self.verbose:
                print('...searching ',len(localFeatures),'features')
                print('...completed in ',toc(localTimer),'s')
                localTimer = tic(0)

            # Define image data
            tiffName2Read = os.path.join(self.normalizedDataPath, self.warpedDataPath, 'fov_'+self.fov2str(localFovID)+'.tif')
            if not os.path.exists(tiffName2Read):
                error('[Error]:missingFile - The requsted tiff stack is not present.')

            # Loop over the data channels and load z stacks
            for c in range(len(dataChannelInds)):

                # Define frames to load
                allZInds = np.arange(self.numZPos)
                possibleFrames = dataChannelInds[c] * self.numZPos + allZInds

                # Find the z positions in the data organization file
                usedZPos = self.dataOrganization.zPos[dataChannelInds[c]]

                # Determine the frames to keep
                keptZInds = allZInds[np.isin(self.zPos, usedZPos)]
                framesToLoad = possibleFrames[np.isin(self.zPos, usedZPos)]

                # Display progress
                if self.verbose:
                    PageBreak(s="-",n=30)
                    print('...loading stack for data channel',dataChannelInds[c])
                    loadTimer = tic(0)


                # Allocate memory for the necessary frames
                localImageStack = np.zeros((self.imageSize[0],self.imageSize[1],len(framesToLoad)), 'uint16')

                # Load these frames
                for z in range(len(framesToLoad)):
                    # Load frames for watershed and seed generation
                    localImageStack[:,:,z] = imread(tiffName2Read, key=framesToLoad[z])

                # Display progress
                if self.verbose:
                    print('...completed in',toc(loadTimer),'s')
                    print('...parsing',len(localFeatures),'features')
                    localTimer = tic(0)
                    totalTimer = tic(0)

                # Loop over local features and add
                for F in range(len(localFeatures)):
                    # Create feature mask
                    mask = localFeatures[F].GeneratePixelMask(localFovID, keptZInds)

                    # Crop to ROI
                    mask = np.logical_and(mask,np.repeat(ROIMask[:, :, np.newaxis], mask.shape[2], axis=2))

                    # Add signal
                    totalSignal[c,localFeatures[F].feature_id] = totalSignal[c,localFeatures[F].feature_id] + np.sum(localImageStack[mask])

                    # Add number of pixels
                    numberOfPixels[c,localFeatures[F].feature_id] = numberOfPixels[c,localFeatures[F].feature_id] + np.sum(mask)

                    # Display progress
                    if self.verbose and not np.mod(F+1, 100):
                        print('...completed ',F+1,'features in ',toc(localTimer),'s')
                        localTimer = tic(0)

                # Display progress
                if self.verbose:
                    print('...completed all features in ',toc(totalTimer),'s')


            # Save total signal
            totalSignalFilePath = os.path.join(localSummationPath,'total_signal_fov_' +self.fov2str(localFovID)+'.pkl')
            pickle.dump(totalSignal,open(totalSignalFilePath,"wb"),pickle.HIGHEST_PROTOCOL)

            # Save number of pixels
            numberOfPixelsFilePath = os.path.join(localSummationPath,'total_pixels_fov_'+self.fov2str(localFovID)+'.pkl')
            pickle.dump(numberOfPixels, open(numberOfPixelsFilePath, "wb"), pickle.HIGHEST_PROTOCOL)

            # Flush display buffer
            if self.verbose:
                print('...saved', totalSignalFilePath)
                print('...saved', numberOfPixelsFilePath)
                print('...completed fov', self.fov2str(localFovID),'in',toc(fovTimer),'s')

    # -------------------------------------------------------------------------
    # Combine raw summation of signal
    # -------------------------------------------------------------------------
    def CombineRawSum(self):
        # Combine the sum of raw signals within features for each fov
        #
        # self.CombineRawSum()

        # Display progress
        if self.verbose:
            PageBreak()
            print('Combining raw signal summation for boundaries in individual fov')

        # Create paths to combined final values
        combinedSignalFilePath = os.path.join(self.normalizedDataPath,self.summationPath,'total_signal.pkl')
        combinedPixelsFilePath = os.path.join(self.normalizedDataPath,self.summationPath,'total_pixels.pkl')

        # Handle existing analysis with overwrite off
        if os.path.exists(combinedSignalFilePath) and os.path.exists(combinedPixelsFilePath):
            print('...found existing combined sum. Skipping analysis.')
            return

        # Map the existing signal and pixel files
        foundSignalFiles,_ = BuildFileStructure(os.path.join(self.normalizedDataPath,self.summationPath),
                                              fileExt='pkl',regExp='total_signal_fov_(?P<fov>[0-9]+)',
                                              fieldNames=['fov'],fieldConv = [int])
        foundPixelsFiles,_ = BuildFileStructure(os.path.join(self.normalizedDataPath,self.summationPath),
                                              fileExt='pkl',regExp='total_pixels_fov_(?P<fov>[0-9]+)',
                                              fieldNames=['fov'],fieldConv=[int])
        foundSignalFiles_df,foundPixelsFiles_df = pd.DataFrame(foundSignalFiles),pd.DataFrame(foundPixelsFiles)
        # Display progress
        if self.verbose:
            print('...found ',len(foundSignalFiles),'signal files')
            print('...found ',len(foundPixelsFiles),'pixel files')

        # Check for missing fov
        missingFovIDs = np.union1d(np.setdiff1d(self.fovIDs, [fSF["fov"] for fSF in foundSignalFiles]),
                                   np.setdiff1d(self.fovIDs, [fPF["fov"] for fPF in foundPixelsFiles]))
        if len(missingFovIDs) > 0:
            print('...the following fov are missing some files associated with raw signal summation')
            print('... ',missingFovIDs)
            error('[Error]:missingData - Combination cannot proceed until all fov have been summed')

        # Load and sum signals and piuxels
        if self.verbose:
            print('...loading total sum and pixel numbers')
            localTimer = tic(0)

        totalSignal = []
        totalPixels = []
        for f in range(self.numFov):
            # Handle the case of the first file load
            if f==0:
                totalSignal = pickle.load(open(foundSignalFiles_df.loc[foundSignalFiles_df.fov == self.fovIDs[f],"filePath"].values[0],"rb"))
                totalPixels = pickle.load(open(foundPixelsFiles_df.loc[foundPixelsFiles_df.fov == self.fovIDs[f],"filePath"].values[0],"rb"))
            else:
                totalSignal = totalSignal + pickle.load(open(foundSignalFiles_df.loc[foundSignalFiles_df.fov == self.fovIDs[f],"filePath"].values[0],"rb"))
                totalPixels = totalPixels + pickle.load(open(foundPixelsFiles_df.loc[foundPixelsFiles_df.fov == self.fovIDs[f],"filePath"].values[0],"rb"))


        # Save combined signals as csv
        combinedSignalFilePath = os.path.join(self.normalizedDataPath,self.summationPath,'total_signal.csv')
        combinedPixelsFilePath = os.path.join(self.normalizedDataPath,self.summationPath,'total_pixels.csv')
        np.savetxt(combinedSignalFilePath, totalSignal,delimiter=",")
        np.savetxt(combinedPixelsFilePath, totalPixels,delimiter=",")

        # Define the feature name file
        featuresNameFilePath = os.path.join(self.normalizedDataPath,self.summationPath,'featureNames.csv')

        # Load features
        foundFeatures = self.GetFoundFeatures()

        # Generate the contents of the feature names file
        featureUIDs = [fF.uID for fF in foundFeatures] # Feature ids

        # Open and write file
        fid = open(featuresNameFilePath, "w")
        fid.write("\n".join(featureUIDs) + "\n")
        fid.close()

        # Reconstitute and return the channel names
        channelNamesFilePath = os.path.join(self.normalizedDataPath, self.summationPath, 'channelNames.csv')
        if len(self.parameters["summation"]["dcIndsForSummation"])==0:
            indsForSum = np.arange(self.numDataChannels)
        else:
            indsForSum = self.parameters["summation"]["dcIndsForSummation"]
        indsForSum = np.arange(self.numDataChannels) ##manually setting, neec to discuss
        channelNames = self.dataOrganization.bitName[indsForSum].values

        # Open and write file
        fid = open(channelNamesFilePath, 'w')
        fid.write("\n".join(channelNames) + "\n")
        fid.close()

        # Delete intermediate files
        if not self.keepInterFiles:
            for f in foundSignalFiles:
                os.remove(f.filePath)
            for f in foundPixelsFiles:
                os.remove(f.filePath)

        # Display progress
        if self.verbose:
            print('...completed in ',toc(localTimer),'s')

#     # -------------------------------------------------------------------------
#     # Get counts per feature
#     # -------------------------------------------------------------------------
#     function [exactCountsIn, correctedCountsIn, exactCountsOut, correctedCountsOut, ...
#             geneNames, featureUIDs] = GetCountsPerFeature(obj)
#         # Return the counts per feature
#         #
#         # [exactCountsIn, correctedCountsIn, exactCountsOut, correctedCountsOut, geneNames, featureUIDs] = self.GetCountsPerFeature()
#
#         # Check to see if the feature counts have been created
#         if ~self.CheckStatus(self.fovIDs(1), 'n')
#             error('[Error]:incompleteAnalysis - The feature counts have not yet been computed.')
#         end
#
#         # Display progress
#         if self.verbose:
#             PageBreak()
#             print('Loading feature counts')
#             localTimer = tic
#         end
#
#         # Load the counts
#         reportsPath = [self.normalizedDataPath 'reports' filesep]
#
#         exactCountsIn = csvread([reportsPath 'countsPerCellExactIn.csv')
#         correctedCountsIn = csvread([reportsPath 'countsPerCellCorrectedIn.csv')
#         exactCountsOut = csvread([reportsPath 'countsPerCellExactOut.csv')
#         correctedCountsOut = csvread([reportsPath 'countsPerCellCorrectedOut.csv')
#
#         # Load the feature unique ids
#         fid = fopen([reportsPath 'featureNames.csv'], 'r')
#         line = fgetl(fid)
#         fclose(fid)
#         featureUIDs = strsplit(line, ',')
#
#         # Load the gene names
#         fid = fopen([reportsPath 'geneNames.csv'], 'r')
#         line = fgetl(fid)
#         fclose(fid)
#         geneNames = strsplit(line, ',')
#
#         # Display progress
#         if self.verbose:
#             print('...completed in ',toc(localTimer),'s')
#         end
#
#     end
#
#     # -------------------------------------------------------------------------
#     # Get fov absolute coordinates
#     # -------------------------------------------------------------------------
#     function coordinates = GetFOVCoordinates(obj, fovID)
#         # Return the absolute coordinates of the specified fov
#         #
#         # coordinates = self.GetFOVCoordinates(fovID)
#         # coordinates = [xstart xend ystart yend] where xstart is the
#         # center of the first pixel and xend is the center of the final
#         # pixel in x
#
#         # Check the input
#         if nargin < 1
#             error('[Error]:invalidArguments - A fov id must be specified')
#         end
#         if ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArguments - The specified fov id is not valid')
#         end
#
#         # Compute the coordinates
#         fovPos = self.fovPos(self.fovIDs == fovID,:)
#
#         coordinates(1:2) = [-1 1]*self.imageSize(1)*(self.pixelSize/1000)/2 + fovPos(1)
#         coordinates(3:4) = [-1 1]*self.imageSize(2)*(self.pixelSize/1000)/2 + fovPos(2)
#
#     end


    # -------------------------------------------------------------------------
    # Return a low resolution mosaic of the sample
    # -------------------------------------------------------------------------
    def  GetMosaic(self, sliceID, framesToLoad=None):
        # Return the low resolution mosaic and the absolute coordinate
        # system
        #
        # mosaicImageStack = self.GetMosaic(sliceID)

        # Check to see if the mosaics have been created
        if not os.path.exists(os.path.join(self.normalizedDataPath,self.mosaicPath)):
            error('[Error]:incompleteAnalysis - The mosaics have not yet been created.')

        # Check to see if the requested sliceID is valid
        if sliceID > len(self.sliceIDs) or sliceID < 1:
            error('[Error]:invalidArgument - The provided sliceID is not valid.')

        # Determine mosaic stack info
        stackInfo = imread(os.path.join(self.normalizedDataPath,self.mosaicPath,'slice_'+str(sliceID)+'.tif'))

        # Determine the frames to load if not provided
        if framesToLoad==None:
            framesToLoad =np.range(len(stackInfo))

        # Check the provided frames to load
        if np.any(~np.isin(framesToLoad, np.range(len(stackInfo)))):
            error('[Error]:invalidArguments - The provided frame ids to load are not valid.')

        # Allocate memory
        mosaicImageStack = np.zeros((stackInfo[0].Width, stackInfo[0].Height, len(framesToLoad)),dtype='uint16')

        # Load the requested frames
        for s in range(len(framesToLoad)):
            mosaicImageStack[:,:,s] =  imread(os.path.join(self.normalizedDataPath,self.mosaicPath,'slice_'+str(sliceID)+'.tif'), key=framesToLoad[s])

        # Load the coordinate system
        coordinates = pickle.load(open(os.path.join(self.normalizedDataPath,self.mosaicPath,'coordinates_slice_'+str(sliceID)+'.pkl')))

        return [mosaicImageStack, coordinates]

    # -------------------------------------------------------------------------
    # Return the combined signal data
    # -------------------------------------------------------------------------
    def GetSummedSignal(self):
        # Return the output of the signal summation process
        # [normalizedSignal, sumSignal, sumPixels, channelNames, featureUIDs] = self.GetSummedSignal

        # Check to see if the sum signals have been combined
        if not os.path.exists(os.path.join(self.normalizedDataPath,self.summationPath)):
            error('[Error]:missingData - The summation has not yet been completed.')

        # Define paths to the different objects
        combinedSignalFilePath = os.path.join(self.normalizedDataPath,self.summationPath, 'total_signal.csv')
        combinedPixelsFilePath = os.path.join(self.normalizedDataPath, self.summationPath,'total_pixels.csv')

        # Load the sum objects
        sumSignal = np.loadtxt(combinedSignalFilePath,delimiter=",")
        sumPixels = np.loadtxt(combinedPixelsFilePath,delimiter=",")

        # Compute the normalized sum
        # Normalized the sum signal by the feature volume
        normalizedSignal = sumSignal/sumPixels

        # Load the featureUIDs
        featureUIDs = []
        with open(os.path.join(self.normalizedDataPath, self.summationPath, 'featureNames.csv'), 'r') as fid:
            for line in fid:
                featureUIDs.append(line.strip())


        # Load the channel names
        channelNames = []
        with open(os.path.join(self.normalizedDataPath, self.summationPath, 'channelNames.csv'), 'r') as fid:
            for line in fid:
                channelNames.append(line.strip())

        return [normalizedSignal, sumSignal, sumPixels, channelNames, featureUIDs]

    # -------------------------------------------------------------------------
    # Parse barcodes into final feature boundaries for individual fov
    # -------------------------------------------------------------------------
    def ParseFOV(self,fovIDs):
        # Parse decoded barcodes into feature boundaries within individual fov
        #
        # self.ParseFOV(fovIDs) # Parse the specified fov barcodes
        # self.ParseFOV([]) # Parse all barcodes

        # -------------------------------------------------------------------------
        # Determine properties of the requested fov ids
        # -------------------------------------------------------------------------
        if len(fovIDs) == 0:
            fovIDs = self.fovIDs
        elif not np.all([f_i in self.fovIDs for f_i in fovIDs]):
            error('[Error]:invalidArguments - An invalid FOV id has been requested')

        # -------------------------------------------------------------------------
        # Make directories if they do not exist
        # -------------------------------------------------------------------------
        barcodeByFovPath = os.path.join(self.normalizedDataPath,self.barcodePath,'barcode_fov')
        # Check for existing barcodes
        if not os.path.exists(barcodeByFovPath):
            error('[Error]:missingData - No barcodes could be found.')

        # Make directory for the parsed barcodes
        parsedBarcodePath = os.path.join(self.normalizedDataPath,self.barcodePath,'parsed_fov')
        # Directory for barcodes by fov
        if not os.path.exists(parsedBarcodePath):
            os.makedirs(parsedBarcodePath,exist_ok=True)

        # -------------------------------------------------------------------------
        # Load feature boundaries
        # -------------------------------------------------------------------------
        # Check for existence
        foundFeaturesPath = os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl')
        if not os.path.exists(foundFeaturesPath):
            error('[Error]:missingData - Final segmentation boundaries could not be found.')

        # Load boundaries
        foundFeatures = pickle.load(open(foundFeaturesPath,"rb"))

        # Copy (to preserve a direct link between parsed barcodes and the boundaries))
        shutil.copyfile(foundFeaturesPath, os.path.join(parsedBarcodePath,'final_found_features.pkl'))

        # -------------------------------------------------------------------------
        # Prepare copies of variables for parsing loop
        # -------------------------------------------------------------------------
        # Make local copy of parameters for segmentation
        parameters = self.parameters["segmentation"]

        # -------------------------------------------------------------------------
        # Parse via individual fov
        # -------------------------------------------------------------------------

        # Loop over individual fov
        for f in fovIDs:
            # Determine local fovID
            localFovID = f

            # Create display strings
            if self.verbose:
                PageBreak()
                print('Started parsing of barcodes in fov',localFovID,'at',tic(1))
                print('...extracting boundaries for this fov')
                fovTimer = tic(0)
                localTimer = tic(0)

            # Find all possible adjacent fov (to cut down on the
            # boundaries that need to be parsed)
            nnIDX,_ = knnsearch2d(self.fovPos, self.fovPos[self.fovIDs==localFovID,:], k=9) # 8 neighbors + 1 duplicate of self
            fovIDsToSearch = self.fovIDs[nnIDX]

            # Reduce the finalBoundaries
            goodFeatureInds = np.zeros((len(foundFeatures),),dtype=bool)
            for F in range(len(foundFeatures)):
                goodFeatureInds[F] = foundFeatures[F].InFov(fovIDsToSearch)

            localFeatures = [foundFeatures[i] for i,x in enumerate(goodFeatureInds) if x]

            # Create display strings
            if self.verbose:
                print('...searching ',len(localFeatures),'boundaries')
                print('...completed in ',toc(localTimer),'s')
                print('Loading barcodes')
                localTimer = tic(0)

            # Check for existing or corrupt barcode list
            barcodeFilePath = os.path.join(barcodeByFovPath,'fov_'+self.fov2str(localFovID)+'_blist.pkl')
            if not os.path.exists(barcodeFilePath):
                error('[Error]:missingData - Could not find specified barcode file.')
            try:
                # Load bList
                bList = pickle.load(open(barcodeFilePath,"rb"))
                isCorrupt = False
            except:
                isCorrupt = True

            if isCorrupt:
                error('[Error]:missingData - The requested barcode file appears to be corrupt.')



            # Display progress
            if self.verbose:
                print('...loaded ',len(bList),'barcodes')
                print('...completed load in ',toc(localTimer),'s')
                print('...trimming barcodes to bounding box')
                localTimer = tic(0)

            # Extract positions of loaded barcodes
            pos = np.vstack(bList["abs_position"])

            # Define bounding box (map to absolute position in the
            # sample)
            boundingBox = parameters["boundingBox"].copy() # Bounding box used to cut segmentation boundaries
            boundingBox[0:2] = boundingBox[0:2] + self.fovPos[self.fovIDs==localFovID,:] # Map to real world coordinates

            # Define positions within this bounding box
            indsToKeep = (pos[:,0] >= boundingBox[0]) & (pos[:,0] <= (boundingBox[0] + boundingBox[2])) & \
                         (pos[:,1] >= boundingBox[1]) & (pos[:,1] <= (boundingBox[1] + boundingBox[3]))

            # Slice list to remove barcodes not inside the bounding box
            bList = bList[indsToKeep]

            # Display progress
            if self.verbose:
                print('...cut ',sum(~indsToKeep),'barcodes outside of segmentation bounding box')
                print('...completed in ',toc(localTimer),'s')
                print('Parsing barcodes')
                localTimer = tic(0)


            # Add additional fields to barcode list
            bList["feature_id"] = -1            # The id of the feature to which the barcode was assigned
            bList["feature_dist"] = 0   # The distance to the nearest feature edge
            bList["in_feature"] = 1     # A boolean indicating whether or not (true/false) the RNA falls within the feature to which it was associated

            # Prepare a list of all barcode positions
            barcodePos = np.vstack(bList["abs_position"].values)

            #Discretize z positions for all barcodes
            zInds = np.digitize(barcodePos[:,2], list(self.zPos)+[np.inf])

            # Handle the case that there are no barcodes or no features
            bList = bList.to_dict(orient='records')
            if len(bList)>0 and len(localFeatures)>0:
                #Loop over z indices
                for z in range(self.numZPos):
                    #Compile a composite list of boundaries in this z
                    #plane
                    combinedBoundaries = np.zeros((0,2))
                    isDilatedBoundary = np.zeros((0,),dtype=bool)
                    localFeatureIDs = np.zeros((0,))
                    for F in range(len(localFeatures)):
                        # Concatenate boundaries and dilated boundaries
                        combinedBoundaries = np.concatenate((combinedBoundaries,
                                                       localFeatures[F].abs_boundaries[z],
                                                       localFeatures[F].DilateBoundary(z, self.pixelSize/1000*parameters["dilationSize"])
                                                       ),0)

                        # Concatenate indices for features
                        localFeatureIDs = np.concatenate((localFeatureIDs,
                                                    F*np.ones((len(localFeatures[F].abs_boundaries[z]),)),
                                                    F*np.ones((len(localFeatures[F].abs_boundaries[z]),))
                                                    ),0)

                        # Concatenate flags for inside or outside feature
                        isDilatedBoundary = np.concatenate((isDilatedBoundary,
                                                      np.zeros((len(localFeatures[F].abs_boundaries[z]),),dtype=bool),
                                                      np.ones((len(localFeatures[F].abs_boundaries[z]),),dtype=bool)
                                                      ),0)


                    #Handle the case of no boundaries in the z-plane
                    if len(combinedBoundaries)==0:
                        continue

                    #Find the barcode indices in this zPlane
                    localBarcodeIndex = np.nonzero(zInds == (z+1))[0]

                    #Find the nearest neighbor point in the boundaries and
                    #the distance
                    [nnIDX, D] = knnsearch_ckdtree(combinedBoundaries, barcodePos[localBarcodeIndex,0:2])

                    #Loop through barcodes to assign found values and
                    #determine if they are in the appropriate features
                    for b in range(len(localBarcodeIndex)):
                        bList[localBarcodeIndex[b]]["feature_id"]= localFeatures[int(localFeatureIDs[nnIDX[b]])].feature_id
                        bList[localBarcodeIndex[b]]["feature_dist"] = D[b]
                        bList[localBarcodeIndex[b]]["in_feature"] = bool(1-isDilatedBoundary[nnIDX[b]])

            else:
                # Display progress
                if self.verbose:
                   print('...handling case of no features or no barcodes...')


            # Display progress
            if self.verbose:
                print('...completed in ',toc(localTimer),'s')
                print('...writing barcode list')
                localTimer = tic(0)


            # Save the barcode list
            # Write binary file for all measured barcodes
            barcodeFile = os.path.join(parsedBarcodePath,'fov_'+self.fov2str(localFovID)+'_blist.pkl')
            WriteBinaryFile(barcodeFile, pd.DataFrame(bList))

            # Display progress
            if self.verbose:
                print('...wrote', barcodeFile)
                print('...complete in',toc(localTimer),'s')
                print('Completed analysis of fov',self.fov2str(localFovID),'at ',tic(2),'in ',toc(fovTimer),'s')


#     # -------------------------------------------------------------------------
#     # Find individual molecules within a fov
#     # -------------------------------------------------------------------------
#     function FindMoleculesFOV(obj,fovIDs)
#         # Find individual molecules (via image regional max), assign to
#         # individual cells
#         #
#         # self.FindMoleculesFOV(fovIDs) # Find molecules, assign to cells,
#         # for all z stacks in all data channels in the specified fov
#         # self.FindMoleculesFOV([]) # Run this analysis for all fov
#
#         # -------------------------------------------------------------------------
#         # Determine properties of the requested fov ids
#         # -------------------------------------------------------------------------
#         if isempty(fovIDs)
#             fovIDs = self.fovIDs
#         elseif ~all(ismember(fovIDs, self.fovIDs))
#             error('[Error]:invalidArguments - An invalid fov id has been requested')
#         end
#
#         # -------------------------------------------------------------------------
#         # Make directories if they do not exist
#         # -------------------------------------------------------------------------
#         moleculeByFovPath = [self.normalizedDataPath filesep 'smFISH' filesep]
#         # Check for existing barcodes
#         if ~exist(moleculeByFovPath, 'dir')
#             os.makedirs(moleculeByFovPath)
#         end
#
#         # -------------------------------------------------------------------------
#         # Prepare copies of variables for parsing loop
#         # -------------------------------------------------------------------------
#         # Make local copy of parameters for segmentation
#         parameters = self.parameters.molecules
#         boundingBox = self.parameters.segmentation.boundingBox
#
#         # Handle the case that no information was provided on the
#         # dataChannels and zstacks
#         if isempty(parameters.molDataChannels)
#             dataChannelDescription = cell(0,2)
#             for i=1:size(self.dataOrganization,1)
#                 dataChannelDescription(end+1,:) = {self.dataOrganization(i).bitName, 1:self.numZPos}
#             end
#         else
#             dataChannelDescription = parameters.molDataChannels
#         end
#
#         # -------------------------------------------------------------------------
#         # Load feature boundaries
#         # -------------------------------------------------------------------------
#         foundFeatures = self.GetFoundFeatures()
#
#         # -------------------------------------------------------------------------
#         # Find spots within individual fov
#         # -------------------------------------------------------------------------
#         spmd (self.numPar)
#             # Loop over individual fov
#             for f=labindex:numlabs:len(fovIDs)
#                 # Determine local fovID
#                 localFovID = fovIDs(f)
#
#                 # Loop over the requested data Channels to compile a list
#                 # of molecules in this fov for all data channels
#                 mList = []
#
#                 for D = 1:size(dataChannelDescription,1)
#
#                     # Create display strings
#                     if self.verbose:
#
#                         PageBreak()
#                        print('Identifying molecules in channel ' dataChannelDescription{D,1} ' for fov ',localFovID,'at ' datestr(now)]
#                         fovTimer = tic
#                         localTimer = tic
#                         # Flush buffer
#                         if numlabs==1
#                             print(char(displayStrings))
#
#                         end
#                     end
#
#                     # Extract the requested frame ids
#                     frameIDs = dataChannelDescription{D,2}
#
#                     # Loop over the requested image frames
#                     for z=1:len(frameIDs)
#
#                         # Load the requested image frame
#                         frame = self.GetImage(localFovID, dataChannelDescription{D,1}, frameIDs(z))
#
#                         # Low pass filter the image to remove background
#                         frame = frame - imgaussfilt(frame, parameters.molLowPassfilterSize)
#
#                         # Compute the region max
#                         rMax = imregionalmax(frame)
#
#                         # Remove values that are too dim
#                         rMax(frame < parameters.molIntensityThreshold) = 0
#
#                         # Combine touching pixels
#                         props = regionprops(rMax, 'Centroid')
#
#                         # Create a combined list of centroids
#                         centroids = cat(1, props.Centroid)
#
#                         # Add the z index
#                         centroids(:,3) = frameIDs(z)
#
#                         # Convert to real coordinates
#                         abs_position = self.Pixel2Abs(centroids, localFovID)
#
#                         # Drop molecules that fall outside of the
#                         # segmentation bounding box
#                         shiftedPos = abs_position(:,1:2) - repmat(self.fovPos(self.fovIDs == localFovID,:), [size(abs_position,1) 1])
#
#                         moleculesToKeep = shiftedPos(:,1) >= boundingBox(1) & shiftedPos(:,2) >= boundingBox(2) & ...
#                             shiftedPos(:,1) <= (boundingBox(1)+boundingBox(3)) & ...
#                             shiftedPos(:,2) <= (boundingBox(2)+boundingBox(4))
#
#                         # Cut the molecules
#                         centroids = centroids(moleculesToKeep,:)
#                         abs_position = abs_position(moleculesToKeep,:)
#
#                         # Calculate the brightness: First identify pixels
#                         # to integrate
#                         xBounds = repmat(round(centroids(:,1)), [1 2]) + repmat([-parameters.molNumPixelSum parameters.molNumPixelSum], ...
#                             [size(centroids,1) 1])
#                         yBounds = repmat(round(centroids(:,2)), [1 2]) + repmat([-parameters.molNumPixelSum parameters.molNumPixelSum], ...
#                             [size(centroids,1) 1])
#
#                         brightness = zeros(1, size(xBounds,1))
#                         for M=1:size(xBounds,1)
#                             brightness(M) = sum(sum(frame(yBounds(M,1):yBounds(M,2), xBounds(M,1):xBounds(M,2))))
#                         end
#
#                         # Create molecule list structure
#                         newMolecules = struct('centroid', num2cell(centroids,2), 'abs_position', num2cell(abs_position,2), ...
#                             'brightness', num2cell(brightness)', ...
#                             'channel', find(strcmp({self.dataOrganization.bitName},  dataChannelDescription{D,1})))
#
#                         # Append these molecules to the list of all
#                         # molecules
#                         mList = cat(1, mList, newMolecules)
#
#                     end
#
#                 end
#
#                 # Create display strings
#                 if self.verbose:
#                    print('...completed in ',toc(localTimer),'s')
#                     localTimer = tic
#                     # Flush buffer
#                     if numlabs==1
#                         print(char(displayStrings))
#
#                     end
#                 end
#
#                 # Create display strings
#                 if self.verbose:
#
#                     PageBreak()
#                    print('Parsing molecules in fov ',localFovID,'at ' datestr(now)]
#                     fovTimer = tic
#                     localTimer = tic
#                     # Flush buffer
#                     if numlabs==1
#                         print(char(displayStrings))
#
#                     end
#                 end
#
#
#                 # Find all possible adjacent fov (to cut down on the
#                 # boundaries that need to be parsed)
#                 nnIDX = knnsearch(self.fovPos, self.fovPos(self.fovIDs==localFovID,:), ...
#                     'K', 9) # 8 neighbors + 1 duplicate of self
#                 fovIDsToSearch = self.fovIDs(nnIDX)
#
#                 # Reduce the finalBoundaries
#                 goodFeatureInds = false(1, len(foundFeatures))
#                 for F=1:len(foundFeatures)
#                     goodFeatureInds(F) = foundFeatures(F).InFov(fovIDsToSearch)
#                 end
#                 localFeatures = foundFeatures(goodFeatureInds)
#
#
#                 # Handle the case that no molecules were found
#                 if isempty(mList)
#                     # Create display strings
#                     if self.verbose:
#                        print('...searching ',len(localFeatures),'boundaries']
#                        print('...complete in ',toc(localTimer),'s')
#                        print('No molecules found. Skipping the parsing.')
#                        print('Completed analysis of fov ' self.fov2str(localFovID,'at ' datestr(now,'in ',toc(fovTimer),'s')
#                         print(char(displayStrings)) # Final buffer flush
#                     end
#
#                     # Continue
#                     continue
#                 end
#
#
#                 # Create display strings
#                 if self.verbose:
#                    print('...searching ',len(localFeatures),'boundaries']
#                    print('...completed in ',toc(localTimer),'s')
#                    print('...parsing ',len(mList),'molecules']
#                     localTimer = tic
#                     # Flush buffer
#                     if numlabs==1
#                         print(char(displayStrings))
#
#                     end
#                 end
#
#                 # Add additional fields to barcode list
#                 [mList(:).feature_id] = deal(int32(-1))            # The id of the feature to which the barcode was assigned
#                 [mList(:).feature_dist] = deal(zeros(1, 'single')) # The distance to the nearest feature edge
#                 [mList(:).in_feature] = deal(zeros(1, 'uint8'))    # A boolean indicating whether or not (true/false) the RNA falls within the feature to which it was associated
#
#                 # Prepare a list of all barcode positions
#                 barcodePos = cat(1, mList.abs_position)
#
#                 #Discretize z positions for all barcodes
#                 zInds = discretize(barcodePos(:,3), [self.zPos Inf])
#
#                 #Loop over z indices
#                 for z=1:self.numZPos
#                     #Compile a composite list of boundaries in this z
#                     #plane
#                     combinedBoundaries = zeros(0,2)
#                     isDilatedBoundary = zeros(0,1)
#                     localFeatureIDs = zeros(0,1)
#                     for F=1:len(localFeatures)
#                         # Concatenate boundaries and dilated boundaries
#                         combinedBoundaries = cat(1, combinedBoundaries, ...
#                             localFeatures(F).abs_boundaries{z},...
#                             localFeatures(F).DilateBoundary(z, self.pixelSize/1000*self.parameters.segmentation.dilationSize))
#
#                         # Concatenate indices for features
#                         localFeatureIDs = cat(1, localFeatureIDs, ...
#                             F*ones(len(localFeatures(F).abs_boundaries{z}),1), ...
#                             F*ones(len(localFeatures(F).abs_boundaries{z}),1))
#
#                         # Concatenate flags for inside or outside feature
#                         isDilatedBoundary = cat(1, isDilatedBoundary, ...
#                             false(len(localFeatures(F).abs_boundaries{z}),1), ...
#                             true(len(localFeatures(F).abs_boundaries{z}),1))
#
#                     end
#
#                     #Find the barcode indices in this zPlane
#                     localBarcodeIndex = find(zInds == z)
#
#                     #Find the nearest neighbor point in the boundaries and
#                     #the distance
#                     [nnIDX, D] = knnsearch(combinedBoundaries, barcodePos(localBarcodeIndex,1:2))
#
#                     #Loop through barcodes to assign found values and
#                     #determine if they are in the appropriate features
#                     for b=1:len(localBarcodeIndex)
#                         mList(localBarcodeIndex(b)).feature_id = int32(localFeatures(localFeatureIDs(nnIDX(b))).feature_id)
#                         mList(localBarcodeIndex(b)).feature_dist = single(D(b))
#                         mList(localBarcodeIndex(b)).in_feature = uint8(~isDilatedBoundary(nnIDX(b)))
#                     end
#                 end
#
#                 # Display progress
#                 if self.verbose:
#                    print('...completed in ',toc(localTimer),'s')
#                    print('...writing molecule list']
#
#                     localTimer = tic
#                     # Flush buffer
#                     if numlabs==1
#                         print(char(displayStrings))
#
#                     end
#                 end
#
#                 # Save the barcode list
#                 # Write binary file for all measured barcodes
#                 moleculeFile = [moleculeByFovPath 'fov_' self.fov2str(localFovID) '_mlist.bin']
#                 WriteBinaryFile(moleculeFile, mList)
#
#                 # Display progress
#                 if self.verbose:
#                    print('...wrote ' moleculeFile]
#                    print('...complete in ',toc(localTimer),'s')
#                    print('Completed analysis of fov ' self.fov2str(localFovID,'at ' datestr(now,'in ',toc(fovTimer),'s')
#                     print(char(displayStrings)) # Final buffer flush
#                 end
#             end # End loop over fov
#         end # End spmd loops
#
#     end # End function

    # -------------------------------------------------------------------------
    # Combine features to create a final set of features
    # -------------------------------------------------------------------------
    def CombineFeatures(self):
        # Combine found features determined in individual fov into a
        # single set of non-overlapping, real-world feature boundaries
        #
        # CombineFeatures()

        # Display progress
        if self.verbose:
            PageBreak()
            print('Combining found features from individual fov')
            totalTimer = tic(0)

        # Create paths to intermediate save states
        allFoundFeaturesPath = os.path.join(self.normalizedDataPath,self.segmentationPath,'all_found_features.pkl')
        finalFoundFeaturesPath =  os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl')

        # Check to see if the combination has already been done and delete
        # if overwrite requested
        if self.overwrite:
            if os.path.exists(finalFoundFeaturesPath):
                if self.verbose: print('...ovewriting final found features list')
                os.remove(finalFoundFeaturesPath)

        # Handle existing analysis with overwrite off
        if os.path.exists(finalFoundFeaturesPath):
            print('...found existing final found features. Skipping analysis.')
            return

        # Compile partial boundaries by fov
        if not os.path.exists(allFoundFeaturesPath):
            # Confirm that all fov have been properly segmented
            foundFiles,_ = BuildFileStructure(os.path.join(self.normalizedDataPath,self.segmentationPath),
                                            fileExt='pkl',regExp = 'found_features_fov_(?<fov>[0-9]+)',
                                            fieldNames=['fov'],fieldConv=[int])
            foundFiles_df = pd.DataFrame(foundFiles)
            # Display progress
            if self.verbose:
                print('...found',len(foundFiles),'fov')

            # Check for missing fov
            missingFovIDs = np.setdiff1d(self.fovIDs, [f_i["fov"] for f_i in foundFiles])
            if len(missingFovIDs) > 0:
                print('...the following fov are missing boundaries')
                print('... ',missingFovIDs)
                error('[Error]:missingData - Combination cannot proceed until all fov have been segmented')

            # Load and concatenate found features
            if self.verbose:
                print('...loading found features')
                localTimer = tic(0)

            allFoundFeatures = []
            for f in range(len(foundFiles)):
                FoundFeature_f = pickle.load(open(foundFiles_df.loc[foundFiles_df["fov"] == self.fovIDs[f],"filePath"].values[0],"rb"))
                allFoundFeatures += FoundFeature_f

            # Display progress
            if self.verbose:
                print('...completed in',toc(localTimer),'s')
                saveTimer = tic(0)

            # Save combined 'raw' features
            with open(allFoundFeaturesPath, "wb") as fout:
                pickle.dump(allFoundFeatures, fout, pickle.HIGHEST_PROTOCOL)

            # Delete tempory files for individual fov
            if not self.keepInterFiles:
                for f in range(len(foundFiles)):
                    os.remove(foundFiles[f]["filePath"])

            # Display progress
            if self.verbose:
                print('...saved combined file in',toc(saveTimer),'s')

        else: # Load existing raw feature boundaries
            print('...found existing raw boundaries...loading')
            loadTimer = tic(0)
            allFoundFeatures = pickle.load(open(allFoundFeaturesPath, 'rb'))
            print('...loaded',len(allFoundFeatures),'features in',toc(loadTimer),'s')

        # -------------------------------------------------------------------------
        # Find features that have broken boundaries
        # -------------------------------------------------------------------------
        # Separate features that are closed from those that are
        # broken
        finalFeatures = [f_i for f_i in allFoundFeatures if not f_i.is_broken]
        featuresToFix = [f_i for f_i in allFoundFeatures if f_i.is_broken]

        # Display progress
        if self.verbose:
            print('...found',len(allFoundFeatures),'total features')
            print('...found',len(finalFeatures),'complete features')
            print('...found',len(featuresToFix),'features with one break')
            joinTimer = tic(0)
            fovTimer = tic(0)

        # Get the primary FOV ids for each cell: these will also serve
        # as a flag to mark features that have been already used to fix
        # other features
        primaryFovIDs=[]
        for F in featuresToFix:
            primaryFovIDs.append(F.fovID[0])

        for f in range(self.numFov):
            # Find local fov id
            localFovID = self.fovIDs[f]

            # Find the features associated with this id
            localFeatures = [featuresToFix[e_i] for e_i,x in enumerate(primaryFovIDs) if x==localFovID]

            # Mark these features as used to prevent future assignment
            primaryFovIDs=[np.nan if x==localFovID else x for x in primaryFovIDs]

            # Find the surrounding fov ids
            [nnIDX, D] = knnsearch2d(self.fovPos, [self.fovPos[f,:]], k=5) # 4 neighbors + 1 duplicate of self
            nnIDX = nnIDX[D>0] # Remove self reference

            # Find all possible pairs
            possibleFeaturesForJoin = [featuresToFix[e_i] for e_i,x in enumerate(np.isin(primaryFovIDs, self.fovIDs[nnIDX])) if x]

            # Loop over all features to fix
            for F in localFeatures:
                # Compute the join penalty for all possible pairs
                penaltyValues = np.empty((len(possibleFeaturesForJoin)+1,))
                penaltyValues[:] = np.nan
                for J in range(len(possibleFeaturesForJoin)):
                    penaltyValues[J]= F.CalculateJoinPenalty(possibleFeaturesForJoin[J])

                # Compute the self penalty
                penaltyValues[-1] = F.CalculateJoinPenalty()

                # Select the minimum
                minPenalty = np.min(penaltyValues)
                minInd = np.argmin(penaltyValues)
                # Determine if the minPenalty is below a given threshold
                if minPenalty < self.parameters["segmentation"]["maxEdgeDistance"]:
                    # Determine whether this is a self join or not
                    if minInd < len(possibleFeaturesForJoin):
                        # Join the feature
                        joinedFeature = F.JoinFeature(possibleFeaturesForJoin[minInd])

                        # Mark the joined feature as used
                        for fix_i in range(len(featuresToFix)):
                            if featuresToFix[fix_i].uID==possibleFeaturesForJoin[minInd].uID:
                                primaryFovIDs[fix_i] = np.nan

                    else:
                        # Join the feature with itself
                        joinedFeature = F.JoinFeature()


                    # Add the joined feature to the final features list
                    finalFeatures.append(joinedFeature)

            if self.verbose:
                print('...completed fov ',localFovID,'in ',toc(fovTimer),'s')
                fovTimer = tic(0)

        # Display progress
        if self.verbose:
            # Calculate statistics on joining process
            selfJoined = np.sum([f_i.num_joined_features==1 for f_i in finalFeatures])
            pairJoined = np.sum([f_i.num_joined_features==2 for f_i in finalFeatures])
            unPaired = len(primaryFovIDs) - selfJoined - pairJoined

            print('...completed all fov in ',toc(joinTimer),'s')
            print('...self-joined ',selfJoined,'features')
            print('...pair-joined ',pairJoined,'features')
            print('...remaining unpaired features ',unPaired,'features')
            localTimer = tic(0)
            print('...saving final features')

        # Assign a unique feature id (for fast indexing) to each final
        # feature
        for i in range(len(finalFeatures)):
            finalFeatures[i].AssignFeatureID(i)

        # Save final joined features
        with open(finalFoundFeaturesPath, "wb") as fout:
            pickle.dump(finalFeatures, fout, pickle.HIGHEST_PROTOCOL)

        # Export a flat file of unique feature ids
        fid = open(os.path.join(self.normalizedDataPath,self.segmentationPath,'featureNames.csv'), 'w')
        featureUIDs = [fF_i.uID for fF_i in finalFeatures]
        fid.write("\n".join(featureUIDs)+"\n")
        fid.close()

        # Display progress
        if self.verbose:
            print('...completed save in ',toc(localTimer),'s')
            print('...completed feature combination in ',toc(totalTimer),'s')


    # -------------------------------------------------------------------------
    # Return found features
    # -------------------------------------------------------------------------
    def GetFoundFeatures(self):
        # Return found features if they have already been parsed
        #
        # foundFeatures = GetFoundFeatures()

        # Determine if features have been found
        if not os.path.exists(os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl')):
            error('[Warning]:incompleteAnalysis - Feature segmentation is not complete.')
        else:
            # Load and return the found features
            foundFeatures = pickle.load(
                open(os.path.join(self.normalizedDataPath,self.segmentationPath,'final_found_features.pkl'),"rb")
            )
        return foundFeatures

#     # -------------------------------------------------------------------------
#     # Get image frame
#     # -------------------------------------------------------------------------
#     function [imageFrame, coordinates] = GetImage(obj, fovID, dataChannel, zStack)
#         # Return the requested frame
#         #
#         # imageFrame = self.GetImage(fovID, dataChannel, zStack)
#
#         # Check input
#         if nargin < 3
#             error('[Error]:invalidArgument - A fov id, a data channel, and z stack must be provided')
#         end
#         if ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArgument - The provided fov id is invalid')
#         end
#         if ~ischar(dataChannel) || ~ismember(dataChannel, {self.dataOrganization.bitName})
#             error('[Error]:invalidArgument - The provided data channel name is invalid')
#         end
#         if zStack < 1 || zStack > self.numZPos
#             error('[Error]:invalidArgument - The provided zStack index is invalid')
#         end
#
#         # Calculate the position of the requested frame
#         channelID = find(strcmp({self.dataOrganization.bitName}, dataChannel))
#         frameID = zStack + self.numZPos*(channelID-1)
#
#         # Load the frame
#         imageFrame = imread([self.normalizedDataPath self.warpedDataPath 'fov_' self.fov2str(fovID) '.tif'], frameID)
#
#         # Define the coordinates
#         lowerLeft = self.Pixel2Abs([1 1], fovID)
#         upperRight = self.Pixel2Abs(self.imageSize, fovID)
#         coordinates.xLimits = [lowerLeft(1) upperRight(1)]
#         coordinates.yLimits = [lowerLeft(2) upperRight(2)]
#
#     end
#
#     # -------------------------------------------------------------------------
#     # Get processed image frame
#     # -------------------------------------------------------------------------
#     function [imageFrame, coordinates] = GetProcessedImage(obj, fovID, dataChannel, zStack)
#         # Return the requested frame
#         #
#         # imageFrame = self.GetProcessedImage(fovID, dataChannel, zStack)
#
#         # Check input
#         if nargin < 3
#             error('[Error]:invalidArgument - A fov id, a data channel, and z stack must be provided')
#         end
#         if ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArgument - The provided fov id is invalid')
#         end
#         if ~ischar(dataChannel) || ~ismember(dataChannel, {self.dataOrganization.bitName})
#             error('[Error]:invalidArgument - The provided data channel name is invalid')
#         end
#         if zStack < 1 || zStack > self.numZPos
#             error('[Error]:invalidArgument - The provided zStack index is invalid')
#         end
#
#         # Calculate the position of the requested frame
#         channelID = find(strcmp({self.dataOrganization.bitName}, dataChannel))
#         frameID = zStack + self.numZPos*(channelID-1)
#
#         # Load the frame
#         imageFrame = imread([self.normalizedDataPath self.processedDataPath 'fov_' self.fov2str(fovID) '.tif'], frameID)
#
#         # Define the coordinates
#         lowerLeft = self.Pixel2Abs([1 1], fovID)
#         upperRight = self.Pixel2Abs(self.imageSize, fovID)
#         coordinates.xLimits = [lowerLeft(1) upperRight(1)]
#         coordinates.yLimits = [lowerLeft(2) upperRight(2)]
#
#     end
#
#     # -------------------------------------------------------------------------
#     # Get Decoded Image
#     # -------------------------------------------------------------------------
#     function [barcodeStack, magnitudeStack, coordinates] = GetDecodedImage(obj, fovID)
#         # Return the decoded image stack
#         #
#         # [barcodeStack, magnitudeStack, coordinates] = self.GetDecodedImage(fovID)
#
#         # Check input
#         if nargin < 2
#             error('[Error]:invalidArgument - A fov id must be provided')
#         end
#         if ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArgument - The provided fov id is invalid')
#         end
#
#         # Check to see if the decoded image is available
#         decodedImagePath = [self.normalizedDataPath self.barcodePath filesep 'decoded_images' filesep 'fov_' self.fov2str(fovID) '.tif']
#         if ~exist(decodedImagePath)
#             error('[Error]:invalidData - The requested image does not exist.')
#         end
#
#         # Load the barcode stack
#         for z=1:self.numZPos
#             # Dynamically find size and allocate memory
#             if z==1
#                 temp = imread(decodedImagePath, z)
#                 barcodeStack = zeros([size(temp) self.numZPos])
#                 magnitudeStack = barcodeStack
#                 barcodeStack(:,:,z) = temp
#                 temp = []
#             else
#                 barcodeStack(:,:,z) = imread(decodedImagePath, z)
#             end
#         end
#
#         # Load the magnitude stack
#         for z=1:self.numZPos
#             magnitudeStack(:,:,z) = imread(decodedImagePath, z + self.numZPos)
#         end
#
#         # Define the coordinates
#         # Handle the crop (somewhat poorly coded)
#         xC = 1:self.imageSize(1)
#         yC = 1:self.imageSize(2)
#
#         xC = xC((self.parameters.decoding.crop+1):(end-self.parameters.decoding.crop))
#         yC = yC((self.parameters.decoding.crop+1):(end-self.parameters.decoding.crop))
#
#         lowerLeft = self.Pixel2Abs([xC(1) yC(1)], fovID)
#         upperRight = self.Pixel2Abs([xC(end) yC(end)], fovID)
#         coordinates.xLimits = [lowerLeft(1) upperRight(1)]
#         coordinates.yLimits = [lowerLeft(2) upperRight(2)]
#
#     end
#
#
#     # -------------------------------------------------------------------------
#     # Return barcode lists
#     # -------------------------------------------------------------------------
#     function bList = GetBarcodeList(obj, fovID)
#         # Read barcode lists as requested
#         #
#         # bList = GetBarcodeList(fovID)
#
#         # Confirm required input
#         if ~exist('fovID', 'var') || ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArguments - A valid fovID must be provided.')
#         end
#
#         # Check to confirm that the requested fovID has been decoded
#         if ~self.CheckStatus(fovID, 'd')
#             error('[Error]:missingData - The requested fov has not yet been decoded.')
#         end
#
#         # Load and return barcode list
#         barcodeByFovPath = [self.normalizedDataPath self.barcodePath filesep 'barcode_fov' filesep]
#         barcodeFile = [barcodeByFovPath 'fov_' self.fov2str(fovID) '_blist.bin']
#
#         bList = ReadBinaryFile(barcodeFile)
#
#     end
#
#     # -------------------------------------------------------------------------
#     # Return molecule list
#     # -------------------------------------------------------------------------
#     function mList = GetMoleculeList(obj, fovID)
#         # Read a list of molecules
#         #
#         # mList = GetMoleculeList(fovID)
#
#         # Confirm required input
#         if ~exist('fovID', 'var') || ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArguments - A valid fovID must be provided.')
#         end
#
#         moleculeByFovPath = [self.normalizedDataPath filesep 'smFISH' filesep]
#         moleculeFile = [moleculeByFovPath 'fov_' self.fov2str(fovID) '_mlist.bin']
#
#         if ~exist(moleculeFile)
#             error('[Error]:missingFile - The requested file does not exist')
#         end
#
#         mList = ReadBinaryFile(moleculeFile)
#
#     end
#
#
#     # -------------------------------------------------------------------------
#     # Return barcode lists
#     # -------------------------------------------------------------------------
#     function bList = GetParsedBarcodeList(obj, fovID)
#         # Read parsed barcode lists as requested
#         #
#         # bList = GetParsedBarcodeList(fovID)
#
#         # Confirm required input
#         if ~exist('fovID', 'var') || ~ismember(fovID, self.fovIDs)
#             error('[Error]:invalidArguments - A valid fovID must be provided.')
#         end
#
#         # Check to confirm that the requested fovID has been decoded
#         if ~self.CheckStatus(fovID, 'p')
#             error('[Error]:missingData - The requested fov has not yet been parsed.')
#         end
#
#         # Load and return barcode list
#         barcodeByFovPath = [self.normalizedDataPath self.barcodePath filesep 'parsed_fov' filesep]
#         barcodeFile = [barcodeByFovPath 'fov_' self.fov2str(fovID) '_blist.bin']
#
#         bList = ReadBinaryFile(barcodeFile)
#
#     end

    # -------------------------------------------------------------------------
    # Warp images
    # -------------------------------------------------------------------------
    def WarpFOV(self, fovIDs):
        # Warp individual fov and produce a warped data stack
        # WarpFOV([]) # Warp all fov
        # WarpFOV(fovIDs) # Warp the fov that match the specified fovids

        # -------------------------------------------------------------------------
        # Determine properties of the requested fov ids
        # -------------------------------------------------------------------------
        if fovIDs==[] or fovIDs=="":
            fovIDs = self.fovIDs
        elif not np.all([f_i in self.fovIDs for f_i in fovIDs]):
            error('[Error]:invalidArguments - An invalid fov id has been requested')

        # -------------------------------------------------------------------------
        # Run processing on individual in parallel (if requested)
        # -------------------------------------------------------------------------
        # Loop over requested fov
        for f in fovIDs:
            # Determine local fov id
            localFovID = f

            # Create display strings
            if self.verbose:
                PageBreak()
                print('Started warping fov',self.fov2str(localFovID))

            # Check to see if the warped tiff file exists and if it is
            # complete
            warpedTiffFileName = 'fov_'+self.fov2str(localFovID)+'.tif' # Define name for warped tiff file
            tiffFileName = os.path.join(self.normalizedDataPath,self.warpedDataPath,warpedTiffFileName)

            if os.path.exists(tiffFileName):
                if self.verbose:
                    print('Found existing warped data file for',self.fov2str(localFovID),'.')

                if self.overwrite: # Repeat analysis
                    os.remove(tiffFileName)
                    if self.verbose:
                        print('Overwriting...')
                else:
                    if self.CheckTiffStack(tiffFileName, self.numDataChannels * self.numZPos): # Check for corrupt file
                        os.remove(tiffFileName)
                        if self.verbose:
                            print('File appears to be corrupt. Deleting.')
                    else:# Skip analysis file exists
                        if self.verbose:
                            print('File appears to be complete. Skipping analysis.')
                        continue

            # Create display strings
            if self.verbose:
                print('Fitting fiducials...')
                localTimer = tic()

            # Switch based on the method for finding fiducials
            if self.parameters["warp"]["fiducialFitMethod"] == "daoSTORM":
                # Handle the daoSTORM case (the only supported option this point)
                # Fit Fiducials for each data channel`
                for c in range(self.numDataChannels):
                    # Skip fiducial fitting if no fiducial round is provided
                    if ("fiducialImageType" not in self.dataOrganization) or \
                            pd.isna(self.dataOrganization["fiducialImageType"][c]) :
                        continue

                    # Find fiducial information
                    if ("fiducialCameraID" in self.dataOrganization) and (not pd.isna(self.dataOrganization["fiducialCameraID"][c])):
                        fileInd = np.nonzero((self.rawDataFiles["imageType"].values==self.dataOrganization["fiducialImageType"][c]) &
                                             (self.rawDataFiles["imagingRound"].values == self.dataOrganization["fiducialImagingRound"][c]) &
                                             (self.rawDataFiles["cameraID"].values == self.dataOrganization["fiducialCameraID"][c]) &
                                             (self.rawDataFiles["fov"].values == localFovID))[0]

                    else:
                        fileInd = np.nonzero((self.rawDataFiles["imageType"].values==self.dataOrganization["fiducialImageType"][c]) &
                                             (self.rawDataFiles["imagingRound"].values == self.dataOrganization["fiducialImagingRound"][c]) &
                                             (self.rawDataFiles["fov"] == localFovID))[0]

                    # Check for consistency
                    if len(fileInd) != 1:
                        print(fileInd)
                        print(c)
                        print(self.dataOrganization["fiducialImageType"][c])
                        print(self.dataOrganization["fiducialImagingRound"][c])
                        if ("fiducialCameraID" in self.dataOrganization):
                            print(self.dataOrganization["fiducialCameraID"][c])
                        print(localFovID)
                        error('[Error]:invalidFileInformation - Either a file is missing or there are multiple files that match an expected pattern.')

                    # Determine file name
                    localFileName = self.rawDataFiles["name"][fileInd[0]]
                    baseName = os.path.splitext(basename(localFileName))[0]

                    # Create daostorm parameters file
                    WriteDaoSTORMParameters(os.path.join(self.normalizedDataPath,self.fiducialDataPath,baseName+'_dao.xml'),
                                            start_frame = self.dataOrganization["fiducialFrame"][c]-1,
                                            max_frame = self.dataOrganization["fiducialFrame"][c],
                                            x_stop = self.imageSize[0], y_stop = self.imageSize[1],
                                            iterations = 1, pixel_size = self.pixelSize,
                                            threshold = self.parameters["warp"]["daoThreshold"],
                                            sigma = self.parameters["warp"]["sigmaInit"],
                                            baseline = self.parameters["warp"]["daoBaseline"])

                    # Run daoSTORM
                    daoSTORM(os.path.join(self.rawDataPath,localFileName),os.path.join(self.normalizedDataPath,self.fiducialDataPath,baseName+'_dao.xml'),
                             overwrite = True,  # Overwrite all files (overwrite protection is provided above)
                             numParallel = 1,
                             savePath = os.path.join(self.normalizedDataPath,self.fiducialDataPath),
                             verbose = False, waitTime = 1, outputInMatlab = True,hideterminal = True)

                # Create display strings
                if self.verbose:
                    print('... completed in',toc(localTimer),'s')
                    PageBreak()
                    print('Constructing affine transforms...')
                    localTimer = tic()  # Restart timer

                # Build affine transformation
                localAffine = []
                localResiduals = []
                for c in range(self.numDataChannels):
                    # Skip warping if no fiducial information is
                    # provided
                    if ("fiducialImageType" not in self.dataOrganization) or pd.isna(self.dataOrganization["fiducialImageType"][c]):
                        localAffine.append(AffineTransform())
                        localResiduals.append(np.zeros((4,)))
                        continue

                    # Identify name of mList
                    if 'fiducialCameraID' in self.dataOrganization and self.dataOrganization["fiducialCameraID"][c]:
                        fileInd = (self.rawDataFiles["imageType"].values == self.dataOrganization["fiducialImageType"][c]) & \
                                  (self.rawDataFiles["imagingRound"].values == self.dataOrganization["fiducialImagingRound"][c]) & \
                                  (self.rawDataFiles["cameraID"].values == self.dataOrganization["fiducialCameraID"][c]) & \
                                  (self.rawDataFiles["fov"].values == localFovID)
                    else:
                        fileInd = (self.rawDataFiles["imageType"].values == self.dataOrganization["fiducialImageType"][c]) & \
                                   (self.rawDataFiles["imagingRound"].values == self.dataOrganization["fiducialImagingRound"][c]) & \
                                   (self.rawDataFiles["fov"].values == localFovID)

                    # Determine file name
                    localFileName = self.rawDataFiles["name"][fileInd].values[0]
                    baseName = os.path.splitext(basename(localFileName))[0]

                    # Fiducial file name
                    fiducialFileName = os.path.join(self.normalizedDataPath,self.fiducialDataPath,baseName+'_mList.bin')

                    # Check to see if it exists
                    if not os.path.exists(fiducialFileName):
                        print('Could not find',fiducialFileName)
                        error('[Error]:missingFile - Could not find a fiducial file:'+fiducialFileName)

                    # Load molecule list
                    movList = ReadMasterMoleculeList(fiducialFileName, fieldsToLoad=['x','y','xc','yc','frame'],verbose=False)
                    movList = pd.DataFrame(movList)
                    # Define reference list as the list from the
                    # first data channel (i.e. bit 1)
                    if c==0:
                        refList = movList.copy()

                    # Generate transform
                    c_tmplA,_,c_tmplR,_,_ = MLists2Transform(refList, movList, ignoreFrames = True,
                                                            controlPointMethod='kNNDistanceHistogram',
                                                            histogramEdges = self.parameters["warp"]["controlPointOffsetRange"],
                                                            numNN = self.parameters["warp"]["numNN"],
                                                            pairDistTolerance = self.parameters["warp"]["pairDistanceTolerance"],
                                                            debugFolder = os.path.join(self.normalizedDataPath, self.fiducialDataPath))
                    localAffine.append(c_tmplA)
                    localResiduals.append(c_tmplR)

                    # Generate additional input
                    if self.verbose:
                        print('...completed affine transform for data channel',c+1)
                        print('...out of',len(refList["x"]),'reference molecules and',len(movList["x"]),'moving molecules matched',len(localResiduals[c]))
                # End construction of affine transformations

                # Create display strings
                if self.verbose:
                    print('... completed in',toc(localTimer),'s')

                # Save data associated with transform for this fov
                with open(os.path.join(self.normalizedDataPath,self.fiducialDataPath,'fov_'+self.fov2str(localFovID)+'_affine.pkl'), "wb") as fout:
                    pickle.dump(localAffine, fout, pickle.HIGHEST_PROTOCOL)
                print("Saving", os.path.join(self.normalizedDataPath,self.fiducialDataPath,'fov_'+self.fov2str(localFovID)+'_affine.pkl'))

                with open(os.path.join(self.normalizedDataPath, self.fiducialDataPath,'fov_' + self.fov2str(localFovID) + '_residuals.pkl'), "wb") as fout:
                    pickle.dump(localResiduals, fout, pickle.HIGHEST_PROTOCOL)
                print("Saving", os.path.join(self.normalizedDataPath, self.fiducialDataPath, 'fov_' + self.fov2str(localFovID) + '_residuals.pkl'))

            else: # Handle the case that an undefined fiducial/warp approach was defined
                error('[Error]:invalidArguments - The provided fiducial fit method is not supported')
            # End switch statement on fiducial fitting/affine transformation construction approach

            # Create display strings
            if self.verbose:
                PageBreak()
                print('Writing warped tiff file...')
                localTimer = tic()  # Restart timer


            # Export a warped tiff stack of fiducials only: Run this
            # first as a complete warped tiff stack is the measure of a
            # complete warp process.
            if self.parameters["warp"]["exportWarpedBeads"]:
                if self.verbose:
                    print('Creating warped fidicual stack...')

                # Define the folder for defaults if it does not exist
                fiducialsPath = os.path.join(self.normalizedDataPath,self.warpedDataPath,'warped_fiducials')
                if not os.path.exists(fiducialsPath):
                    os.makedirs(fiducialsPath,exist_ok=True)

                # Define the bead tiff file name
                fiducialsTiffFileName = os.path.join(fiducialsPath,'fov_'+self.fov2str(localFovID)+'.tif')

                # Overwrite any previous analysis
                if os.path.exists(fiducialsTiffFileName):
                    if self.verbose:
                        print('...found existing warped fidicual stack.... removing...')
                    os.remove(fiducialsTiffFileName)


                # Create tiff file
                fiducialsTiffFile =  TiffWriter(fiducialsTiffFileName, bigtiff=True,append=True)

                # Create tiff tags
                fiducialsTiffTagStruct = {}
                # fiducialsTiffTagStruct["shape"] = self.imageSize
                fiducialsTiffTagStruct["photometric"] = TIFF.PHOTOMETRIC.MINISBLACK
                fiducialsTiffTagStruct["bitspersample"] = 16
                # fiducialsTiffTagStruct["samplesperpixel"] = 1
                fiducialsTiffTagStruct["rowsperstrip"] = 16
                fiducialsTiffTagStruct["planarconfig"] = TIFF.PLANARCONFIG.CONTIG
                fiducialsTiffTagStruct["software"] = 'Ruifeng'
                fiducialsTiffTagStruct["description"] = 'images='+str(self.numDataChannels)+'\n' + \
                                                         'channels=1\n' + \
                                                         'slices=1\n' + \
                                                         'frames='+str(self.numDataChannels)+'\n' + \
                                                         'hyperstack=true\n' + \
                                                         'loop=false\n'

                # Loop over all data channels
                try:
                    for c in range(self.numDataChannels):
                        # Skip fiducial fitting if no fiducial round is
                        # provided
                        if ("fiducialImageType" not in self.dataOrganization) or \
                                (pd.isna(self.dataOrganization["fiducialImageType"][c])):
                            print('...no fiducials for channel', c)
                            continue

                        # Find fiducial information
                        if 'fiducialCameraID' in self.dataOrganization and self.dataOrganization["fiducialCameraID"][c]:
                            fileInd = np.nonzero(
                                (self.rawDataFiles["imageType"].values == self.dataOrganization["fiducialImageType"][c]) &
                                (self.rawDataFiles["imagingRound"].values == self.dataOrganization["fiducialImagingRound"][c]) &
                                (self.rawDataFiles["cameraID"].values == self.dataOrganization["fiducialCameraID"][c]) &
                                (self.rawDataFiles["fov"].values == localFovID))[0]
                        else:
                            fileInd = np.nonzero(
                                (self.rawDataFiles["imageType"].values == self.dataOrganization["fiducialImageType"][c]) &
                                (self.rawDataFiles["imagingRound"].values == self.dataOrganization["fiducialImagingRound"][c]) &
                                (self.rawDataFiles["fov"].values == localFovID))[0]

                        # Determine file name
                        localFileName = self.rawDataFiles["name"][fileInd].values[0]

                        # Switch based on the imageExt to load fiducial
                        # image
                        localImage = []
                        if self.imageExt == 'dax':
                            localImage = ReadDax(os.path.join(self.rawDataPath,localFileName),
                                                 startFrame=self.dataOrganization["fiducialFrame"][c],
                                                 endFrame = self.dataOrganization["fiducialFrame"][c],
                                                 verbose=False)
                        elif self.imageExt in ['tiff','tif']:
                            # Read tiff frame
                            localImage = imread(os.path.join(self.rawDataPath,localFileName))[self.dataOrganization["fiducialFrame"][c]-1]
                        else:
                            error('[Error]:unsupportedFile - The specified image file ext is not yet supported.')

                        # Check for image load problems
                        if len(localImage)==0:
                            error('[Error]:invalidData - The requested frame does not exist')

                        # Warp the image
                        # ra = imref2d(localImage.shape()) # Create a crop
                        localImage = warp(localImage, np.linalg.inv(localAffine[c]),preserve_range=True)
                        ## same processing method in CV2
                        # localImage = cv2.warpAffine(localImage, localAffine[c][:2, :],
                        #                             localImage.shape, flags=cv2.INTER_LINEAR,
                        #                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        localImage = np.rint(localImage).astype('uint16')
                        # Write tiff file
                        fiducialsTiffFile.write(localImage, **fiducialsTiffTagStruct)

                        if self.verbose:
                            print('...completed channel',c,'of',self.numDataChannels)
                except Exception as e:
                    # Close tiff file and delete existing file
                    fiducialsTiffFile.close()
                    os.remove(fiducialsTiffFileName)

                    # Update log
                    print('Encountered error: ',e)
                    raise(e)
                fiducialsTiffFile.close()
                if self.verbose:
                    print('...completed warped fiducial stack')
             # End if statement for creating a warped fiducial stack

            # Create tiff file
            tiffFile = TiffWriter(tiffFileName, bigtiff=True,append=True)

            # Create tiff tags
            tiffTagStruct = {}
            # tiffTagStruct["shape"] = self.imageSize
            tiffTagStruct["photometric"] = TIFF.PHOTOMETRIC.MINISBLACK
            tiffTagStruct["bitspersample"] = 16
            # tiffTagStruct["samplesperpixel"] = 1
            tiffTagStruct["rowsperstrip"] = 16
            tiffTagStruct["planarconfig"] = TIFF.PLANARCONFIG.CONTIG
            tiffTagStruct["software"] = 'Ruifeng'
            tiffTagStruct["description"] = 'images=' + str(self.numDataChannels) + '\n' + \
                                            'channels=1\n' + \
                                            'slices=1\n' + \
                                            'frames=' + str(self.numDataChannels) + '\n' + \
                                            'hyperstack=true\n' + \
                                            'loop=false\n'

            if self.verbose:
                PageBreak()
                print('Creating the warped tiff stack...')

            # Gracefully handle external kill commands
            try:
                # Load, warp, and write tiff files
                for c in range(self.numDataChannels):
                    # Identify file properties based on file organization
                    imageType = self.dataOrganization["imageType"][c]
                    imagingRound = self.dataOrganization["imagingRound"][c]
                    color = self.dataOrganization["color"][c]
                    frames = self.dataOrganization["frame"][c]
                    if not isinstance(frames,list):
                        frames = [frames]
                    localZPos = self.dataOrganization["zPos"][c]
                    if ('imagingCameraID' not in self.dataOrganization) or \
                            pd.isna(self.dataOrganization['imagingCameraID'][c]):
                        localCameraID = ''
                    else:
                        localCameraID = self.dataOrganization["imagingCameraID"][c]

                    # Identify dax file
                    localFile = self.rawDataFiles[(self.rawDataFiles["imageType"].values==imageType) &
                                                  (self.rawDataFiles["fov"].values == localFovID) &
                                                  (self.rawDataFiles["cameraID"].values==localCameraID) &
                                                  (self.rawDataFiles["imagingRound"].values == imagingRound)]

                    # Identify the affine transform to be used to adjust this color channel
                    localIDs = np.nonzero(self.parameters["warp"]["colorTransforms"]["color"] == str(self.dataOrganization["color"][c]))[0] # Find all transforms that match the specified color (they will be applied in order)

                    # Extract this transform
                    if len(localIDs)>0:
                        localTransforms = self.parameters["warp"]["colorTransforms"][localIDs]
                    else:
                        localTransforms = []

                    # Loop over z and load dax
                    for z in range(len(self.zPos)):
                        # Find frame
                        frameInd = np.nonzero(localZPos == self.zPos[z])[0]

                        # If the position doesn't exist, assume it is 1 (this
                        # will replicate frames for data sets that do not have
                        # the all z stacks
                        if len(frameInd) == 0:
                            frameInd = 0
                        else:
                            frameInd = frameInd[0]

                        # Switch based on the imageExt
                        if self.imageExt == 'dax':
                                # Read single data frame
                                localImage = ReadDax(localFile["filePath"],
                                                     startFrame = frames[frameInd],
                                                     endFrame = frames[frameInd],
                                                     verbose=False)
                        elif self.imageExt in ['tiff','tif']:
                            # Read tiff frame
                            localImage = imread(localFile["filePath"].values[0])[frames[frameInd]-1]
                        else:
                            error('[Error]:unsupportedFile - The specified image file ext is not yet supported.')

                        # Check for image load problems
                        if localImage.size==0:
                            error('[Error]:invalidData - The requested frame does not exist')

                        # Apply a color based transform if needed
                        # ra = imref2d(size(localImage)); # Create a crop
                        if len(localTransforms) > 0:
                            # Loop over chromatic transforms and apply
                            # in order
                            for T in localTransforms:
                                if T.type in ['similarity', 'Similarity', 'Euclidean', 'euclidean']:
                                    # Create the affine transform based
                                    # on the provided data
                                    trans = AffineTransform(T.transform)
                                    # Apply the transform
                                    localImage = warp(localImage, np.linalg.inv(trans), preserve_range=True)
                                    # localImage =  cv2.warpAffine(localImage, trans[:2,:],
                                    #                 localImage.shape,flags=cv2.INTER_LINEAR,
                                    #                 borderMode=cv2.BORDER_CONSTANT,  borderValue=0)

                                elif T.type in ['Invert', 'invert']:
                                    # Invert the axes of the image
                                    # as needed
                                    # Flip x axis
                                    if T.transform[0]:
                                        localImage = localImage[:, ::-1]
                                    # Flip y axis
                                    if T.transform[1]:
                                        localImage = localImage[::-1, :]
                                else:
                                    error('[Error]:unsupportedFile - The transform type provided is not supported')

                        # Remove translation of the image due to stage
                        # alignment using the fiducial bead affine transform
                        localImage = warp(localImage, np.linalg.inv(localAffine[c]), preserve_range=True)
                        # localImage = cv2.warpAffine(localImage, localAffine[c][:2,:],
                        #                             localImage.shape,flags=cv2.INTER_LINEAR,
                        #                             borderMode=cv2.BORDER_CONSTANT,  borderValue=0)

                        # Reorient image to a fixed orientation
                        # Image width represents X,
                        # Image height represents Y,
                        # X/Y increase with increasing pixel id
                        cameraOrientation = self.parameters["warp"]["cameraOrientation"]
                        if cameraOrientation[2]:
                            localImage = np.transpose(localImage) # Exchange X/Y
                        if cameraOrientation[0]:
                            localImage = np.flip(localImage, 1) # Invert the X axis (rows)
                        if cameraOrientation[1]:
                            localImage = np.flip(localImage, 0) # Invert the Y axis (columns)

                        localImage = np.rint(localImage).astype('uint16')
                        # Write tiff file
                        tiffFile.write(localImage,**tiffTagStruct)

                        if self.verbose:
                            print('...completed channel',c,'of',self.numDataChannels)
                    # End loop over z
                # End loop over channels
            except Exception as e:
                # Close tiff file and delete existing file
                tiffFile.close()
                os.remove(tiffFileName)

                # Update log
                print('Encountered error:',e)
                # Rethrow err
                raise(e)

            # Close tiff file
            tiffFile.close()

            # Create display strings
            if self.verbose:
                print('... completed in',toc(localTimer),'s')
                print('Completed warping of fov',self.fov2str(localFovID),'at',tic(1))
        # End loop over fovIds

    # -------------------------------------------------------------------------
    # Preprocess Images
    # -------------------------------------------------------------------------
    def PreprocessFOV(self, fovIDs):
        # Preprocess individual FOV and produce a processed tiff stack
        # PreprocessFOV([]) # Deconvolve all fov
        # PreprocessFOV(fovIDs) # Deconvolve the fov that match the specified fovids

        # -------------------------------------------------------------------------
        # Determine properties of the requested fov ids
        # -------------------------------------------------------------------------
        if fovIDs==[] or fovIDs=="":
            fovIDs = self.fovIDs
        elif not np.all([f_i in self.fovIDs for f_i in fovIDs]):
            error('[Error]:invalidArguments - An invalid fov id has been requested')

        # -------------------------------------------------------------------------
        # Check to see if the pixel histogram field has already been
        # populated--no need to repeat analysis if it does
        # -------------------------------------------------------------------------
        if len(self.pixelHistograms)==0 or self.overwrite:

            # Create Generic Tiff Tag
            tiffTagStruct = {}
            # tiffTagStruct["shape"] = self.imageSize
            tiffTagStruct["photometric"] = TIFF.PHOTOMETRIC.MINISBLACK
            tiffTagStruct["bitspersample"] = 16
            tiffTagStruct["planarconfig"] = TIFF.PLANARCONFIG.CONTIG
            tiffTagStruct["software"] = 'Ruifeng'
            tiffTagStruct["description"] = 'images=' + str(self.numBits*self.numZPos) + '\n' + \
                                            'channels=1\n' + \
                                            'slices='+str(self.numZPos)+'\n' + \
                                            'frames=' + str(self.numBits) + '\n' + \
                                            'hyperstack=true\n' + \
                                            'loop=false\n'

            # Loop over fov
            for f in fovIDs:
                # Determine local fovID
                localFovID = f

                # Create display strings
                if self.verbose:
                    PageBreak()
                    print('Started preprocessing of fov',self.fov2str(localFovID))
                    localTimer = tic()

                # Create tiff to read file name
                tiffName2Read = os.path.join(self.normalizedDataPath,self.warpedDataPath,'fov_'+self.fov2str(localFovID)+'.tif')
                if not os.path.exists(tiffName2Read):
                    error('[Error]:missingFile - The requsted tiff stack is not present.')

                # Create tiff to write file name
                tiffName2Write = os.path.join(self.normalizedDataPath,self.processedDataPath,'fov_'+self.fov2str(localFovID)+'.tif')

                # Create pixel histogram file and initialize
                pixelHistogramFile = os.path.join(self.normalizedDataPath,self.processedDataPath, 'pixel_histograms','pixel_data_fov_'+self.fov2str(localFovID)+'.pkl')

                if not os.path.exists(os.path.join(self.normalizedDataPath,self.processedDataPath,'pixel_histograms') ):
                    os.makedirs(os.path.join(self.normalizedDataPath,self.processedDataPath,'pixel_histograms'))

                pixelHistogram = np.zeros((self.numBits, 65535)) # Allocate memory and initialize to zero

                if os.path.exists(tiffName2Write):
                    if self.verbose:
                        print('Found existing warped data file for',self.fov2str(localFovID),'.')

                    if self.overwrite: # Repeat analysis
                        if self.verbose: print('[...Overwriting, deleted',tiffName2Write,"]")
                        os.remove(tiffName2Write)
                    else:
                        # Check for corrupt tiff file
                        if self.CheckTiffStack(tiffName2Write, self.numBits*self.numZPos):
                            if self.verbose: print('File appears to be corrupt. Deleting.')
                            os.remove(tiffName2Write)
                        elif  not os.path.exists(pixelHistogramFile): # Check for missing pixel histogram
                            if self.verbose: print('Pixel histogram is missing. Repeating analysis')
                            os.remove(tiffName2Write)
                        else: # Handle the case that the file is not corrupt and the pixel histogram file exists, i.e. the analysis is complete
                            if self.verbose:
                                print('File appears to be complete. Skipping analysis.')
                            continue

                # Create writing tiff
                tiffToWrite =TiffWriter(tiffName2Write, bigtiff=True,append=True)

                # Gracefully handle interrupts
                try:
                    # Switch to run the specific algorithm
                    processingMethod = self.parameters["preprocess"]["preprocessingMethod"]

                    if processingMethod == 'highPassDecon':
                        # Loop over imaging round
                        for b in range(self.numBits):
                            if self.verbose:
                                print('...Processing Bit',b+1,"/",self.numBits,end=" ")
                                bit_st = tic(0)
                            # Loop over z
                            for z in range(self.numZPos):
                                # Set directory and load
                                tiffToRead = imread(tiffName2Read, key = b*self.numZPos + z)
                                localFrame = tiffToRead.astype(np.float)

                                # High pass filter (and threshold on zero values b/c uint16)
                                # highPassSigma = self.parameters["preprocess"]["highPassKernelSize"]
                                # highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
                                # gaussfilt_img = cv2.GaussianBlur(localFrame, (highPassFilterSize, highPassFilterSize), highPassSigma, borderType=cv2.BORDER_REPLICATE)
                                gaussfilt_img = gaussian(localFrame, sigma=self.parameters["preprocess"]["highPassKernelSize"],preserve_range=True,truncate=2)
                                localFrame1 = localFrame - gaussfilt_img
                                localFrame1[localFrame1 < 0] = 0
                                localFrame1 = np.rint(localFrame1)
                                # Deconvolve
                                if self.parameters["preprocess"]["numIterDecon"] > 0:
                                    # localFrame2_1 = deconvolve_lucyrichardson(localFrame1,highPassFilterSize, highPassSigma,self.parameters["preprocess"]["numIterDecon"])
                                    localFrame2 = deconvlucy_x(localFrame1, self.parameters["preprocess"]["deconKernel"], self.parameters["preprocess"]["numIterDecon"])
                                    # localFrame2_2 = restoration.richardson_lucy(localFrame1, self.parameters["preprocess"]["deconKernel"],
                                    #                                           iterations=self.parameters["preprocess"]["numIterDecon"],
                                    #                                           filter_epsilon=sys.float_info.epsilon,clip=False)
                                else:
                                    localFrame2 = localFrame1

                                # Write frame
                                localFrame2 = np.rint(localFrame2).astype("uint16")
                                tiffToWrite.write(localFrame2,**tiffTagStruct)

                                # # Write directory for next frame (unless it is the last
                                # # one)
                                # if (b*self.numZPos + z) != self.numBits*self.numZPos:
                                #     tiffToWrite.writeDirectory()

                                # Accumulate pixel histograms
                                pixelHistogram[b,:] = pixelHistogram[b,:] + np.histogram(localFrame2.flatten(), list(range(65535+1)))[0]

                            if self.verbose:
                                print('[', toc(bit_st),"s]",sep="")

                    elif processingMethod == 'highPassErosion':
                        # Loop over imaging round
                        for b in range(self.numBits):
                            if self.verbose:
                                print('...Processing Bit',b,"/",self.numBits,end=" ")
                                bit_st = tic(0)
                            # Loop over z
                            for z in range(self.numZPos):
                                # Set directory and load
                                tiffToRead = tiffToRead[b*self.numZPos + z]
                                localFrame = tiffToRead.astype("uint16")

                                # High pass filter (and threshold on zero values b/c uint16)
                                # highPassSigma = self.parameters["preprocess"]["highPassKernelSize"]
                                # highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
                                # gaussfilt_img = cv2.GaussianBlur(localFrame, (highPassFilterSize, highPassFilterSize), highPassSigma, borderType=cv2.BORDER_REPLICATE)
                                gaussfilt_img = gaussian(localFrame,sigma=self.parameters["preprocess"]["highPassKernelSize"],preserve_range=True, truncate=2)
                                localFrame1 = localFrame - gaussfilt_img
                                localFrame1[localFrame1 < 0] = 0
                                localFrame1 = np.rint(localFrame1).astype('uint16')

                                # Erode
                                localFrame2 = binary_erosion(localFrame1, self.parameters["preprocess"]["erosionElement"])

                                # Write frame
                                localFrame2 = np.rint(localFrame2).astype('uint16')
                                tiffToWrite.write(localFrame2,**tiffTagStruct)

                                # Accumulate pixel histograms
                                pixelHistogram[b,:] = pixelHistogram[b,:] + np.histogram(localFrame2.flatten(), list(range(65535)))

                            if self.verbose:
                                print('[', toc(bit_st), "s]",sep="")

                    elif processingMethod == 'highPassDeconWB':
                        # Loop over imaging round
                        for b in range(self.numBits):
                            if self.verbose:
                                print('...Processing Bit', b, "/", self.numBits, end=" ")
                                bit_st = tic(0)
                            # Loop over z
                            for z in range(self.numZPos):
                                # Set directory and load
                                tiffToRead = tiffToRead[b*self.numZPos + z]
                                localFrame = tiffToRead.astype("uint16")

                                # High pass filter (and threshold on zero values b/c uint16)
                                # highPassSigma = self.parameters["preprocess"]["highPassKernelSize"]
                                # highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
                                # gaussfilt_img = cv2.GaussianBlur(localFrame, (highPassFilterSize, highPassFilterSize), highPassSigma, borderType=cv2.BORDER_REPLICATE)
                                gaussfilt_img = gaussian(localFrame,
                                                         sigma=self.parameters["preprocess"]["highPassKernelSize"],
                                                         preserve_range=True, truncate=2)
                                localFrame1 = localFrame - gaussfilt_img
                                localFrame1[localFrame1 < 0] = 0
                                localFrame1 = localFrame1.astype("uint16")

                                # Deconvolve
                                localFrame2 = restoration.unsupervised_wiener(localFrame1,self.parameters["preprocess"]["deconKernel"],
                                                                              max_num_iter = self.parameters["preprocess"]["numIterDecon"])

                                # Write frame
                                localFrame2 = np.rint(localFrame2).astype('uint16')
                                tiffToWrite.write(localFrame2,**tiffTagStruct)

                                # Accumulate pixel histograms
                                pixelHistogram[b,:] = pixelHistogram[b,:] + np.histogram(localFrame2.flatten(), list(range(65535)))

                            if self.verbose:
                                print('[', toc(bit_st), "s]",sep="")

                    else:
                        error('[Error]:invalidArguments - Invalid preprocessing method')
                except Exception as e:
                    # Close tiff file and delete partially constructed file
                    tiffToWrite.close()
                    os.remove(tiffName2Write)

                    # Update log
                    print('Encountered error: ',e )
                    # Rethrow error
                    raise(e)

                # Close tiff files
                tiffToWrite.close()

                # Write the pixel histogram file
                with open(pixelHistogramFile, "wb") as fout:
                    pickle.dump(pixelHistogram, fout, pickle.HIGHEST_PROTOCOL)
                print("Saving", pixelHistogramFile)

                # Create display strings
                if self.verbose:
                    print('... completed in',toc(localTimer),'s')
                    print('Completed preprocessing of fov',self.fov2str(localFovID),'at',tic(1))
        else:
            print('[Warning]: The pixel histograms field is not empty, indicating that preprocessing is complete on this data set.')

    # -------------------------------------------------------------------------
    # Combine and generate report for affine transforms
    # -------------------------------------------------------------------------
    def GenerateWarpReport(self):
        # Generate a report on the fiducial warping

        # -------------------------------------------------------------------------
        # Map and load transforms and residuals if needed
        # -------------------------------------------------------------------------
        if len(self.affineTransforms)==0 and len(self.residuals)==0:

            fiducialDataPath = os.path.join(self.normalizedDataPath,self.fiducialDataPath)
            # Display progress
            if self.verbose:
                PageBreak()
                print('Searching for warping files in',fiducialDataPath)
                localTimer = tic()

            # Map the location of affine transform and residuals
            affineFiles,_ = BuildFileStructure(fiducialDataPath,
                                             regExp='fov_(?P<fov>[0-9]+)_affine',
                                             fileExt='pkl', fieldNames=['fov'],fieldConv = [int])

            residualFiles,_ = BuildFileStructure(fiducialDataPath,
                                               regExp='fov_(?P<fov>[0-9]+)_residual',
                                               fileExt = 'pkl',fieldNames=['fov'], fieldConv=[int])

            if self.verbose:
                print('Loading transformation files for',len(affineFiles),'fov')

            # Check for existing files
            if len(affineFiles)==0 or len(residualFiles)==0:
                error('[Error]:missingFiles - Could not find residual or affine files.')

            # Confirm that an affine transform and a residual file exists for
            # each fovID
            if len(np.setdiff1d(self.fovIDs, [i["fov"] for i in affineFiles]))>0:
                error('[Error]:missingFiles - Some affine files are missing.')

            if len(np.setdiff1d(self.fovIDs, [i["fov"] for i in residualFiles]))>0:
                error('[Error]:missingFiles - Some affine files are missing.')

            # Load and combine transforms

            # Sort files in order of increasing fov id
            sind = np.argsort([i["fov"] for i in affineFiles])
            affineFiles = [affineFiles[i] for i in sind]
            sind = np.argsort([i["fov"] for i in residualFiles])
            residualFiles = [residualFiles[i] for i in sind]

            # Allocate memory
            self.affineTransforms = np.tile(AffineTransform(), (self.numDataChannels,self.numFov))
            self.residuals = np.ndarray(shape= (self.numDataChannels,self.numFov),dtype=np.ndarray)

            # Load and store affine files and residuals
            for i in range(len(affineFiles)):
                affine_tform = pickle.load(open(affineFiles[i]["filePath"],"rb"))
                self.affineTransforms[:,i] = affine_tform
                residual_data = pickle.load(open(residualFiles[i]["filePath"], "rb"))
                self.residuals[:,i] = residual_data

            # Display progress
            if self.verbose:
                print('...completed in',toc(localTimer),'s')

            # Save these new fields in the merfish decoder
            self.Save()

            # Delete these files
            if self.verbose and not self.keepInterFiles:
                print('Removing all fiducial files...')
                localTimer = tic()
                shutil.rmtree(fiducialDataPath) # Delete folder
                print('...completed in',toc(localTimer),'s')

        if len(self.geoTransformReport)==0:
            # Update progress
            if self.verbose:
                print('Preparing warp report')
                localTimer = tic()

            # -------------------------------------------------------------------------
            # Create warp report
            # -------------------------------------------------------------------------
            # Check to see if directory exists
            if not os.path.exists(os.path.join(self.normalizedDataPath,self.reportPath)):
                os.makedirs(os.path.join(self.normalizedDataPath,self.reportPath))


            # Generate Transform Report and Save
            reportPath = (os.path.join(self.normalizedDataPath, self.reportPath))
            report= GenerateGeoTransformReport(self.affineTransforms, self.residuals,reportPath=reportPath,
                                                            edges = self.parameters["warp"]["geoTransformEdges"])

            # Update progress
            if self.verbose:
                print('...completed in',toc(localTimer), 's')

            # Save report
            self.geoTransformReport = report
            self.Save()
        else:
            print('A warp report already exists.\n[Skipped OR Clear mDecoder.geoTransformReport and re-run to allow recreation]')
            report = ""
        return report

    # -------------------------------------------------------------------------
    # Reset the scale factors used to normalize different data channels
    # -------------------------------------------------------------------------
    def ResetScaleFactors(self):
        # Reset scale factors
        # self.ResetScaleFactors()      # Reset scale factors

        # Reset scale factors
        self.scaleFactors = []
        if self.verbose:
            PageBreak()
            print('Cleared existing scale factors')


    # -------------------------------------------------------------------------
    # Combine and generate report for affine transforms
    # -------------------------------------------------------------------------
    def OptimizeScaleFactors(self, numFOV, **kwargs):
        # Optimize scale factors
        # OptimizeScaleFactors([])      # Optimize on all FOV
        # OptimizeScaleFactors(numFOV)  # Optimize on numFOV randomly
        # selected FOV ids
        # OptimizeScaleFactors(numFOV, 'overwrite', true) # Overwrite
        # existing optimization

        # -------------------------------------------------------------------------
        # Handle varargin
        # -------------------------------------------------------------------------
        parameters = {}
        parameters['overwrite'] = False    # Overwrite existing scale factors
        parameters['useBlanks'] = True   # Use blank barcodes in the optimization process or not
        parameters['blankFunc'] = lambda x: len(x)==0  # Function to identify blanks based on codebook entry

        for k_i in kwargs:
            parameters[k_i] = kwargs[k_i]

        # -------------------------------------------------------------------------
        # Clear existing scale factors if desired
        # -------------------------------------------------------------------------
        if parameters["overwrite"]:
            self.ResetScaleFactors()

        # -------------------------------------------------------------------------
        # Check to see if optimization has already been completed
        # -------------------------------------------------------------------------
        if len(self.scaleFactors) > 0:
            print('[Warning]:overwrite - Scale factors have been initialized. Clear before rerunning optimization')
            self.ResetScaleFactors()

        # -------------------------------------------------------------------------
        # Check to see if the initial scale factors have been defined
        # -------------------------------------------------------------------------
        if len(self.initScaleFactors) == 0:
            error('[Error]:missingValues - Scale factors have not been initialized')

        # -------------------------------------------------------------------------
        # Select fov IDs for optimization
        # -------------------------------------------------------------------------
        if numFOV==0:
            self.optFovIDs = self.fovIDs
        elif numFOV > self.numFov:
            print('[Warning]: More fov ids were requested than are available. Using all fov ids')
            self.optFovIDs = self.fovIDs
        else:
            self.optFovIDs = self.fovIDs[np.random.choice(self.numFov, numFOV)]
            self.optFovIDs = self.fovIDs[10:10+numFOV]

        # Display fov IDs
        if self.verbose:
            print('Using the following fov IDs for optimization:')
            print(self.optFovIDs)

        # -------------------------------------------------------------------------
        # Generate decoding matrices
        # -------------------------------------------------------------------------
        # Generate the exact barcodes
        exactBarcodes,_ = self.GenerateDecodingMatrices()

        # Cut blanks (if requested)
        if not parameters["useBlanks"]:
            # Identify blanks
            isBlank = np.array([parameters["blankFunc"](i) for i in self.codebook["id"]])

            # Remove blanks
            exactBarcodes = exactBarcodes[~isBlank, :]

            if self.verbose:
                print('Not using',np.sum(isBlank),'blank barcodes for optimization')
        else:
            if self.verbose:
                print('Using all barcodes, including blanks, for optimization.')

        # -------------------------------------------------------------------------
        # Initialize scale factors and other quantities to report
        # -------------------------------------------------------------------------
        numIter = self.parameters["optimization"]["numIterOpt"]
        localScaleFactors = np.zeros((numIter, self.numBits))
        localScaleFactors[0,:] = self.initScaleFactors
        onBitIntensity = np.zeros((numIter, self.numBits))
        allCounts = np.zeros((numIter, exactBarcodes.shape[0]))

        # -------------------------------------------------------------------------
        # Iterate
        # -------------------------------------------------------------------------
        for i in range(numIter):
            # Display progress
            if self.verbose:
                PageBreak()
                print('Starting',i+1,'of',numIter,'iterations')
                iterTimer = tic()

            # Loop over all files for analysis.
            # TODO:perform in parallel to decrease time
            # Create memory for accumulated pixel traces
            accumPixelTraces = np.zeros((exactBarcodes.shape[0],  self.numBits)) # The accumulated pixel traces for all barcodes
            localCounts = np.zeros((exactBarcodes.shape[0],)) # The accumulated number of barcodes

            # Loop over files to analyze
            for f in self.optFovIDs:
                # Determine the local fovID
                localFovID = f

                # Display progress
                if self.verbose:
                    print('>> Optimizing',localFovID)
                    print('Loading stack and decoding...')
                    localTimer = tic()

                # Create tiff to read file name
                tiffName2Read = os.path.join(self.normalizedDataPath,self.processedDataPath,'fov_'+self.fov2str(localFovID)+'.tif')
                if not os.path.exists(tiffName2Read):
                    error('[Error]:missingFile - The requested tiff stack is not present.')

                # Read (and low pass filter and crop) the tiff stack
                localData = self.ReadAndFilterTiffStack(tiffName2Read)

                # Decode data
                decodedImage, localMagnitude, pixelVectors,_ = self.DecodePixels(localData,
                                                                                 localScaleFactors[i,:],
                                                                                 exactBarcodes,
                                                                                 self.parameters["decoding"]["distanceThreshold"])

                # Save memory by clearing local data
                localData = []

                # Set low intensity barcodes to zero
                decodedImage[localMagnitude.reshape(decodedImage.shape,order="F") < self.parameters["decoding"]["minBrightness"]] = 0

                # Display progress
                if self.verbose:
                    print('...completed in',toc(localTimer),'s')
                    print('Compiling pixel traces')
                    localTimer = tic()

                # Loop over codebook entries
                for b in range(exactBarcodes.shape[0]):
                    # Define connected regions
                    # conn = bwconncomp(decodedImage == b, self.parameters["decoding"]["connectivity"])
                    conn = measure.label((decodedImage == (b+1)).astype(np.int))

                    # Identify connected regions
                    # Remove regions smaller than a minimum area
                    properties = measure.regionprops(conn)
                    properties = [x for x in properties if x.area >= self.parameters["decoding"]["minArea"]]

                    # Place additional cuts on properties foroptimization
                    properties = [x for x in properties if x.area >= self.parameters["optimization"]["areaThresh"]]

                    # Accumulate the number of each barcode
                    localCounts[b] = localCounts[b] + len(properties)

                    # Accumulate the pixel traces
                    for l in properties:
                        PixelIdxList = np.array(sorted([sub2ind(decodedImage.shape, c_i[0],c_i[1],order="F") for c_i in l.coords]),dtype=np.int32)
                        localMag = localMagnitude[PixelIdxList]
                        localTraces = pixelVectors[PixelIdxList,:] * np.tile(localMag.reshape(-1,1), (1,self.numBits))
                        localPixelTrace = np.mean(localTraces,0)
                        localPixelTrace = localPixelTrace/np.sqrt(np.sum(localPixelTrace*localPixelTrace)) # Renormalize
                        accumPixelTraces[b,:] = accumPixelTraces[b,:] + localPixelTrace # Accumulate


                # Display progress
                if self.verbose:
                   print('...completed in',toc(localTimer),'s')

            allCounts[i,:]=localCounts
            # ------------------------------------------------------------------------
            # Compute new scale factors
            #--------------------------------------------------------------------------
            # Normalize and zero pixel traces
            normPixelTraces = accumPixelTraces/np.tile(allCounts[i,:].reshape(-1,1), (1,self.numBits))
            normPixelTraces[exactBarcodes== 0] = np.nan

            # Compute the average intensity of the onBitIntensity
            onBitIntensity[i,:] = np.nanmean(normPixelTraces,0)
            refactors = onBitIntensity[i,:]/np.mean(onBitIntensity[i,:])

            # Record new scale factors
            if i < numIter-1:
                localScaleFactors[i+1,:] = localScaleFactors[i,:]*refactors
            if self.verbose:
                print('Completed iteration',i,' in',toc(iterTimer),'s')
        # Save scale factors
        self.scaleFactors = localScaleFactors[i,:]
        self.UpdateField(scaleFactors=localScaleFactors[i,:])
        # Create and display optimization report
        self.GenerateOptimizationReport(localScaleFactors, onBitIntensity, allCounts)

    # -------------------------------------------------------------------------
    # DecodeFOV
    # -------------------------------------------------------------------------
    def DecodeFOV(self, fovIDs,exactBarcodes,singleBitErrorBarcodes):
        # Decode individual images
        # DecodeFOV([])                 # Decode all FOV
        # DecodeFOV(fovIDs)             # Decode specified fovIDs

        # -------------------------------------------------------------------------
        # Determine properties of the requested fov ids
        # -------------------------------------------------------------------------
        if fovIDs==[] or fovIDs=="":
            fovIDs = self.fovIDs
        elif not np.all([f_i in self.fovIDs for f_i in fovIDs]):
            error('[Error]:invalidArguments - An invalid fov id has been requested')

        # -------------------------------------------------------------------------
        # Initialize and check scale factors
        # -------------------------------------------------------------------------
        localScaleFactors = self.scaleFactors # Use the scale factors created by compiling the results of all optimized FOV
        if np.size(localScaleFactors)==0:
            error('[Error]:nonexistingVariable - Decoding cannot be run until scale factors have been initialized')

        # -------------------------------------------------------------------------
        # Decode individual FOV
        # -------------------------------------------------------------------------
        # Loop over individual fov
        for f in fovIDs:
            # Determine local fovID
            localFovID = f

            # Create display strings
            if self.verbose:
                PageBreak()
                print('Started decoding of fov',self.fov2str(localFovID))

            # Define barcode file path and check for uncorrupted existance
            barcodeFile = os.path.join(self.normalizedDataPath,self.barcodePath, 'barcode_fov','fov_'+self.fov2str(localFovID)+'_blist.pkl')

            # Erase if overwrite
            if self.overwrite:
                if os.path.exists(barcodeFile):
                    os.remove(barcodeFile)
                    if self.verbose: print('Overwriting...',barcodeFile)

            # Check if corrupt and erase if it is
            if os.path.exists(barcodeFile):
                if self.verbose: print('Found existing barcode file',barcodeFile)
                isCorrupt = False
                try:
                    fileHeader = ReadBinaryFileHeader(barcodeFile)
                except:
                    isCorrupt = True

                if isCorrupt:
                    if self.verbose: print('File is corrupt. Overwriting...')
                    os.remove(barcodeFile)


            # Skip if the file exists
            if os.path.exists(barcodeFile):
                if self.verbose: print('File is complete. Skipping analysis')
                continue

            # Display progress
            if self.verbose:
                print('Loading preprocessed stack')
                localTimer = tic()

            # Create tiff to read file name
            tiffName2Read =os.path.join(self.normalizedDataPath,self.processedDataPath, 'fov_'+self.fov2str(localFovID)+'.tif')
            if not os.path.exists(tiffName2Read):
                error('[Error]:missingFile - The requested tiff stack is not present.')

            # Read (and low pass filter) the tiff stack
            localData = self.ReadAndFilterTiffStack(tiffName2Read)

            # Create display strings
            if self.verbose:
                print('... completed in',toc(localTimer),'s')
                print('Starting decoding')
                print('... starting barcode assignment')
                localTimer = tic(0)
                assignmentTimer = tic(0)

            # Decode data
            [decodedImage, localMagnitude, pixelVectors, D] = self.DecodePixels(localData,
                                                                                localScaleFactors,
                                                                                exactBarcodes,
                                                                                self.parameters["decoding"]["distanceThreshold"]
                                                                                )

            # Clear local data to open up memory
            localData = []

            # Set low intensity barcodes to zero
            decodedImage[localMagnitude.reshape(decodedImage.shape, order="F") < self.parameters["decoding"]["minBrightness"]] = 0

            # Create display strings
            if self.verbose:
                print('... ... completed assignment in',toc(assignmentTimer),'s')
                print('... saving decoded image')
                saveTimer = tic(0)

            # Save the decoded image and the magnitude image
            self.SaveDecodedImageAndMagnitudeImage(decodedImage, np.reshape(localMagnitude, decodedImage.shape,order="F"), localFovID)

            # Create display strings
            if self.verbose:
                print('... ... completed save in',toc(saveTimer),'s')
                print('... starting metadata assembly')
                metadataTimer = tic()


            # Clear measured barcodes
            measuredBarcodes = []
            if self.verbose:
                process_bar = ""
                print("****|****|****|****|****|****|****|****|****|****|100%")
            for b in range(self.numBarcodes): # Loop over codebook entries
                # Define connected regions
                conn = measure.label(decodedImage == (b+1))

                # Identify connected regions
                properties = measure.regionprops(conn)

                # Remove regions smaller than a minimum area
                properties = [x for x in properties if x.area >= self.parameters["decoding"]["minArea"]]

                # Compile properties of measured barcodes
                measuredBarcodes += self.GenerateBarcodes(properties,
                                                          localMagnitude,
                                                          singleBitErrorBarcodes,
                                                          pixelVectors, D, b, localFovID)
                if self.verbose:
                    process_per = (b+1) *100 / 140
                    process_n = int(process_per // 2)
                    if process_n > len(process_bar):
                        delta_n = process_n-len(process_bar)
                        process_bar+=">" * delta_n
                        print('>'*delta_n, end="")
            print("")


            # End loop over barcodes
            measuredBarcodes_pd = pd.DataFrame(measuredBarcodes)
            # Create display strings
            if self.verbose:
                print('... ... completed metadata assembly in',toc(metadataTimer),'s')
                print('... completed decoding in',toc(localTimer), 's')
                print('... saving',len(measuredBarcodes),'barcodes')
                localTimer = tic()

            # Write binary file for all measured barcodes
            WriteBinaryFile(barcodeFile, measuredBarcodes_pd)

            # Finish and display the progress strings
            if self.verbose:
                print('... completed in',toc(localTimer),'s')



    # -------------------------------------------------------------------------
    # SaveDecodedImageAndMagnitudeImage
    # -------------------------------------------------------------------------
    def SaveDecodedImageAndMagnitudeImage(self, decodedImage, localMagnitude, fovID):
        # Helper function: save these images as tiff stacks

        # Create path if necessary
        imagePath =os.path.join(self.normalizedDataPath,self.barcodePath,'decoded_images')
        if not os.path.exists(imagePath):
            os.makedirs(imagePath,exist_ok=True)

        # Create tiff Tag structure
        tiffTagStruct = {}
        # tiffTagStruct["shape"] = self.imageSize
        tiffTagStruct["photometric"] = TIFF.PHOTOMETRIC.MINISBLACK
        # tiffTagStruct["bitspersample"] = 16
        # tiffTagStruct["samplesperpixel"] = 1 ## tag id: 277
        # tiffTagStruct["sampleformat"] = TIFF.SAMPLEFORMAT.IEEEFP  ## tag id: 339
        tiffTagStruct["rowsperstrip"] = 16
        tiffTagStruct["planarconfig"] = TIFF.PLANARCONFIG.CONTIG
        tiffTagStruct["software"] = 'Ruifeng'
        tiffTagStruct["description"] = 'images=' + str(self.numZPos*2) + '\n' + \
                                       'channels=1\n' + \
                                       'slices=' +str(self.numZPos) +'\n' + \
                                       'frames=2\n' + \
                                       'hyperstack=True\n' + \
                                       'loop=False\n'
        tiffExtrTags = [(339,"i",1,TIFF.SAMPLEFORMAT.IEEEFP,False),(277,"i",1,1,False)]



        # Write tiff for the decodedImage
        tiffImage = TiffWriter(os.path.join(imagePath,'fov_'+self.fov2str(fovID)+'.tif'), bigtiff=True, append=True)

        # Write the decoded image
        if self.numZPos > 1:
            for z in range(self.numZPos):
                tiffImage.write((decodedImage[:,:,z]).astype('float'),**tiffTagStruct,extratags=tiffExtrTags)
            # Write the magnitude image
            for z in range(self.numZPos):
                tiffImage.write((localMagnitude[:, :, z]).astype('float'), **tiffTagStruct, extratags=tiffExtrTags)
        else:
            tiffImage.write((decodedImage).astype('float'), **tiffTagStruct, extratags=tiffExtrTags)
            tiffImage.write((localMagnitude).astype('float'), **tiffTagStruct, extratags=tiffExtrTags)

        # Close tiff files
        tiffImage.close()


#     # -------------------------------------------------------------------------
#     # Convert pixel coordinates into absolute, real-world coordinates
#     # -------------------------------------------------------------------------
#     function absPos = Pixel2Abs(obj, pixelPos, fovID)
#         # Convert a set of pixel coordinates in a given fov to the absolute coordinate system
#         #
#         # absPos = self.Pixel2Abs(pixelPos, fovID)
#
#         # Convert x and y
#         absPos = self.fovPos(self.fovIDs == fovID,:) + ...
#             self.pixelSize/1000*self.parameters.decoding.stageOrientation.* ...
#             (double(pixelPos(:,1:2)) - self.imageSize/2)
#
#         # Convert z via linear interpolation
#         if size(pixelPos,2) == 3
#             absPos(:,3) = interp1(1:self.numZPos, self.zPos, double(pixelPos(:,3)))
#         end
#
#     end

    # -------------------------------------------------------------------------
    # GenerateBarcodes
    # -------------------------------------------------------------------------
    def GenerateBarcodes(self, properties, localMagnitude, singleBitErrorBarcodes, pixelTraces, D, b, fovID):

        # Define the basic barcode structure.... just for reference
        #         measuredBarcodes = repmat(struct(...
        #             'barcode', uint64(0), ...
        #             'barcode_id', uint16(0), ...
        #             'fov_id', uint16(0), ...
        #             'total_magnitude', single(0), ...
        #             'pixel_centroid', single(zeros(1, 2 + floor(self.numZPos-1))), ...
        #             'weighted_pixel_centroid', single(zeros(1, 2 + floor(self.numZPos-1))), ...
        #             'abs_position', single(zeros(1,3)), ...
        #             'area', uint16(0), ...
        #             'pixel_trace_mean', single(zeros(1, self.numBits)), ...
        #             'pixel_trace_std', single(zeros(1, self.numBits)), ...
        #             'is_exact', uint8(0), ...
        #             'error_bit', uint8(0), ...
        #             'error_dir', uint8(0), ...
        #             'av_distance', single(0)), ...
        #             [1 0])

        measuredBarcodes = []
        # Loop over all barcodes
        for p in properties:
            # Transfer barcode properties
            measuredBarcodes_p = {}
            measuredBarcodes_p['barcode'] = np.uint64(self.codebook["barcode"][b])          # The barcode
            measuredBarcodes_p['barcode_id'] = np.uint64(b)     # The order of the entry in the codebook

            # Transfer fov id
            measuredBarcodes_p['fov_id'] = np.uint64(fovID)

            p_PixelIdxList = np.array(sorted([sub2ind(p._label_image.shape, c_i[0],c_i[1],order="F") for c_i in p.coords]),dtype=np.int32)

            # Compute weighted centroid
            y,x,z = ind2sub3d([self.imageSize[0]-2*self.parameters["decoding"]["crop"],self.imageSize[1]-2*self.parameters["decoding"]["crop"],self.numZPos],
                              p_PixelIdxList,order="F")
            magnitude = localMagnitude[p_PixelIdxList]
            totalMagnitude = np.sum(magnitude)
            weightedX = np.sum(x*magnitude)/totalMagnitude
            weightedY = np.sum(y*magnitude)/totalMagnitude
            weightedZ = np.sum(z*magnitude)/totalMagnitude

            # Compute total magnitude
            measuredBarcodes_p['total_magnitude'] = np.float(totalMagnitude)     # The total brightness of all pixel traces

            # Transfer position properties
            if len(p.centroid) == 2: # Handle single z slice case
                measuredBarcodes_p['pixel_centroid'] = np.rint(p.centroid + (self.parameters["decoding"]["crop"])*np.array([1,1]))  # The absolute pixel location in the image
                measuredBarcodes_p['weighted_pixel_centroid'] = ([weightedY,weightedX] + (self.parameters["decoding"]["crop"])*np.array([1,1])).astype("float")  # The X/Y/Z position in camera coordinates weighted by magnitude
            elif len(p.centroid) == 3:
                measuredBarcodes_p['pixel_centroid'] = (p.centroid + (self.parameters["decoding"]["crop"])*np.array([1,1,0])).astype("uint16")    # The absolute pixel location in the image
                measuredBarcodes_p['weighted_pixel_centroid']=([weightedY,weightedX,weightedZ] + (self.parameters["decoding"]["crop"])*np.array([1,1,0])).astype("float")   # The X/Y/Z position in camera coordinates weighted by magnitude

            # Calculate absolute position
            absXYPos = self.fovPos[self.fovIDs == fovID,:].flatten() + \
                       self.pixelSize / 1000 * np.array(self.parameters["decoding"]["stageOrientation"]) *  \
                       (measuredBarcodes_p['weighted_pixel_centroid'][0:2] - np.array(self.imageSize)/2)

            weightedZ = np.round(weightedZ, 2) # Handle round-off error
            zLow = self.zPos[int(np.floor(weightedZ))] # Handle weighted Z points inbetween z planes
            zHigh = self.zPos[int(np.ceil(weightedZ))]
            absZPos = zLow + (zHigh - zLow)*(weightedZ-np.floor(weightedZ))

            measuredBarcodes_p['abs_position'] =np.append(absXYPos, absZPos)                        # The absolute position in the stage coordinates

            # Transfer area
            measuredBarcodes_p['area'] = np.uint16(p.area)                                # The number of pixels

            # Compute average pixel traces
            localPixelTrace = np.mean(pixelTraces[p_PixelIdxList,:],0)
            measuredBarcodes_p['pixel_trace_mean'] = localPixelTrace.astype('float')
            measuredBarcodes_p['pixel_trace_std'] = np.std(pixelTraces[p_PixelIdxList,:],0)

            # Compute location within the Hamming sphere
            nnID,_ = knnsearch2d(singleBitErrorBarcodes[b,:,:], localPixelTrace.reshape(1,-1))
            measuredBarcodes_p['is_exact'] = np.uint8(nnID[0] == 0)      # Whether or not this barcode best matches the exact barcode
            measuredBarcodes_p['error_bit'] = np.uint8(nnID[0])          # The bit at which an error occurred (0 if no error occurred)
            if measuredBarcodes_p['error_bit'] > 0:
                measuredBarcodes_p['error_dir'] = np.uint8(singleBitErrorBarcodes[b,0,measuredBarcodes_p['error_bit']-1]>0)     # The identity of the original bit (1 = 1->0 error 0 = 0->1 error)
            else:
                measuredBarcodes_p['error_dir'] = np.uint8(0)

            # Record average distance
            measuredBarcodes_p['av_distance'] = np.float(np.mean(D[p_PixelIdxList]))         # The average distance from the pixel traces to the matched barcode

            measuredBarcodes.append(measuredBarcodes_p)
        return measuredBarcodes

    # -------------------------------------------------------------------------
    # Combine pixel histograms
    # -------------------------------------------------------------------------
    def ReadAndFilterTiffStack(self, tiffName2Read):

        # Allocate memory for image
        localData = np.zeros((self.imageSize[1], self.imageSize[0], self.numZPos, self.numBits))

        # Loop over frames, loading, accumulating pixel values, and
        # filtering (if requested)
        for b in range(self.numBits):
            # Loop over z-stacks
            for z in range(self.numZPos):
                # Determine frame location in stack
                frame = b*self.numZPos + z
                # Set directory and load
                tiffToRead = imread(tiffName2Read,key=frame)
                localFrame = tiffToRead
                # Low pass filter
                if self.parameters["decoding"]["lowPassKernelSize"] > 0:
                    # lowPassSigma = self.parameters["decoding"]["lowPassKernelSize"]
                    # lowPassFilterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
                    # localData[:,:,z,b] = cv2.GaussianBlur(localFrame,(lowPassFilterSize,lowPassFilterSize),lowPassSigma, borderType=cv2.BORDER_REPLICATE)
                    localData[:, :, z, b] = np.rint(gaussian(localFrame, sigma=self.parameters["decoding"]["lowPassKernelSize"],
                                             preserve_range=True, truncate=2))
                else:
                    localData[:,:,z,b] = localFrame

        # Crop edges
        localData = localData[(self.parameters["decoding"]["crop"]):(-self.parameters["decoding"]["crop"]),
                    (self.parameters["decoding"]["crop"]):(-self.parameters["decoding"]["crop"]), :, :]

        return localData


    # -------------------------------------------------------------------------
    # Initialize scale factors
    # -------------------------------------------------------------------------
    def InitializeScaleFactors(self):
        # Initialize scale factors and combine pixel histograms

        # Combine pixel histograms
        if np.size(self.pixelHistograms)==0:
            # Display progress
            if self.verbose:
                PageBreak()
                print('Combining pixel histograms...')
                combineTimer = tic()

            # Identify the saved pixel histogram files
            hist_folder = os.path.join(self.normalizedDataPath,self.processedDataPath,'pixel_histograms')
            foundFiles,_ = BuildFileStructure(hist_folder,regExp='pixel_data_fov_(?P<fov>[0-9]+)',
                                            fileExt='pkl',fieldNames = ['fov'],fieldConv=[int])
            fileType = 'pkl'

            # Allow for csv files (and different naming convention)
            if len(foundFiles)==0:
                hist_folder = os.path.join(self.normalizedDataPath, self.processedDataPath, 'pixel_histograms')
                foundFiles = BuildFileStructure(hist_folder, regExp = 'fov_(?P<fov>[0-9]+)',
                                                fileExt='csv', fieldNames=['fov'],fieldConv=[int])
                fileType = 'csv'

            # Display progress
            if self.verbose:
                print('...found',len(foundFiles),'pixel histogram files')

            # Check that all fov are present
            if len(np.setdiff1d(self.fovIDs, [i["fov"] for i in foundFiles]))>0:
                print('The following fov ids do not have pixel_histograms:')
                print(np.setdiff1d(self.fovIDs, foundFiles["fov"]))
                error('[Error]:missingFiles - A pixel histogram for all fov ids was not found.')

            # Initialize and allocate memory
            pixelHistograms =  np.zeros((self.numBits, 65535)) # Allocate memory and initialize to zero

            foundFiles_DF = pd.DataFrame(foundFiles)
            # Loop over all fov ids
            for f in range(self.numFov):
                localFovID = self.fovIDs[f]
                if fileType == 'pkl':
                    file_path = foundFiles_DF.loc[foundFiles_DF["fov"] == localFovID,"filePath"].values[0]
                    localData = pickle.load(open(file_path,"rb"))
                elif fileType=='csv':
                    localData = pd.read_csv(foundFiles_DF.loc[foundFiles_DF["fov"] == localFovID,"filePath"])
                else:
                    error('[Error]:invalidArguments - Unrecognized file extension requested for pixel histograms')
                pixelHistograms = pixelHistograms + localData # Combine pixel histogram

            # Update progress
            if self.verbose:
                print('...completed in',toc(combineTimer),'s')

            # Remove pixel histogram files
            if self.verbose and not self.keepInterFiles:
                print('Deleting intermediate files...')
                localTimer = tic()
                shutil.rmtree(os.path.join(self.normalizedDataPath,self.processedDataPath,'pixel_histograms'))
                print('...completed in',toc(localTimer),'s')

            # Update pixel histograms in decoder
            self.UpdateField(pixelHistograms = pixelHistograms)

        # Create report path for saving the initial scale factor selection
        if not os.path.exists(os.path.join(self.normalizedDataPath,self.reportPath)):
            os.makedirs(os.path.join(self.normalizedDataPath,self.reportPath),exist_ok=True)

        # Create figure to display selections
        file_name = "Intensity normalization"
        fig = plt.figure(file_name,facecolor='w',figsize=(10,8))

        markers = ['r', 'g', 'b', 'y', 'm', 'c',
                   'r--', 'g--', 'b--', 'y--', 'm--', 'c--',
                   'r.-', 'g.-', 'b.-', 'y.-', 'm.-', 'c.-']

        # Loop over bits
        minValue = np.inf
        initScaleFactors = np.zeros((self.numBits,),dtype=np.int32)
        quantilePos = np.zeros((self.numBits,))
        for i in range(self.numBits):
            cumSum = np.cumsum(self.pixelHistograms[i,:])
            cumSum = cumSum/cumSum[-1] # Normalize to 1

            localScale = np.argmin(np.abs(cumSum - self.parameters["optimization"]["quantileTarget"]))
            initScaleFactors[i]= localScale + 1 # Add 1 b/c 0 is the first bin in the histogram

            plt.semilogx(cumSum, markers[np.mod(i, len(markers))],linewidth=0.6)

            # Record minimum value for improved display
            minValue = np.min([minValue,cumSum[0]])
            quantilePos[i] = cumSum[initScaleFactors[i]]

        # Plot selection points
        for i in range(self.numBits):
            plt.semilogx(np.array([1,1])*initScaleFactors[i] ,np.array([0,1])*quantilePos[i], 'k--',linewidth=0.6)

        # Add plot annotations
        plt.xlabel('Intensity')
        plt.ylabel('Cumulative probability')
        plt.legend([str(i) for i in range(self.numBits)],loc="lower right")
        plt.ylim([minValue,1])

        # Save and close report figure
        plt.savefig(os.path.join(self.normalizedDataPath,self.reportPath, file_name + "."+self.parameters["display"]["formats"] ))
        plt.close()

        # Normalize the scale factors to 1 to facilitate real brightness
        # measures
        if self.parameters["optimization"]["normalizeToOne"]:
            if self.verbose:
                print('Normalizing scale factors to 1')
            initScaleFactors = initScaleFactors/np.mean(initScaleFactors)

        # Record the initial scale factors
        self.initScaleFactors = initScaleFactors
        self.UpdateField(initScaleFactors=initScaleFactors)

        # Display progress
        if self.verbose:
            print('Using the following initial scale factors: ')
            print(self.initScaleFactors)


    # -------------------------------------------------------------------------
    # GenerateDecodingMatrices
    # -------------------------------------------------------------------------
    def GenerateDecodingMatrices(self):
        # Generate matrics used in the decoding process from the loaded codebook

        # Display progress
        if self.verbose:
            PageBreak()
            print('Generating decoding matrices')
            localTimer = tic()

        # Extract the binary barcodes
        binaryBarcodes = de2bi(self.codebook["barcode"], self.numBits)

        # Calculate magnitude and normalize to unit length
        magnitudeTemp = np.sqrt(np.sum(binaryBarcodes*binaryBarcodes,1))
        weightedBarcodes = binaryBarcodes/np.tile(magnitudeTemp.reshape(-1,1), (1,binaryBarcodes.shape[1]))

        singleBitErrorBarcodes=[]
        # Compute the single bit error matrices for computing error rates
        for b in range(self.numBarcodes):
            # Create an array of the correct and all single bit flip barcodes
            singleBitFlipIntegers = [self.codebook["barcode"][b]]+[bitFlip(self.codebook["barcode"][b],n) for n in range(self.numBits)]
            singleBitFlipBinary = de2bi(singleBitFlipIntegers,self.numBits)

            # Normalize
            localMag = np.sqrt(np.sum(singleBitFlipBinary*singleBitFlipBinary,1))
            singleBitErrorBarcodes.append(singleBitFlipBinary/np.tile(localMag.reshape(-1,1), (1,singleBitFlipBinary.shape[1])))

        if self.verbose:
            print('...completed in',toc(localTimer),'s')
        return weightedBarcodes, np.array(singleBitErrorBarcodes)

    # -------------------------------------------------------------------------
    # GenerateOptimizationReport
    # -------------------------------------------------------------------------
    def GenerateOptimizationReport(self, localScaleFactors, onBitIntensity, allCounts):
        # Check to see if directory exists
        reportPath = os.path.join(self.normalizedDataPath,self.reportPath)
        if not os.path.exists(reportPath):
            os.makedirs(reportPath,exist_ok=True)

        # Display report
        file_name = "Optimization report"
        fig = plt.figure(file_name,figsize=(12,5))

        # Compile data to plot
        data = [localScaleFactors, onBitIntensity, np.log10(allCounts)]
        titles = ['Scale factor', 'On-bit intensity', 'Counts (log$_{10}$)']

        # Create subplots
        for p in range(len(data)):
            fig.add_subplot(1, len(data), p+1)
            plt.pcolor(data[p], cmap=parula_map)
            plt.ylabel('Iteration')
            plt.title(titles[p])
            plt.xlim((1,data[p].shape[1]))
            plt.ylim((1,data[p].shape[0]))
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(reportPath,file_name+"."+self.parameters["display"]["formats"]))
        plt.close()


    # -------------------------------------------------------------------------
    # Save Function
    # -------------------------------------------------------------------------
    def Save(self, **kwargs):
        # Save the MERFISHDesigner object in a directory specified by dirPath
        # self.Save(dirPath)

        # -------------------------------------------------------------------------
        # Check directory validity
        # -------------------------------------------------------------------------
        # Set default save path
        if len(kwargs) == 0:
            dirPath = os.path.join(self.normalizedDataPath, self.mDecoderPath)  # Make path an absolute path
        else:
            if "dirPath" in kwargs:
                dirPath = kwargs["dirPath"]
            else:
                dirPath = os.path.join(self.normalizedDataPath, self.mDecoderPath)  # Make path an absolute path

        # Make the directory if needed
        if not os.path.exists(dirPath):
            os.makedirs(dirPath,exist_ok=True)

        # Update MERFISH decoder path
        # self.mDecoderPath = dirPath  # This will make the path an absolute rather than a relative path...

        # -------------------------------------------------------------------------
        # Save fields
        # -------------------------------------------------------------------------
        save_dict = {}
        for k_i in self.__dict__:
            if k_i in ["parallel","numPar","fov2str"]:
                continue
            save_dict[k_i] = self.__dict__[k_i]

        with open(os.path.join(dirPath,"mDecoder.pkl"), "wb") as fout:
            pickle.dump(save_dict, fout, pickle.HIGHEST_PROTOCOL)
        print("Saving",os.path.join(dirPath,"mDecoder.pkl"))




#     # -------------------------------------------------------------------------
#     # SetParameter
#     # -------------------------------------------------------------------------
#     function SetParameter(obj, varargin)
#         # Set fields in the parameters structure
#
#         # Define default parameter sets
#         defaultParameters = cell(0,2)
#         defaultParameters(end+1,:) = {'warp', MERFISHDecoder.DefaultWarpParameters()}
#         defaultParameters(end+1,:) = {'preprocess', MERFISHDecoder.DefaultPreprocessingParameters()}
#         defaultParameters(end+1,:) = {'decoding', MERFISHDecoder.DefaultDecodingParameters()}
#         defaultParameters(end+1,:) = {'optimization', MERFISHDecoder.DefaultOptimizationParameters()}
#         defaultParameters(end+1,:) = {'display', MERFISHDecoder.DefaultDisplayParameters()}
#         defaultParameters(end+1,:) = {'segmentation', MERFISHDecoder.DefaultSegmentationParameters()}
#         defaultParameters(end+1,:) = {'quantification', MERFISHDecoder.DefaultQuantificationParameters()}
#         defaultParameters(end+1,:) = {'summation', MERFISHDecoder.DefaultSummationParameters()}
#         defaultParameters(end+1,:) = {'molecules', MERFISHDecoder.DefaultMoleculeParameters()}
#
#         # Create flag to catch parameters not updated
#         requestedFields = varargin(1:2:end)
#
#         # Loop over parameter sets
#         for p=1:size(defaultParameters,1)
#             # Find matches to defaults
#             localDefaults = defaultParameters{p,2}
#             matchInd = find(ismember(varargin(1:2:end), localDefaults(:,1)))
#             matchInd = sort([(2*matchInd -1) 2*matchInd])
#             if ~isempty(matchInd)
#                 parameters = ParseVariableArguments(varargin(matchInd), localDefaults, mfilename)
#                 # Transfer fields
#                 fieldsToUpdate = varargin(matchInd(1:2:end))
#                 for f=1:len(fieldsToUpdate)
#                     # Store parameters
#                     self.parameters.(defaultParameters{p,1}).(fieldsToUpdate{f}) = parameters.(fieldsToUpdate{f})
#                 end
#                 # Mark fields as updated
#                 requestedFields = setdiff(requestedFields, fieldsToUpdate)
#             end
#         end
#
#         # Raise warning
#         if ~isempty(requestedFields)
#             print('[Warning]:invalidParameters - One or more of the requested parameters are not recognized')
#             for r=1:len(requestedFields)
#                 print(requestedFields{r})
#             end
#         end
#
#     end


    # -------------------------------------------------------------------------
    # Update normalized data path
    # -------------------------------------------------------------------------
    def UpdateNormalizedDataPath(self,newPath):
        # Update the normalized data path. Useful when the directory has
        # been copied to a new location
        # self.UpdateNormalizedDataPath(newPath)

        # -------------------------------------------------------------------------
        # Check validity of arguments
        # -------------------------------------------------------------------------
        if not os.path.exists(newPath):
            error('[Error]:invalidPath - A valid path must be provided')

        # -------------------------------------------------------------------------
        # Update path
        # -------------------------------------------------------------------------
        self.normalizedDataPath = newPath


    # -------------------------------------------------------------------------
    # Update Field
    # -------------------------------------------------------------------------
    def UpdateField(self, **kwargs):
        # Save the MERFISHDesigner object in a directory specified by dirPath
        # self.UpdateField(fields...)

        # -------------------------------------------------------------------------
        # UpdateFields
        # -------------------------------------------------------------------------
        validFields = list(self.__dict__.keys())

        # -------------------------------------------------------------------------
        # Save fields
        # -------------------------------------------------------------------------
        for i in kwargs:
            if i in validFields:
                if self.verbose:
                    print('Updating',i)
                self.__dict__[i] = kwargs[i]
            else:
                print(f'[Warning]: {i} is not a valid field')
        self.Save()

    # -------------------------------------------------------------------------
    # LoadField
    # -------------------------------------------------------------------------
    def LoadField(self, **kwargs):
        # Update the current MERFISHDecoder with revised values saved to disk
        # self.LoadField(fields...)

        # -------------------------------------------------------------------------
        # UpdateFields
        # -------------------------------------------------------------------------
        validFields = self.__dict__

        # -------------------------------------------------------------------------
        # Save fields
        # -------------------------------------------------------------------------
        for i in kwargs:
            if i in validFields:
                if self.verbose:
                    print('Updating',i)
                self.__dict__[i]= kwargs[i]
            else:
                print(f'[Warning]: {i} is not a valid field. Skippped!')


    # -------------------------------------------------------------------------
    # Downsample dataset: This hidden function allows datasets to be
    # downsampled for the purpose of illustration
    # -------------------------------------------------------------------------
    def Downsample(self, bitNamesToRemove, fovIDsToRemove):
        # Identify the bits to remove

        shouldKeepBit = ~np.isin(self.dataOrganization["bitName"].values, bitNamesToRemove)
        shouldKeepFOV = ~np.isin(self.fovIDs, fovIDsToRemove)

        # Cut the data organization file
        self.dataOrganization = self.dataOrganization(shouldKeepBit)

        # Cut the nubmer of data channels
        self.numDataChannels = len(self.dataOrganization)

        # Save the original largest fovID for the purposes of matching the
        # pad in fov strings
        self.originalMaxFovID = np.max(self.fovIDs)

        # Cut the fov
        self.fovIDs = self.fovIDs[shouldKeepFOV]
        self.fovPos = self.fovPos[shouldKeepFOV,:]
        self.numFov = len(self.fovIDs)

        # Clear values associated with optimization
        self.scaleFactors = []
        self.optFovIDs = []
        self.rawDataFiles = []
        self.affineTransforms = self.affineTransforms(shouldKeepBit, shouldKeepFOV)
        self.residuals = self.residuals(shouldKeepBit, shouldKeepFOV)
        self.geoTransformReport = []
        self.pixelHistograms = []
        self.initScaleFactors = []


    # -------------------------------------------------------------------------
    # Build a MERFISHDecoder object from a saved version
    # -------------------------------------------------------------------------
    @staticmethod
    def Load(dirPath,mDecoderPath, **kwargs):
        # obj = MERFISHDecoder.Load(dirPath)

        filePath = os.path.join(dirPath,mDecoderPath,'mDecoder.pkl')
        # -------------------------------------------------------------------------
        # Check provided path
        # -------------------------------------------------------------------------
        if not os.path.exists(filePath):
            error('[Error] Load mDecoder:invalidArguments - The provided path is not valid. No mDecoder is found!')

        # -------------------------------------------------------------------------
        # Handle varargin
        # -------------------------------------------------------------------------
        parameters = {}
        parameters['verbose'] = True   # Display load progress
        for i in kwargs:
            parameters[i] = kwargs[i]

        # -------------------------------------------------------------------------
        # Create empty object (to define fields to load)
        # -------------------------------------------------------------------------
        obj = MERFISHDecoder()

        # -------------------------------------------------------------------------
        # Define fields to load
        # -------------------------------------------------------------------------
        fieldsToLoad = np.setdiff1d(list(obj.__dict__.keys()), ['parallel', 'numPar', 'mDecoderPath', 'fov2str']) ## Update me

        # -------------------------------------------------------------------------
        # Display progress
        # -------------------------------------------------------------------------
        if parameters["verbose"]:
            PageBreak()
            print('Loading MERFISH Decoder from',filePath)
            loadTimer = tic()

        with open(filePath, 'rb') as fin:
            loaded_dict = pickle.load(fin)

        # -------------------------------------------------------------------------
        # Check to see if valid -- all previous versions have verbose field
        # -------------------------------------------------------------------------
        if "verbose" not in loaded_dict:
            error('[Error]: The mDecoder appears to be corrupt!')

        # -------------------------------------------------------------------------
        # Load the version number
        # -------------------------------------------------------------------------
        try:
            version = loaded_dict["version"]
        except: # Handle the case that the saved version is not in matb format or is non-existent
            version = '0.1'

        # -------------------------------------------------------------------------
        # Load properties/data
        # -------------------------------------------------------------------------
        if version in ['0.1', '0.2', '0.3','0.4', '0.5', '0.6']:
            for i in fieldsToLoad:
                if i in loaded_dict:
                    obj.__dict__[i]= loaded_dict[i]
                else:
                    print(f"[Warning] Did not find a default field: {i}")
        else:
            error('[Error]:unsupportedVersion. The version is not supported')

        if parameters["verbose"]:
            print('Version:',obj.version)
            print('FOV:',obj.numFov)
            print('Bits:',obj.numBits)
            print('Data channels:',obj.numDataChannels)
            print('Number of cameras:',obj.numCameraIDs)
            print('...Loaded in',toc(loadTimer), 's')

        # -------------------------------------------------------------------------
        # Handle updated parameters fields (version up-conversion)
        # -------------------------------------------------------------------------
        # Define default parameter sets
        _, defaultParameters = obj.InitializeParameters()

        # Compare to defaults to loaded parameters
        for c in defaultParameters: # Loop over parameters sub sets
            # Get field type
            parametersStruct = defaultParameters[c]
            # Check to see if the loaded parameters have this sub-set
            if c not in obj.parameters:
                obj.parameters[c] = defaultParameters[c]
                print('Set missing',c,'parameters to the default values.')
            else: # Investigate fields one by one with this sub-set (if it exists)
                # Get parameter names
                localParameterFields = list(parametersStruct.keys())
                # Loop over parameter names
                for f in localParameterFields:
                    # If the parameter is missing, add it and give it the
                    # default value
                    if f not in obj.parameters[c]:
                        obj.parameters[c][f] = parametersStruct[f]
                        print('Set missing value ',c+'.'+f,' to the default value.')

        # -------------------------------------------------------------------------
        # Handle up-conversion of rawDataFiles
        # -------------------------------------------------------------------------
        if len(obj.rawDataFiles)!=0 and  ('cameraID' not in obj.rawDataFiles):
            print('Adding a cameraID field to rawDataFiles to maintain version compatibility')
            obj.rawDataFiles["cameraID"] = ''


        # -------------------------------------------------------------------------
        # Handle updated parameters fields (important for some version up-conversions)
        # -------------------------------------------------------------------------
        padNum2str = lambda x, y: str(x).zfill(int(np.ceil(np.log10(y + 1))))
        if len(obj.originalMaxFovID)>0:
            obj.fov2str =lambda x: padNum2str(x, np.max(obj.originalMaxFovID))
        else:
            obj.fov2str = lambda x: padNum2str(x, np.max(obj.fovIDs))
        # -------------------------------------------------------------------------
        # Handle the location of the mDecoder (the normalizedDataPath is
        # the load path)
        # -------------------------------------------------------------------------
        obj.UpdateNormalizedDataPath(dirPath)

        return obj

    # -------------------------------------------------------------------------
    # Generate Warp defaults
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultWarpParameters():
        # Generate the default parameters information for the warping method
        # defaultCell = MERFISHDecoder.WarpDefaultParameters()

        defaults = {}
        defaults['warpingDataPath'] = ""             # Path to saved data for fiducial information
        defaults['fiducialFitMethod'] = "daoSTORM"           # Method for fiducial fitting
        defaults['controlPointOffsetRange'] = np.arange(-60,61,0.5)                              # The histogram properties for determining crude offset
        defaults['numNN'] = 10          # The number of nearest neighbors to include in the search
        defaults['pairDistanceTolerance']=3  # The multiple of the histogram distance used to judge paired beads
        defaults['pixelSize'] = 109     # Pixel size in nm/pixel
        defaults['sigmaInit'] = 1.6     # Initial guess for PSF size for fiducial images
        defaults['daoThreshold'] = 500  # The minimum brightness for fitting fiducials
        defaults['daoBaseline'] = 100   # The assumed baseline of the camera

        defaults['exportWarpedBeads'] = True #Export a warped set of bead images

        # Set the orientation of the camera to the stage: The first/second element invert X/Y if '1' the third element transposes X/Y
        defaults['cameraOrientation'] = [0,0,0]
        defaults['geoTransformEdges'] =  np.array([np.arange(0,2048,25),np.arange(0,2048,25)])  # The histogram bins for position calculating position dependent bias in warping

        defaults['colorTransforms'] = {"color":"", 'transform':[], 'type':''}   # A structure array that includes affine transforms for all color channels.
              # The color entries must match those provided in the data organization file
        return defaults


    # -------------------------------------------------------------------------
    # Generate Preprocessing Default Parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultPreprocessingParameters():
        # Generate the default parameters information for the preprocessing method
        # defaultCell = MERFISHDecoder.PreprocessingDefaultParameters()

        defaults = {}
        # Select the preprocessing method
        defaults['preprocessingMethod'] =   "highPassDecon"      # Options: {'highPassDecon', 'highPassErosion', 'highPassDeconWB'}

        # Parameters for high pass
        defaults['highPassKernelSize'] = 3          # The size of the Gaussian filter used for high pass filtering

        # Parameters for deconvolution
        defaults['deconKernel'] = fgauss2D((10,10), 2)     # The kernel to use for Lucy Richardson Deconvolution (no decon applied if empty)
        defaults['numIterDecon'] = 20

        # Parameters for GPU deconvolution: DEPRECATED FOR NOW
        # defaults['deconGPUsigma', 'nonnegative', 2}  # The sigma for the Lucy Richardson kernel for the GPU decon
        # defaults['deconGPUkernel', 'nonnegative', 10} # The kernel size for the Lucy Richardson kernel for the GPU decon

        # Parameters for erosion
        defaults['erosionElement'] = disk(1)      # The morphological structuring element used for image erosion

        return defaults

    # -------------------------------------------------------------------------
    # Generate Decoding Default Parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultDecodingParameters():
        # Generate the default parameters for the decoding process

        # Create empty cell
        defaults = {}

        # Parameters for additional preprocessing
        defaults['lowPassKernelSize'] = 1           # The size of kernel for an intial low pass average (0 indicates on averaging)
        defaults['crop'] = 40        # The number of pixels to crop from each edge of the image

        # Parameters for decoding
        defaults['decodingMethod'] = "distanceHS1"         # The method to employ for decoding
        defaults['distanceThreshold'] = 0.5176           # The distance defining the Hamming Sphere around each barcode to assign to that barcode

        # Parameters for pre-saving cuts
        defaults['minBrightness'] = 10**0              # The minimum brightness to call a pixel as a barcode
        defaults['minArea'] = 1                     # The minimum area to save a barcode
        # The connectivity matrix used to connect objects: the default is to NOT connect objects between z-planes
        defaults['connectivity'] = np.stack((np.zeros((3,3)), conndef(2, 'max'), np.zeros((3,3))))



        # Parameters for absolute coordinates
        defaults['stageOrientation'] = [1,1 ]         # The orientation of the different stage axis wrt to the camera
        defaults['pixelSize'] = 109     # Pixel size in nm/pixel

        return defaults

    # -------------------------------------------------------------------------
    # Generate Segmentation Default Parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultSegmentationParameters():
        # Generate the default parameters for the feature segmentation process

        # Create empty cell
        defaults = {}

        # Parameters for determining segmentation method
        # A method in which a seed frame is used to create required features for a watershed approach
        defaults['segmentationMethod']= 'seededWatershed'    # The size of kernel for an intial low pass average (0 indicates on averaging)

        # Parameters defining the location of image data
        defaults['watershedSeedChannel']= "DAPI"   # The frame (or frames) used to define the seed associated with each watershed
        defaults['watershedChannel']= "polyT"        # The frame (or frames) used to define the watershed.
        # Parameters for segmentation
        defaults['seedFrameFilterSize']= 5      # Size of a guassian kernal to filter frame
        defaults['seedFrameErosionKernel']= diamond(28)[9:48, 9:48]    # The kernal for erosion of the seed frame, match Matlab's strel('disk', 20)

        defaults['seedThreshold']= "adaptive"          # The threshold for the seed frames
        defaults['seedConnectionKernel']= diamond(28)[9:48, 9:48]     # The kernel for connecting nearby seeds
        defaults['seedDilationKernel']=  diamond(6)[2:11,2:11]         # The kernel for dilating seed centroids prior to watershed, match Matlab's strel('disk', 5)
        defaults['minCellSize']=  100              # The minimum number of voxels in a cell
        defaults['watershedFrameFilterSize']=5   # Size of guassian kernel to filter frame
        defaults['watershedFrameThreshold']= "adaptive"   # The brightness threshold for being in a cell
        defaults['ignoreZ']=False                # Ignore z in the segmentation process

        # Parameters for converting to real world coordinates
        defaults['boundingBox']= np.array([-100, -100, 200, 200])       # The bounding box in microns centered on the middle of the fov

        # Parameters defining stitching together features from different
        # FOV
        defaults['maxEdgeDistance']= 4           # The maximum distance between the end of an edge in one frame and that of another to be connected
        defaults['maxFeatureCentroidDistance']=5
        # Parameters for parsing of barcodes into features
        defaults['dilationSize']=0.1           # The fraction of a pixel size by which all boundaries will be expanded outwards to facilitate rapid parsing of barcodes

        # Parameters for display of results/archival of results
        defaults['saveSegmentationReports']= True   # Should segmentation reports be generated and saved?

        return defaults

    # -------------------------------------------------------------------------
    # Default summation parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultSummationParameters():
        # Generate the default parameters for the summation of raw data

        # Create empty cell
        defaults = {}

        # Parameters for summation
        defaults['areaBounds']= [0,500]                 # The lower and upper bounds on the area of individual features for calculation. The bounds are not inclusive.
        defaults['dcIndsForSummation']=  np.arange(18)        # The indices for the data channels to use for summation
        defaults['zIndForSummation']= []         # The indices associated with the z-channels for summation: REMOVE ME! NO LONGER USED!
        return defaults

    # -------------------------------------------------------------------------
    # Default smFISH molecule parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultMoleculeParameters():
        # Generate the default parameters for the fast identification of
        # individual molecules

        # Create empty cell
        defaults = {}

        # Parameters for summation
        defaults['molLowPassfilterSize']= 5   # The size of the low pass guassian filter in pixels
        defaults['molIntensityThreshold']=1000  # Intensity threshold
        defaults['molNumPixelSum']= 1        # The number of pixels to sum in each direction for the brightness of the spot
        defaults['molDataChannels']= {'RS0763':4, 'RS1199':4,'RS1040':4}      # Information on the data channels to analyze

        return defaults


    # -------------------------------------------------------------------------
    # Generate Optimization Default Parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultOptimizationParameters():
        # Generate the default parameters for the optimization process

        defaults = {}
        defaults['weightingOptimizationMethod']= "equalOnBits" # Method for optimizing image weights
        defaults['quantileTarget']=0.9              # The quantile to set to 1 for initial weighting of histograms
        defaults['areaThresh']=4                # The area threshold for barcodes to be used in optimization
        defaults['optNumFov']=50                 # The number of fov to use in the optimization process
        defaults['numIterOpt']=10     # The number of iterations to perform in the optimization of weighting
        defaults['normalizeToOne']=False   # Normalize the scale factors to 1?
        return defaults

    # -------------------------------------------------------------------------
    # Generate Default Display Parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultDisplayParameters():
        # Generate the default parameters for display

        defaults = {}
        defaults['visibleOption']= True            # Display figures as they are created: On/Off: Trur/False
        defaults['overwrite']=True     # Overwrite existing figures
        defaults['formats']= "png"                  # The figure formats to save
        defaults['useExportFig'] = False # Use the export_fig package (not always available)

        # Parameters for creating low resolution mosaics
        defaults['downSample']=10  # The value for downsampling images when creating low resolution mosaics
        defaults['mosaicZInd']=3  # The z index to select for generating mosaic indices
        return defaults

    # -------------------------------------------------------------------------
    # Generate Default Quantification Parameters
    # -------------------------------------------------------------------------
    @staticmethod
    def DefaultQuantificationParameters():
        # Generate the default parameters for quantification of data

        defaults = {}
        defaults['minimumBarcodeArea']= 4
        defaults['minimumBarcodeBrightness']=10**0.75
        defaults['minimumDistanceToFeature']=np.inf
        defaults['zSliceRange']=[]
        return defaults

    # -------------------------------------------------------------------------
    # InitializeParameters
    # -------------------------------------------------------------------------
    @staticmethod
    def InitializeParameters(**kwargs):
        # Create a default parameters structure and set fields in this structure if specified
        #
        # parameters = self.InitializeParameters() # Return all defaults
        # parameters = self.InitializeParameters('name', value, ...)
        # [~, defaultParameters] = self.InitializeParameters() # Return a cell array containing each parameters descriptor

        # Define default parameter sets
        defaultParameters: Dict[str, Any] = {}
        defaultParameters['warp'] = MERFISHDecoder.DefaultWarpParameters()
        defaultParameters['preprocess'] = MERFISHDecoder.DefaultPreprocessingParameters()
        defaultParameters['decoding'] = MERFISHDecoder.DefaultDecodingParameters()
        defaultParameters['optimization'] = MERFISHDecoder.DefaultOptimizationParameters()
        defaultParameters['display'] = MERFISHDecoder.DefaultDisplayParameters()
        defaultParameters['segmentation'] = MERFISHDecoder.DefaultSegmentationParameters()
        defaultParameters['quantification'] = MERFISHDecoder.DefaultQuantificationParameters()
        defaultParameters['summation'] = MERFISHDecoder.DefaultSummationParameters()
        defaultParameters['molecules'] = MERFISHDecoder.DefaultMoleculeParameters()

        # Loop over parameter sets
        parameters = {}
        for p in defaultParameters:
            # Store parameters
            localDefaults = defaultParameters[p]
            parameters[p] = {}
            for k_i in localDefaults:
                if k_i in kwargs:
                    parameters[p][k_i] = kwargs[k_i]
                else:
                    parameters[p][k_i] = localDefaults[k_i]

        return parameters,defaultParameters

    # -------------------------------------------------------------------------
    # Combine and generate report for affine transforms
    # -------------------------------------------------------------------------
    @staticmethod
    def CheckTiffStack(tiffFileName, expectedNumFrames):
        # Check the status of a tiff stack -- is it complete?

        isCorrupt = False  # Assume it is not corrupt
        try: # See if one can open information on the tiff stack
            # Check for validity
            # tiffInfo = imfinfo(tiffFileName) ## MATLAB code
            tiffInfo = tiffimginfo(tiffFileName)
            # Check to see if the number of frames are correct
            if len(tiffInfo) < expectedNumFrames:
                isCorrupt = True # If it can be opened, but there are not enough frames, it is corrupt
        except:
            isCorrupt = True # If the info file cannot be opened, it is corrupt

        return  isCorrupt

    # -------------------------------------------------------------------------
    # Decode an image
    # -------------------------------------------------------------------------
    @staticmethod
    def DecodePixels(imageStack, scaleFactors, decodingVectors, distanceThreshold):
        # ------------------------------------------------------------------------
        # [decodedImage, localMagnitude, imageStack] = self.DecodePixels(imagePath, scaleFactors, decodingVectors, distanceThreshold)
        # This function takes an image stack (width, height, z, numFrames) and
        # decodes it by comparing the individual normalized pixel vectors
        # to the normalized vectors provided in the decodingVectors matrix. Pixels
        # outside of a given distance threshold to the nearest barcode are
        # discarded.
        #

        # Reshape imageStack to create pixel traces
        if np.ndim(imageStack) == 3: # Handle case of no z position
            imageHeight = imageStack.shape[0]
            imageWidth = imageStack.shape[1]
            stackLength = imageStack.shape[2]
            numZPos = []
            pixelTraces = imageStack.reshape((imageHeight*imageWidth,stackLength),order="F")
        else:
            imageHeight = imageStack.shape[0]
            imageWidth =imageStack.shape[1]
            numZPos = imageStack.shape[2]
            stackLength = imageStack.shape[3]
            pixelTraces = imageStack.reshape((imageHeight*imageWidth*numZPos,stackLength),order="F")

        # Determine number of pixels
        numPixels = pixelTraces.shape[0]

        # Type cast
        pixelTraces = pixelTraces.astype(np.float)

        # Equalize bit brightness distributions via provided scale factors
        pixelTraces = pixelTraces/np.tile(scaleFactors, (numPixels,1))

        # Calculate the magnitude to normalize
        localMagnitude = np.sqrt(np.sum(pixelTraces*pixelTraces, axis=1))

        # Create the pixel traces
        goodInds = localMagnitude > 0 # Handle the divide by zero case
        pixelTraces[goodInds,:] = pixelTraces[goodInds,:]/np.tile(localMagnitude[goodInds].reshape(-1,1), (1,pixelTraces.shape[1]))

        # Find nearest neighbor barcodes (on the N-dimensional unit sphere)
        barcodeID, D = knnsearch2d(decodingVectors, pixelTraces)
        barcodeID = barcodeID[0] + 1
        D = D[0]
        # Associate pixels to barcodes for which they are within a N-1 sphere
        # defined by the distanceThreshold
        exactInds = D <= distanceThreshold

        # Decode image
        if numZPos==1:
            decodedImage = np.zeros((imageHeight*imageWidth,))
            decodedImage[exactInds] = barcodeID[exactInds]
            decodedImage = decodedImage.reshape((imageHeight, imageWidth), order="F")
        else:
            decodedImage = np.zeros((imageHeight * imageWidth * numZPos,))
            decodedImage[exactInds] = barcodeID[exactInds]
            decodedImage = decodedImage.reshape((imageHeight,imageWidth,numZPos), order="F")

        return decodedImage, localMagnitude, pixelTraces, D




