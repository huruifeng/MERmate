import uuid

from merfish.analysis.image_data import BuildImageDataStructures, TransformImageData
from merfish.analysis.image_data import AlignFiducials
from merfish.analysis.image_data import TransferInfoFileFields
from merfish.analysis.image_data import ReadMasterMoleculeList
from merfish.reports.reports import GenerateCompositeImage, GenerateOnBitHistograms
from merfish.analysis.words import CreateWords, DecodeWords

from merfish.codes.codebook_process import CodebookToMap,SECDEDCorrectableWords
from utils.funcs import *

def AnalyzeMERFISH(dataPath, **kwargs): ## return [words, totalImageData, totalFiducialData, parameters]
    # ------------------------------------------------------------------------
    # [words, parameters] = AnalyzeMERFISH(dataPath, varargin)
    # This function analyzes a series of raw conventional images in the
    #   specified directory and creates a words structure which represents all
    #   of the identified words in the data.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    #--------------------------------------------------------------------------
    # Outputs
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    # Parameters for parsing file names
    parameters['imageTag'] = 'STORM'        # Base tag for all images
    parameters['imageMListType'] = 'alist'  # Flag for image mlist
    parameters['fiducialMListType'] ='list' # Flag for fiducial mlists
    
    # Parameters for parsing file names
    parameters['fileExt'] = 'bin'           # Delimiters for bin files
    # Labels for fields in image name
    parameters['fieldNames']=['movieType', 'hybNum', 'cellNum', 'isFiducial', 'binType']


    parameters['fieldConv'] = [str,int,int, lambda x:x=='c2', str] # Conversion functions for fields in image name

    parameters['appendExtraFields'] = True   # How to handle names that don't match this pattern
    
    # Parameters for the cells to analyze
    parameters['cellsToAnalyze']= []        # List of cell/FOV ids to analyze
    
    # Parameters on the number of hybridizations
    parameters['numHybs']=16         # Number of hybridizations
    parameters['bitOrder'] = list(range(16))          # Order of bits
    
    # Parameters for fiducial tracking
    parameters['maxD'] = 8             # Maximum distance for fiducial tracking
    parameters['fiducialFrame'] = 1    # Reference frame for fiducial markers
    parameters['fiducialWarp2Hyb1']= False
    
    # Parameters for constructing words from spots
    parameters['maxDtoCentroid'] = 1   # Distance between spots in different rounds
    
    # Parameters for decoding words
    parameters['codebookPath'] = ''       # Path to codebook
    parameters['codebook'] = {}             # Codebook structure
    parameters['exactMap']={}                # containers.Map for decoding exact matches
    parameters['correctableMap'] = {}          # containers.Map for decoding correctable matches
    parameters['errCorrFunc'] = SECDEDCorrectableWords        # Error correction function
    parameters['keyType'] =  "binStr"     # print type for binary word, e.g. binary or decimal
    
    # Parameters for progress reports and intermediate figures
    parameters['savePath'] = ''            # Path to save incidental figures
    parameters['reportsToGenerate'] = {}       # List of flags for generating different reports

    parameters['showPlots']= True
    parameters['showCorrPlots']= True
    parameters["troubleshoot"] = True

    parameters['verbose']=True          # print progress?
    parameters['printedUpdates']= True    # print additional forms of progress?

    ## update the paramteres provided by users.
    for i in kwargs:
        parameters[i] = kwargs[i]


    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not os.path.exists(dataPath):
        error('[Error]:invalidArguments-A valid data path is required.')

    # -------------------------------------------------------------------------
    # Provide overview of analysis
    # -------------------------------------------------------------------------
    parameterFieldsToprint = ['codebookPath', 'imageTag', 'imageMListType',
                              'fiducialMListType', 'fileExt', 'numHybs', 'bitOrder', 
                              'maxD', 'wordConstMethod', 'savePath'
                              ]
    
    if parameters["printedUpdates"]:
        print('--------------------------------------------------------------')
        print('Analyzing Multiplexed FISH Data')
        print('--------------------------------------------------------------')
        print('Analysis parameters')
        for i in parameterFieldsToprint:
            print('  ->',i, ':',parameters[i])
    
    # -------------------------------------------------------------------------
    # Find data for analysis
    # -------------------------------------------------------------------------
    if parameters["printedUpdates"]:
        print('--------------------------------------------------------------')
        print('Finding data in', dataPath)
    
    foundFiles = BuildImageDataStructures(dataPath, **parameters)[0]
    numCells = max([file_i["cellNum"] for file_i in foundFiles])
    
    if parameters["printedUpdates"]:
        print('    Found ',numCells,' cells')
    
    if numCells == 0:
        error('[Error]: No valid cells found')
    
    # -------------------------------------------------------------------------
    # Load codebook and generate maps
    # -------------------------------------------------------------------------
    if len(parameters["codebook"]) <=0 and len(parameters["codebookPath"]) > 0:
        parameters["codebook"] = fastaread(parameters["codebookPath"])
    
    if len(parameters["codebook"]) > 0:
        parameters["exactMap"] = CodebookToMap(parameters["codebook"],keyType = parameters["keyType"])
        if callable(parameters["errCorrFunc"]):
            parameters["correctableMap"] = CodebookToMap(parameters["codebook"],
                                                         keyType=parameters["keyType"],
                                                         errCorrFunc= parameters["errCorrFunc"],
                                                         mapContents="correctable")
    
    if parameters["printedUpdates"]:
        print('--------------------------------------------------------------')
        if len(parameters["exactMap"]) == 0:
            print('No codebook provided. Found words will not be decoded.')
        else:
            print('exactMap words:',len(parameters["exactMap"]))

        if len(parameters["correctableMap"]) == 0:
            print('No error correction will be applied.')
        else:
            print('correctableMap words:',len(parameters["correctableMap"]))
    
    # -------------------------------------------------------------------------
    # Prepare loop variables
    # -------------------------------------------------------------------------
    words = []
    totalImageData = []
    totalFiducialData = []
    
    # -------------------------------------------------------------------------
    # Determine cells to analyze
    # -------------------------------------------------------------------------
    if len(parameters["cellsToAnalyze"]) == 0:
        cellIDs = list(range(numCells))
    else:
        cellIDs = [c_i for c_i in parameters["cellsToAnalyze"] if c_i>= 0 and c_i<numCells]
    
    # -------------------------------------------------------------------------
    # Loop over all cells
    # -------------------------------------------------------------------------
    for i in cellIDs:
        parameters['cellIDs'] = i+1   ## cellIDs start from 1
        # ---------------------------------------------------------------------
        # print cell number
        # ---------------------------------------------------------------------
        if parameters["printedUpdates"]:
            print('--------------------------------------------------------------')
            print('Analyzing data for cell ', i+1, ' of ',numCells)
    
        # ---------------------------------------------------------------------
        # Identify all files for this cell
        # ---------------------------------------------------------------------
        movieType_arr = np.array([k["movieType"]==parameters["imageTag"] for k in foundFiles])
        binType_arr_img = np.array([k["binType"]==parameters["imageMListType"] for k in foundFiles])
        binType_arr_fid = np.array([k["binType"] == parameters["fiducialMListType"] for k in foundFiles])
        cellNum_arr = np.array([k["cellNum"]== (cellIDs[i]+1) for k in foundFiles])
        isFiducial_arr = np.array([k["isFiducial"] for k in foundFiles])

        imageData_bool =  movieType_arr & binType_arr_img & cellNum_arr & np.logical_not(isFiducial_arr)
        fiducialData_bool = movieType_arr & binType_arr_fid & cellNum_arr & isFiducial_arr

        foundFiles = np.array(foundFiles)
        imageData = foundFiles[imageData_bool]
        fiducialData = foundFiles[fiducialData_bool]
    
        if parameters["printedUpdates"]:
            print('    Found',len(imageData), 'image files')
            if parameters["verbose"]:
                for j in range(len(imageData)):
                    print('       ',imageData[j]["filePath"])
        if parameters["printedUpdates"]:
            print('    Found', len(fiducialData), 'fiducial files')
            if parameters["verbose"]:
                for j in range(len(imageData)):
                    print('       ', fiducialData[j]["filePath"])

        # ---------------------------------------------------------------------
        # Cross checks on found files
        # ---------------------------------------------------------------------
        if len(imageData) != len(fiducialData):
            print('[Warning]:AnalyzeMultiFISH- Cell ',i, 'does not have equal numbers of data and fiducial images')
            continue

        if len(imageData) < parameters["numHybs"]:
            print('[Warning]:AnalyzeMultiFISH - Cell ',i, ' has fewer data images than hybs')
            continue

        if len(fiducialData) < parameters["numHybs"]:
            print('[Warning]:AnalyzeMultiFISH - Cell ',i, ' has fewer fiducial images than hybs')
            continue
    
        # ---------------------------------------------------------------------
        # Sort files based on hyb number: Almost certainly not necessary
        # ---------------------------------------------------------------------
        sidx = np.argsort([iD["hybNum"] for iD in imageData])
        imageData = imageData[sidx]
        sidx = np.argsort([fD["hybNum"] for fD in fiducialData])
        fiducialData = fiducialData[sidx]
    
        # ---------------------------------------------------------------------
        # Generate and append unique IDs for each image and fiducial data set
        # ---------------------------------------------------------------------
        for j in imageData:
            j["uID"] = str(uuid.uuid4())
        for j in fiducialData:
            j["uID"] = str(uuid.uuid4())
        # ---------------------------------------------------------------------
        # Load and transfer information on the corresponding dax
        # ---------------------------------------------------------------------
        imageData = TransferInfoFileFields(imageData, **parameters)
        # fiducialData = TransferInfoFileFields(fiducialData, 'parameters', parameters)

        # ---------------------------------------------------------------------
        # Generate a measure of focus lock quality for all images
        # ---------------------------------------------------------------------
        #imageData = GenerateFocusLockQuality(imageData, 'parameters', parameters)

        # ---------------------------------------------------------------------
        # Load Molecule Lists and Fiducial Positions
        # ---------------------------------------------------------------------
        if parameters["printedUpdates"] and parameters["verbose"]:
            print('--------------------------------------------------------------')
            print('Loading molecules lists')
        for j in range(parameters["numHybs"]):
            imageData[j]["mList"]= ReadMasterMoleculeList(imageData[j]["filePath"],
                                                          compact=True, transpose=True,verbose = False)
            fiducialData[j]["mList"] = ReadMasterMoleculeList(fiducialData[j]["filePath"],
                                                           compact = True, transpose=True, verbose=False)

            if parameters["printedUpdates"] and parameters["verbose"]:
                print('    ',imageData[j]["name"], ':', len(imageData[j]["mList"]), ' molecules')
                print('    ',fiducialData[j]["name"],':', len(fiducialData[j]["mList"]), ' beads')
        # ---------------------------------------------------------------------
        # Create Tiled Image (if desired)
        # ---------------------------------------------------------------------
        #GenerateTiledImage(imageData, 'parameters', parameters)

        # ---------------------------------------------------------------------
        # Add Transforms to Fiducial Data
        # ---------------------------------------------------------------------
        fiducialData, parameters = AlignFiducials(fiducialData, **parameters)

        # ---------------------------------------------------------------------
        # Transform Image Data and Transfer Fiducial Data
        # ---------------------------------------------------------------------
        imageData, parameters = TransformImageData(imageData,fiducialData,**parameters)

        # -------------------------------------------------------------------------
        # Create Words from Spots
        # -------------------------------------------------------------------------
        wordsByCell, parameters = CreateWords(imageData, **parameters)

        # -------------------------------------------------------------------------
        # Decode words
        # -------------------------------------------------------------------------
        if len(parameters["codebook"]) >0:
           [wordsByCell, parameters] = DecodeWords(wordsByCell, **parameters)
           if parameters["printedUpdates"]:
                    print('    Found', np.sum([w_i["isExactMatch"] for w_i in wordsByCell]), 'exact matches')
                    print('    Found', np.sum([w_i["isCorrectedMatch"] for w_i in wordsByCell]), 'corrected matches')


        if parameters["printedUpdates"] and parameters["verbose"]:
            print('--------------------------------------------------------------')
            print('Creating composite image with words...')
        # ---------------------------------------------------------------------
        # Create Composite image with words
        # ---------------------------------------------------------------------
        GenerateCompositeImage(wordsByCell, imageData, **parameters)

        if parameters["printedUpdates"] and parameters["verbose"]:
            print('--------------------------------------------------------------')
            print('Creating and saving cell by cell on bit histogram...')
        # ---------------------------------------------------------------------
        # Create Cell By Cell On Bit Histogram
        # ---------------------------------------------------------------------
        GenerateOnBitHistograms(wordsByCell, **parameters, numOnBitsHistAllCells=False)

        # -------------------------------------------------------------------------
        # Append Words and imageData
        # -------------------------------------------------------------------------
        words = words + wordsByCell
        totalImageData = totalImageData + list(imageData)
        totalFiducialData = totalFiducialData + list(fiducialData)

    if parameters["printedUpdates"]:
        print('--------------------------------------------------------------')
        print('Completed Multiplexed FISH Analysis! (^_^)')

    return [words,totalImageData,totalFiducialData,parameters]
