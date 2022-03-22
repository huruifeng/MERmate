import matplotlib
import numpy as np
import uuid
import copy

from merfish.analysis.image_data import CreateMoleculeList
from utils.funcs import knnsearch2d, error
from utils.misc import bi2de


def CreateWords(imageData,**kwargs):
    # ------------------------------------------------------------------------
    # words = CreateWords(imageData,varargin)
    # This generates a series of word structures from an imageData structure by
    #   first finding connected objects across hyb images and then using the
    #   centroids of these objects to construct actual words.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # imageData/A structure array with elements equal to the number of
    #   images to align. For information on the fields of this structure see
    #   CreateImageData
    #--------------------------------------------------------------------------
    # Outputs
    # words/A structure array of words with the following fields
    #   --
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #    'binSize' -- size of bins in nm
    #    'minDotPerBin' -- min number of localizations to call a bin occupied
    #    'minLocsPerDot' -- min number of localization in all bins assigned to a cluster to be called an mRNA
    #    'minArea' -- min area in bins to be called a cluster of localization
    #    'maxArea' -- max area in bins to be called a cluster of localization
    #    'maxDtoCentroid' -- the maximum distance from a centroid to any
    #       individual hybridization
    #--------------------------------------------------------------------------
    # Based on FindMRNA.m and AssignConvSpotsToCentroids.m
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['wordConstMethod'] = 'perLocalization'  #options: {'commonCentroid','perLocalization'}
    parameters['binSize'] = 0.25
    parameters['minDotPerBin'] = 1
    parameters['minLocsPerDot'] = 1
    parameters['minArea'] = 0
    parameters['maxArea'] = 10
    parameters['showPlots'] = True
    parameters['clusterFig']=[]
    parameters['histFig'] = []
    parameters['imageSize'] = [256,256]
    parameters['verbose'] = True
    parameters['printedUpdates']= True
    parameters['numHybs'] = 16

    # Report parameters
    parameters['reportsToGenerate'] = {}
    parameters['useSubFolderForCellReport'] = True
    parameters['overwrite'] = True
    parameters['figFormats'] = 'png'

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    # -------------------------------------------------------------------------
    # Pull out mLists
    # -------------------------------------------------------------------------
    mLists = [iD["mList"] for iD in imageData]
    mListFields = list(mLists[0][0].keys())

    # -------------------------------------------------------------------------
    # Clear words
    # -------------------------------------------------------------------------
    words = []

    # -------------------------------------------------------------------------
    # Display status
    # -------------------------------------------------------------------------
    if parameters["printedUpdates"] and parameters["verbose"]:
        print('--------------------------------------------------------------')
        print('Creating words with method: ', parameters["wordConstMethod"])

    # -------------------------------------------------------------------------
    # Select word construction method
    # -------------------------------------------------------------------------
    if parameters["wordConstMethod"] ==  'commonCentroid':
        print("[INFO]: Method <commonCentroid> is under construction, using <perLocalization>.")
        parameters["wordConstMethod"] == 'perLocalization'
        # # -------------------------------------------------------------------------
        # # Identify putative words by finding connected regions in all images
        # # -------------------------------------------------------------------------
        # # Create image parameters
        # bin_size = parameters["binSize"]
        # end_point1 = parameters["imageSize"][0]
        # end_point2 = parameters["imageSize"][1]
        #
        # xall = []
        # yall = []
        # for mL_i in mLists:
        #     xall += [mL_ii["xc"] for mL_ii in mL_i]
        #     yall += [mL_ii["yc"] for mL_ii in mL_i]
        #
        # edges = [np.arange(bin_size, end_point1 + 0.2 * bin_size, bin_size),
        #           np.arange(bin_size, end_point2 + 0.2 * bin_size, bin_size)]  ## to include the endpoints:start, stop+0.5*step
        # centers = [[e_i - bin_size * 0.5 for e_i in edges[0]] + [edges[0][-1] + 0.5 * bin_size],
        #             [e_i - bin_size for e_i in edges[1]] + [edges[1][-1] + 0.5 * bin_size]]
        #
        # M,_,_ = np.histogram2d(yall,xall,bins = centers)
        #
        # # -------------------------------------------------------------------------
        # # Find connected regions in combined images
        # # -------------------------------------------------------------------------
        # P = measure.regionprops((M>=parameters["minDotPerBin"]).astype(int),M)[0]
        #
        # # -------------------------------------------------------------------------
        # # Cut regions
        # # -------------------------------------------------------------------------
        # clusterLocs = [P["PixelValues"]]
        # clusterarea = [p_i["Area"] for p_i in P]
        # goodClusters = (clusterLocs>parameters["minLocsPerDot"]) & \
        #                (clusterarea > parameters["minArea"]) & \
        #                (clusterarea < parameters["maxArea"])
        #
        # # -------------------------------------------------------------------------
        # # Identify Potential Centroids
        # # -------------------------------------------------------------------------
        # putativeWordCentroids = np.concatenate(P["Centroid"]*parameters["binSize"])
        # putativeWordCentroids = putativeWordCentroids[goodClusters,:]
        #
        # # -------------------------------------------------------------------------
        # # Calculate distances
        # # -------------------------------------------------------------------------
        # idx = np.zeros((len(putativeWordCentroids), len(imageData)))
        # d = np.empty((len(putativeWordCentroids), len(imageData))) # If empty, no distances are triggered
        # d[:] = np.inf
        # for i in range(len(imageData)):
        #     if imageData[i]["mList"]["xc"]: # Handle lost frames
        #         [idx[:,i], d[:,i]] = knnsearch2d(imageData[i]["mList"]["xc"],imageData[i]["mList"]["yc"],putativeWordCentroids)
        #
        # # -------------------------------------------------------------------------
        # # Allocate Word Structure
        # # -------------------------------------------------------------------------
        # words = CreateWordsStructure(len(putativeWordCentroids), parameters["numHybs"])
        #
        # # -------------------------------------------------------------------------
        # # Build word properties specific to method
        # # -------------------------------------------------------------------------
        # for i in range(len(putativeWordCentroids)):
        #     words[i]["measuredCodeword"] = d[i,:] <= parameters["maxDtoCentroid"]
        #     words[i]["mListInds"] = idx[i, words[i]["measuredCodeword"]]
        #     words[i]["wordCentroidX"] = putativeWordCentroids[i,0]
        #     words[i]["wordCentroidY"] = putativeWordCentroids[i,1]

    if parameters["wordConstMethod"] == 'perLocalization':
        minPhotsPerStain = 1
        numHybes = len(imageData)
        # record the positions of all spots in all hybes
        spotPostionsPerHybe = []
        for h in range(numHybes):
            spotPostionsPerHybe.append(np.asarray([[mL_i["xc"],mL_i["yc"]] for mL_i in imageData[h]["mList"]]))
        putativeWordCentroids = np.concatenate(spotPostionsPerHybe, axis=0)

        if np.all([len(x) > 0 for x in spotPostionsPerHybe]): # Don't build words for cells without all mLists
            # Assign localizations to mRNA centroids
            wordsDetected = np.zeros((len(putativeWordCentroids),numHybes))
            brightnessPerSpot = np.empty((len(putativeWordCentroids),numHybes))
            brightnessPerSpot[:] = np.nan
            xPerSpot = np.empty((len(putativeWordCentroids),numHybes))
            xPerSpot[:] = np.nan
            yPerSpot = np.empty((len(putativeWordCentroids), numHybes))
            yPerSpot[:] = np.nan
            idxPerSpot = np.empty((len(putativeWordCentroids), numHybes))
            idxPerSpot[:] = np.nan

            for h in range(numHybes):
                xyc = np.asarray([[mL_i["xc"], mL_i["yc"]] for mL_i in imageData[h]["mList"]])
                [idx,di] = knnsearch2d(xyc,putativeWordCentroids,1)
                brightnessPerSpot[:,h] = np.asarray([iD_i["a"] for iD_i in imageData[h]["mList"]])[idx[0]]
                brightnessPerSpot[di[0] >= parameters["maxDtoCentroid"],h] = 0
                validIdx = brightnessPerSpot[:,h] > minPhotsPerStain
                wordsDetected[:,h] = validIdx.astype(int)
                xPerSpot[:,h] = np.asarray([iD_i["xc"] for iD_i in imageData[h]["mList"]])[idx[0]]
                xPerSpot[~validIdx,h] = np.nan
                yPerSpot[:,h] = np.asarray([iD_i["yc"] for iD_i in imageData[h]["mList"]])[idx[0]]
                yPerSpot[~validIdx,h] = np.nan
                idxPerSpot[:,h] = idx[0]
                idxPerSpot[~validIdx,h] = np.nan

            # compute centroids of all codewords
            meanX = np.nanmean(xPerSpot,1)
            meanY = np.nanmean(yPerSpot,1)
            wordLocations = np.vstack((meanX,meanY)).T

            # Remove redundantly recorded codewords ID'd by shared centroids.
            [_,uniqueIdx] = np.unique(wordLocations,return_index=True,return_inverse=False,axis=0)
            uniqueIdx = np.sort(uniqueIdx)
            wordLocations = wordLocations[uniqueIdx]

            wordsDetected = wordsDetected[uniqueIdx,:]
            idxPerSpot = idxPerSpot[uniqueIdx,:]


            # -------------------------------------------------------------------------
            # Allocate Word Structure
            # -------------------------------------------------------------------------
            numWords = wordsDetected.shape[0]
            words = CreateWordsStructure(numWords, parameters["numHybs"])

            # -------------------------------------------------------------------------
            # Build word properties specific to method
            # -------------------------------------------------------------------------
            for i in range(numWords):
                words[i]["measuredCodeword"] = wordsDetected[i, :]
                words[i]["mListInds"] = idxPerSpot[i, ~np.isnan( idxPerSpot[i,:] )]
                words[i]["wordCentroidX"] = wordLocations[i, 0]
                words[i]["wordCentroidY"] = wordLocations[i, 1]
        else:
            words = CreateWordsStructure(0, parameters["numHybs"])
    else:
        error('[Error]:CreateWords - Unknown word construction method')

    # -------------------------------------------------------------------------
    # Fill out word structure
    # -------------------------------------------------------------------------
    for i in range(len(words)):
        # Transfer basic names and experiment info
        words[i]["imageNames"] = [iD["name"] for iD in imageData]
        words[i]["imagePaths"] = [iD["filePath"] for iD in imageData]
        words[i]["bitOrder"] = parameters["bitOrder"]
        words[i]["numHyb"] = parameters["numHybs"]
        words[i]["cellID"] = imageData[0]["cellNum"]
        words[i]["wordNumInCell"] =i

        # Transfer image position
        words[i]["imageX"] = imageData[0]["Stage_X"]
        words[i]["imageY"] = imageData[0]["Stage_Y"]

        # Generate and transfer unique IDs
        words[i]["uID"] = str(uuid.uuid4())
        words[i]["imageUIDs"] = [iD["uID"] for iD in imageData]
        words[i]["fiducialUIDs"] = [iD["fidUID"] for iD in imageData]

        # Record properties of identification
        words[i]["hasFiducialError"] = [iD["hasFiducialError"] for iD in imageData]

        # Determine measured and actual codeword
        words[i]["codeword"] = words[i]["measuredCodeword"][words[i]["bitOrder"]]
        words[i]["intCodeword"] = bi2de(words[i]["codeword"],direction="left") #Save an integer version of the codeword
        words[i]["numOnBits"] = np.sum(words[i]["measuredCodeword"])

        # Save On Bit Indicies
        words[i]["measuredOnBits"] = np.nonzero(words[i]["measuredCodeword"])[0]
        words[i]["onBits"] = np.nonzero(words[i]["codeword"])[0]
        words[i]["paddedCellID"] = np.ones((int(words[i]["numOnBits"]),),'int32')*words[i]["cellID"]

        # Transfer mList properties
        for k in range(len(words[i]["measuredOnBits"])):
            listID = words[i]["measuredOnBits"][k]
            moleculeID = int(words[i]["mListInds"][k])
            for fieldID in range(len(mListFields)):
                words[i][mListFields[fieldID]][listID] = [mL_i[mListFields[fieldID]] for mL_i in mLists[listID]][moleculeID]

        # Add fields for identifying words/genes
        words[i]["geneName"] = ''
        words[i]["isExactMatch"] = False
        words[i]["isCorrectedMatch"] = False

        # Add focus lock quality data
        words[i]["focusLockQuality"] = [iD_i["focusLockQuality"] for iD_i in imageData]

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["printedUpdates"]:
        print('    Reconstructed ',len(words), ' words')
        if parameters["verbose"]:
            edges = np.arange(1, parameters["numHybs"] + 0.2 * 1, 1)  ## to include the endpoints:start, stop+0.5*step
            centers = [e_i - 1 * 0.5 for e_i in edges] + [edges[-1] + 0.5 * 1]
            [n_x, e_x] = np.histogram([w_i["numOnBits"] for w_i in words], bins=centers)
            for j in range(parameters["numHybs"]):
                print(f'        {n_x[j]:4d} words with {int(edges[j]):3d} on bits')

    return words, parameters


def DecodeWords(words, **kwargs):
    # ------------------------------------------------------------------------
    # words = DecodeWords(words, exactMap, correctableMap, varargin)
    # This function decodes words based on the maps provided in the
    #   containers.Map objects, exactMap and correctableMap.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # words/A structure array of found words. These structures must contain the
    #   following fields:
    #   --codeword: A logical array containing the desired word.
    # exactMap/A containers.Map object containing keys and values corresponding
    #   to 'correct' codewords.  If only correctable matches are desired, then
    #   [] can be passed for this argument.
    # correctableMap/A containers.Map object containing keys and values
    #   corresponding to codewords that are not exact matches to entries in the
    #   codebook but which can be associated with a codeword via some method,
    #   e.g. an error correcting code.  If only exact matches are desired, then
    #   [] can be passed for this argument.
    #--------------------------------------------------------------------------
    # Outputs
    # words/The same structure array with the addition of the following fields
    #   --geneName: The string entry in the codebookMap corresponding to
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # --keyType ('int', 'binStr'): A string specifying the type of the key for
    #   the providedcontainers.Map
    #--------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['keyType'] = 'binStr' # The type of key for the containers.Map

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    if "correctableMap" not in parameters or "exactMap" not in parameters:
        error('[Error]:invalidArguments - exactMap and correctableMap are required')
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------

    if not isinstance(words, list) or ('codeword' not in words[0]):
        error('[Error]:invalidArguments - The first argument must be a word list or array')

    # -------------------------------------------------------------------------
    # Determine Key Conversion Function
    # -------------------------------------------------------------------------
    list2str = lambda x: "".join(x) # Useful short hand for removing whitespace
    if parameters["keyType"] == 'int':
            keyConv= lambda x: bi2de(list2str(x))
    elif parameters["keyType"]=='binStr':
            keyConv = lambda x: list2str(x)

    # -------------------------------------------------------------------------
    # Decode words
    # -------------------------------------------------------------------------
    exactMap = parameters["exactMap"]
    correctableMap = parameters["correctableMap"]
    for w_i in words:
        # ---------------------------------------------------------------------
        # Decode exact matches
        # ---------------------------------------------------------------------
        if len(exactMap) > 0:
            try:
                w_i["geneName"] = exactMap[keyConv([str(int(c_i)) for c_i in w_i["codeword"]])]
                w_i["isExactMatch"] = True
            except Exception as e:
                # print(e)
                pass

        # ---------------------------------------------------------------------
        # Decode exact matches
        # ---------------------------------------------------------------------
        if len(correctableMap)>0:
            try:
                w_i["geneName"] = correctableMap[keyConv([str(int(c_i)) for c_i in w_i["codeword"]])]
                w_i["isCorrectedMatch"] = True
            except Exception as e:
                # print(e)
                pass

    return words, parameters



def CreateWordsStructure(numElements, numHybs):
    # ------------------------------------------------------------------------
    # words = CreateWordsStructure(numElements, numHybs)
    # This function creates an array of empty word structures.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    #   numElements/int. The number of elements to create.
    #   numHybs/int. The number of hybs in the given words (used to preallocate
    #   memory). 
    #--------------------------------------------------------------------------
    # Outputs
    #   words/structure array. This array contains the following fields:
    #       ## Codeword properties
    #       --uID: A unique string for each word. Useful for archival and
    #       indexing
    #       --codeword: A logical array specifying the word in the bit order
    #       corresponding to the codebook.
    #       --measuredCodeword: A logical array specifing the value of each bit
    #       in the order in which they were measured.
    #       --intCodeword: The unsigned integer corresponding to the codeword
    #       field
    #       
    #       ## Decoded word properties
    #       --geneName: The name of the corresponding word in the codebook
    #       --isExactMatch: A boolean which is determined if the codeword
    #       corresponding to the geneName is an exact match to the measured
    #       codeword.
    #       --isCorrectedMatch: A boolean which represents if the geneName was
    #       determined using some error correction
    #
    #       ## Covnenient shorthand properties of the codewords
    #       --numOnBits: The number of on bits in the measured codeword
    #       --onBits: The indices of the on bits in the order in the codebook
    #       --measuredOnBits: The indicites of the on bits in the order in
    #       which they were measured.
    #
    #       ## Image properties
    #       --imageNames: A cell array of the names of the image files for each
    #       of the measured images, in the measurement order
    #       --imagePaths: Paths to these images
    #       --imageUIDs: A cell array of unique IDs generated for each image
    #       (see file structures)
    #       --imageX: The x position of the image in um
    #       --imageY: The y position of the image in um
    #       
    #       ## Properties of the codeword in the cell
    #       --wordCentroidX: X position of the word in the warp reference frame
    #       in pixels
    #       --wordCentroidY: Y position
    #       --cellID: The cell/FOV number from the movies/images used to
    #       generate this word
    #       --wordNumInCell: The number of the generated word in the FOV/cell,
    #       e.g. 1st, 2nd, 3rd
    #       --paddedCellID: An array of the cellID equal in length to the
    #       number of on bits. Useful for indexing from word arrays.
    #
    #       ## Fiducial alignment/warp properties
    #       --fiducialUIDs: The unique ID strings for each of the images used
    #       to generate the fiducial warps
    #       --hasFiducialError: A logical array specifying whether or not each
    #       hyb had a fiducial error.  In the order in which the hybs were
    #       measured.
    #       
    #       ## Focus lock properties
    #       --focusLockQuality: A number determining the quality of the focus
    #       for each hyb image.  
    #
    #       ## Molecule list properties
    #       Words also contain all of the fields found in molecule lists. See
    #       CreateMoleculeList and the analysis software, e.g. DaoSTORM, for
    #       more details. 
    
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # -- None.
    #--------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if numElements < 0 or numHybs < 0:
        error('[Error]:invalidArguments - Invalid values for numElements and numHybs')
    
    # -------------------------------------------------------------------------
    # Define fields and defaultProperties
    # -------------------------------------------------------------------------
    defaultWord = {}
    
    # Codeword values and properties
    defaultWord['uID'] = ''
    defaultWord['codeword']=[False] * numHybs
    defaultWord['measuredCodeword']= []
    defaultWord['intCodeword'] = np.nan
    
    defaultWord['geneName'] = ''
    defaultWord['isExactMatch'] = False
    defaultWord['isCorrectedMatch'] = False
    defaultWord['numOnBits'] = 0
    defaultWord['onBits']=[np.nan]*numHybs
    defaultWord['measuredOnBits']=[np.nan]*numHybs
    
    # Measurement properties
    defaultWord['bitOrder']=list(range(numHybs))
    defaultWord['numHyb']=np.nan
    defaultWord['imageNames']=[[]]*numHybs
    defaultWord['imagePaths']=[[]]*numHybs
    defaultWord['imageUIDs']=[[]]*numHybs
    defaultWord['imageX']=np.nan
    defaultWord['imageY']=np.nan
    
    # Properties of the codeword in the cell
    defaultWord['wordCentroidX']=np.nan
    defaultWord['wordCentroidY']=np.nan
    defaultWord['cellID']=np.nan
    defaultWord['wordNumInCell']=np.nan
    
    # Fiducial alignment/warp properties
    defaultWord['fiducialUIDs']=[[]]*numHybs
    defaultWord['hasFiducialError']=[False]*numHybs
    defaultWord['paddedCellID']=[np.nan]*numHybs
    
    # Focus Lock Properties
    defaultWord['focusLockQuality']=[np.nan] * numHybs
    
    # Molecule properties
    defaultWord['mListInds']=[np.nan] * numHybs
    
    # -------------------------------------------------------------------------
    # Generate fields and types from molecules lists
    # -------------------------------------------------------------------------
    defaultMList = CreateMoleculeList(numHybs)
    mListFields = list(defaultMList.keys())

    # -------------------------------------------------------------------------
    # Transfer Molecule list fields
    # -------------------------------------------------------------------------
    for mL_i in mListFields:
        defaultWord[mL_i] = defaultMList[mL_i]

    # -------------------------------------------------------------------------
    # Create Array
    # -------------------------------------------------------------------------
    if numElements == 1:
        return defaultWord
    else:
        ## https://stackoverflow.com/questions/2785954/creating-a-list-in-python-with-multiple-copies-of-a-given-object-in-a-single-lin
        words = [copy.deepcopy(defaultWord) for _ in range(numElements)]
        return words



























