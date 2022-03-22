import os

import matplotlib
import numpy as np
import re
import glob
from os.path import basename,dirname
import struct
import uuid
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

from merfish.analysis.points import WarpPoints, tforminv_x
from merfish.analysis.image_utils import maketform, tforminv, MatchFeducials

from utils.funcs import tic, error, toc, knnsearch2d


def BuildImageDataStructures(folderPath, **kwargs): #
    # ------------------------------------------------------------------------
    # [imageData] = BuildImageDataStructures(folderPath, varargin)
    # This function creates imageData structures for all files in the
    # folderPath that satisfy the provided criteria.  It is a wrapper for the 
    # function BuildFileStructure.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # folderPath/ A path to the desired folder
    #--------------------------------------------------------------------------
    # Outputs
    # imageData/ A structure array containing a structure for each file found
    # in the folder.  See CreateImageDataStructure for field information.
    #--------------------------------------------------------------------------
    # Variable Inputs
    # See BuildFileStructure
    #--------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}

    parameters['fileExt'] = '*' # File extension to return
    parameters['delimiters'] = ['_'] # Delimiters to use to split
    parameters['fieldNames'] =[] # FieldNames
    parameters['fieldConv'] =  [str]  # Conversion functions
    parameters['appendExtraFields'] = False # Conversion functions
    parameters['requireFlag'] = '' # Conversion functions
    
    # -------------------------------------------------------------------------
    # Parse and normalize necessary input
    # -------------------------------------------------------------------------
    if not os.path.exists(folderPath):
        raise('[Error]:invalidArguments - A valid folder is required.')

    ## update the paramteres provided by users.
    for i in kwargs:
        parameters[i] = kwargs[i]

    # -------------------------------------------------------------------------
    # Find Files
    # -------------------------------------------------------------------------
    foundFiles = BuildFileStructure(folderPath, **parameters)[0]
    
    # -------------------------------------------------------------------------
    # Transfer undefined fields
    # -------------------------------------------------------------------------
    numElements = 1
    defaultImageDataStruct = CreateImageDataStructure(numElements)
    if numElements > 1:
        defaultImageDataStruct = defaultImageDataStruct[0]

    extraFields = [i for i in defaultImageDataStruct.keys() if i not in foundFiles[0].keys()]

    imageData = foundFiles
    for foundFile_i in imageData:
        for j in extraFields:
            foundFile_i[j] = defaultImageDataStruct[j]

    return [imageData, parameters]




def BuildFileStructure(folderPath, **kwargs):
    # ------------------------------------------------------------------------
    # [parsedFileStruct] = BuildFileStructure(folderPath, varargin)
    # This function parses the names of all files within a dataPath (and included
    # directories) to produce a structure array of properties specified by
    # those names.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # folderPath/ A path to the desired folder
    #--------------------------------------------------------------------------
    # Outputs
    # parsedFileStruct/ A structure array containing a variety of default
    #   and user specified fields
    #   -- name: The local name of the file
    #   -- filePath: The full path to the file
    #   -- Additional fields can be specified
    #--------------------------------------------------------------------------
    # Variable Inputs
    # 'fileExt'/string ('*'): The extension of files to be returned. By default all
    #   files are returned.
    # 'delimiters'/cell of strings ({'_'}): The characters used to split a
    #   complex file name into parts. The default is '_'.
    # 'fieldNames'/cell of strings ({'field1'}): The names assigned to portion of a
    #   split string. Any entry without a field name will not be assigned.
    # 'fieldConv'/cell of conversion functions ({'char'}): The function used to
    #   convert the parsed entry to a data type. If not specified for a field,
    #   it will remain a string.
    # 'appendExtraFields'/boolean ('False'): If there are extra fields beyond
    #   the specified fields, they will be combined to generate the final field.
    #--------------------------------------------------------------------------
    # Example:
    # Consider a name STORM_01_03.dax, where the delimiter is '_' the first
    # entry represents the name

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['fileExt']= '*' # File extension to return
    parameters['delimiters']= ['_'] # Delimiters to use to split
    parameters['fieldNames']=[] # FieldNames
    parameters['fieldConv']=[str] # Conversion functions
    parameters['appendExtraFields']= False
    parameters['excludeFlags']=[] # A cell array of strings that are excluded
    parameters['requireFlag']= '' # A string that is required
    parameters['requireExactMatch'] = False
    parameters['containsDelimiters'] = 0 # An integer specifying a field that might have internal delimiters
    parameters['regExp'] ="" # A regular expression with tokens that match fieldNames

    # -------------------------------------------------------------------------
    # Parse and normalize necessary input
    # -------------------------------------------------------------------------
    if not os.path.exists(folderPath):
        raise('[Error]: invalidArguments - A valid folder is required.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    ## update the paramteres provided by users.
    for i in kwargs:
        parameters[i] = kwargs[i]
# -------------------------------------------------------------------------
    # Check for field names that will overwrite hard coded field names
    # -------------------------------------------------------------------------
    if ('name' in parameters["fieldNames"]) or ('filePath' in parameters["fieldNames"]):
        raise('[Error]: invalidArguments - "name" and "filePath" are protected argements.')

    # -------------------------------------------------------------------------
    # Define old functionality of internal delimiters
    # -------------------------------------------------------------------------
    if parameters["containsDelimiters"] == 0:
        parameters["containsDelimiters"] =len(parameters["fieldConv"])

    # -------------------------------------------------------------------------
    # Additional parsing of parameters
    # -------------------------------------------------------------------------
    # Add extension delimiter
    parameters["delimiters"].append('.')
    # Coerce unfilled conversion functions
    for i in range(len(parameters["fieldConv"]),len(parameters["fieldNames"])):
        parameters["fieldConv"].append(str)

    # -------------------------------------------------------------------------
    # Find files
    # -------------------------------------------------------------------------
    if len(parameters["requireFlag"]) > 0:
        fileData = glob.glob(os.path.join(folderPath,'*'+parameters["requireFlag"]+'*.'+parameters["fileExt"]))
    else:
        fileData = glob.glob(os.path.join(folderPath,'*.'+parameters["fileExt"]))
    fileData_baseName = [basename(file_i) for file_i in fileData]

    # -------------------------------------------------------------------------
    # Exclude files with any of the excluded strings
    # -------------------------------------------------------------------------
    if len(parameters["excludeFlags"]) > 0:
        for e in parameters["excludeFlags"]:
            for file_i in fileData:
                if e in basename(file_i):
                    fileData.remove(file_i)

    # -------------------------------------------------------------------------
    # Parse Names
    # -------------------------------------------------------------------------
    parsedFileStruct = []
    count = 0
    for i in fileData:
        # Switch on whether or not a regular expression was provided
        if len(parameters["regExp"]) > 0:
            # Apply regular expression and capture token values
            parameters['regExp'] = parameters['regExp'].replace("?<","?P<")
            re_g = re.search(parameters['regExp'],basename(i))
            if not re_g:
                continue
            match_g = [i for i in re_g.re.groupindex]
            localStruct = {}
            for m_i in match_g:
                localStruct[m_i] = re_g.group(m_i)

            #Apply field conversions
            for f in parameters["fieldNames"]:
                if f not in localStruct: # Add empty field if not found
                    localStruct[f] = ""
                else:
                    idx = parameters["fieldNames"].index(f)
                    localStruct[f] = parameters["fieldConv"][idx](localStruct[f])

            localStruct["name"] = basename(i)
            localStruct["filePath"] = i
            localStruct["regExp"] = parameters["regExp"]

            parsedFileStruct.append(localStruct)

        else:
            # Split text
            sep_ls = []
            for d_i in parameters["delimiters"]:
                if d_i in [".","\\","?","*"]:
                    sep_ls.append("\\"+d_i)
                else:
                    sep_ls.append(d_i)
            splitText = re.split("|".join(sep_ls),basename(i))

            # Remove extension
            splitText = splitText[:-1]
            # Combine final field if appropriate
            if len(splitText) > len(parameters["fieldNames"]) and parameters["appendExtraFields"]:
                # Identify field with internal delimiters and recombine split string
                combinedString = []
                lenDiff = len(splitText) - len(parameters["fieldNames"])
                startInd = parameters["containsDelimiters"]
                finishInd = lenDiff + startInd
                for sp_i in splitText[startInd:finishInd]:
                    combinedString += [sp_i, "_"]
                combinedString=combinedString[:-1]

                    # Replace split text with the revised values
                oldSplitText = splitText
                splitText = []
                splitText = oldSplitText[:parameters["containsDelimiters"]]
                splitText += combinedString
                splitText += oldSplitText[parameters["containsDelimiters"]+lenDiff:]

            if parameters["requireExactMatch"]:
                splitCondition = len(splitText) == len(parameters["fieldNames"])
            else:
                splitCondition = len(splitText) <= len(parameters["fieldNames"])

            # Parse split text
            if splitCondition:
                parsedFileStruct_count = {}
                parsedFileStruct_count["name"] = basename(i)
                parsedFileStruct_count["filePath"] = i
                for j in range(len(splitText)):
                    parsedFileStruct_count[parameters["fieldNames"][j]] =  parameters["fieldConv"][j](splitText[j])
                parsedFileStruct_count["delimiters"] = parameters["delimiters"]
                count = count + 1
                parsedFileStruct.append(parsedFileStruct_count)

    # -------------------------------------------------------------------------
    # Issue warning if some files did not fit the pattern
    # -------------------------------------------------------------------------
    if len(parsedFileStruct) < len(fileData):
        print('[Warning]:unparsedFiles- Some files did not fit the specified pattern.')

    return [parsedFileStruct, parameters]



def CreateImageDataStructure(numElements):
    # ------------------------------------------------------------------------
    # imageData = CreateImageDataStructure(numElements)
    # This function creates an array of empty imageData structures.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    #   numElements/int. The number of elements to create.
    #--------------------------------------------------------------------------
    # Outputs
    # imageData/A structure array of length specified by numElements. Each
    #   element contains the following fields
    #
    #       ## File information
    #       --name: The name of the file
    #       --filePath: The path to the file
    #       --infFilePath: The path to the corresponding .inf file
    #       --uID: A string unique to this instance of this element
    #
    #       ## Movie/Image information
    #       --movieType: A string specifying the type of movie, e.g. STORM,
    #       bleach
    #       --hybNum: The number of the hyb
    #       --cellNum: The number of the FOV or cell
    #       --isFiducial: A boolean specifying whether the image is of fiducial
    #       markers
    #       --binType: A string defining the type of bin file to use for
    #       analysis, e.g. alist or med300_alist
    #       --delimiters: A cell array of the delimiters used to parse the
    #       filename
    #
    #       ## Movie/Image information from .inf file
    #       --imageH: The height of the image in pixels
    #       --imageW: The width of the image in pixels
    #       --Stage_X: The x position of the stage in um
    #       --Stage_Y: The y position of the stage in um
    #
    #       ## Focus lock
    #       --focusLockQuality: A scaler specifying the quality of the focus
    #       lock for this image
    #
    #       ## Molecules
    #       --mList: A structure containing information on all of the molecules
    #       identified in this image.  See CreateMoleculeList for field
    #       information.
    #
    #       ## Warp/Alignment
    #       --tform: An affine transformation structure used to align this
    #       image to a common coordinate system
    #       --warpErrors: A 1x5 vector containing errors associated with
    #       aligning fiducial markers
    #       --hasFiducialError: A boolean specifying whether there was an error
    #       in warping this image
    #       --fiducialErrorMessage: The provided error message
    #       --fiducialUID: The unique string (ID) of the imageData structure
    #       corresponding to the fiducial image used to warp this image.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # -- None.

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if numElements < 0:
        raise ('Functions:invalidArguments - Invalid values for numElements and numHybs')

    # -------------------------------------------------------------------------
    # Define fields and defaultProperties
    # -------------------------------------------------------------------------
    fieldsAndValues = {}

    # File name and unique ID
    fieldsAndValues['name']=''
    fieldsAndValues['filePath']=''
    fieldsAndValues['infFilePath']=''
    fieldsAndValues['uID']=''

    # Movie type and details
    fieldsAndValues['movieType']=''
    fieldsAndValues['hybNum']=-1
    fieldsAndValues['cellNum']=-1
    fieldsAndValues['isFiducial']=False
    fieldsAndValues['binType']=''
    fieldsAndValues['delimiters']=[]

    # Movie properties from inf file
    fieldsAndValues['imageH']=0
    fieldsAndValues['imageW']=0
    fieldsAndValues['Stage_X']=0
    fieldsAndValues['Stage_Y']=0

    # Movie focus quality
    fieldsAndValues['focusLockQuality']=0

    # Molecule lists
    fieldsAndValues['mList']=CreateMoleculeList(0)

    # Warp properties
    fieldsAndValues['tform']=maketform('affine',np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    fieldsAndValues['warpErrors']=np.zeros((5,))
    fieldsAndValues['hasFiducialError']=False
    fieldsAndValues['fiducialErrorMessage']=[]
    fieldsAndValues['fidUID']=''

    # -------------------------------------------------------------------------
    # Create imageData Structure
    # -------------------------------------------------------------------------
    defaultImageData = {}
    for i in fieldsAndValues:
        defaultImageData[i] = fieldsAndValues[i]

    if numElements > 1:
        ## !!!ATTENTION!!!
        # https://stackoverflow.com/questions/2785954/creating-a-list-in-python-with-multiple-copies-of-a-given-object-in-a-single-lin
        # imageData = np.tile(defaultImageData, numElements)
        imageData = [defaultImageData.copy() for _ in range(numElements)]
    else:
        imageData = defaultImageData

    return imageData


def CreateMoleculeList(numElements, **kwargs):
    #--------------------------------------------------------------------------
    # MStruct = CreateMoleculeList(numElements)
    # This function creates an array of empty molecule structures. It is used
    # to allocate memory for molecule lists.
    #--------------------------------------------------------------------------
    # Outputs:
    # MList/array of molecule structure: An empty array of molecule structures
    #--------------------------------------------------------------------------
    # Inputs:
    # numElements/integer(1): The number of elements to include in the molecule
    #   list
    #--------------------------------------------------------------------------
    # Variable Inputs:
    # 'fieldsToLoad'/cell (all fields): Create only the subset of fields provided
    #   in this option.

    #--------------------------------------------------------------------------
    # Hardcoded Variables
    #--------------------------------------------------------------------------
    quiet = 0

    fieldNames = ['x','y','xc','yc','h','a','w','phi','ax','bg','i','c',
                  'density','frame','length','link','z','zc']

    #--------------------------------------------------------------------------
    # Default Parameters
    #--------------------------------------------------------------------------
    fieldsToLoad = fieldNames

    #--------------------------------------------------------------------------
    # Parse Variable Input
    #--------------------------------------------------------------------------
    if len(kwargs) > 1:
        for param_i in kwargs:
            parameterName = param_i
            parameterValue = kwargs[param_i]
            if parameterName == 'compact' and isinstance(parameterValue,bool):
                    compact = parameterValue
            elif parameterName == 'fieldsToLoad' and isinstance(parameterValue,list):
                    fieldsToLoad = parameterValue
            else:
                raise ('[Error]: The parameter <',parameterName,'> is not recognized.')

    
    #--------------------------------------------------------------------------
    # Parse Variable Input
    #--------------------------------------------------------------------------
    fieldIndsToLoad = np.nonzero(np.isin(fieldNames, fieldsToLoad))[0]

    #--------------------------------------------------------------------------
    # Create Molecule Structure
    #--------------------------------------------------------------------------
    MList = {}
    for i in fieldIndsToLoad:
        MList[fieldNames[i]]= np.array([np.nan] * numElements)

    return MList

def bytes_to_int(bytes):
    result = 0
    for b in bytes[::-1]:
        result = result * 256 + int(b)
    return result

def bytes_to_float(bytes):
    result = struct.unpack('f', bytes)
    return result[0]


def bytes_to_str(bytes):
    result = ""
    for b in bytes:
        result += chr(int(b))
    return result

def parseBinData(binData):
    '''
    #--------------------------------------------------------------------------
    # Organization of the _list.bin files
    #--------------------------------------------------------------------------
    #{
    4 byte "M425" string tag
    4 byte integer number of frames N
    4 byte integer status
    Frame 0
     |  |__4 byte integer number of molecules M
     |  |__variable size M molecule structures
     |      |__72 bytes structure Molecule_1
     |      |   |__4 byte float X in pixels from the middle of top left pixel
     |      |   |__4 byte float Y in pixels from the middle of top left pixel
     |      |   |__4 byte float Xc same as X but corrected for drift
     |      |   |__4 byte float Yc same as Y but corrected for drift
     |      |   |__4 byte float h peak height in first frame
     |      |   |__4 byte float a integrated area
     |      |   |__4 byte float w width
     |      |   |__4 byte float phi (for 3D data distance from calibration curve in WxWy space)
     |      |   |__4 byte float Ax axial ratio Wx/Wy
     |      |   |__4 byte float b local background
     |      |   |__4 byte float i direct intensity
     |      |   |__4 byte integer channel number
     |      |   |    (0: non-specific, 1-3: specific, 4-8: crosstalk, 9: Z rejected)
     |      |   |__4 byte integer valid (not used) (overwritten with density)
     |      |   |__4 byte integer frame where the molecule first appeared
     |      |   |__4 byte integer length of the molecule trace in frames
     |      |   |__4 byte integer link index of the molecule in the next frame list
     |      |   |    (or -1 for link end)
     |      |   |__4 byte float Z in nanometers from cover glass
     |      |   |__4 byte float Zc same as Z but corrected for drift
     |      |__72 bytes structure Molecule_2
     |      |__72 bytes structure Molecule_i
     |      |__72 bytes structure Molecule_M
     |__
    #}
    '''

    format = [['x', 'y', 'xc', 'yc', 'h', 'a', 'w', 'phi', 'ax', 'bg', 'i', 'c',
               'density', 'frame', 'length', 'link', 'z', 'zc'],
              ['float', 'float', 'float', 'float', 'float', 'float', 'float',
               'float', 'float', 'float', 'float', 'int', 'int', 'int', 'int',
               'int', 'float', 'float', 'float']]
    data_dict = {}
    pos_n = 0

    data_dict["tag"] = bytes_to_str(binData[pos_n:pos_n+4])
    pos_n += 4

    data_dict["frame_num"] = bytes_to_int(binData[pos_n:pos_n+4])
    pos_n += 4

    data_dict["status"] = bytes_to_int(binData[pos_n:pos_n + 4])
    pos_n += 4

    ##Frames
    data_dict["frame_num"] = 1 if data_dict["frame_num"] == 0 else data_dict["frame_num"]
    data_dict["frames"] = []
    for f_i in range(data_dict["frame_num"]):
        frame_i  = {}
        frame_i["mol_num"] = bytes_to_int(binData[pos_n:pos_n + 4])
        pos_n += 4

        frame_i["molecules"] = []
        for m_i in range(frame_i["mol_num"]):
            mol_i = {}
            for attr_i in range(len(format[0])):
                if format[1][attr_i] == "float":
                    mol_i[format[0][attr_i]] = bytes_to_float(binData[pos_n:pos_n + 4])
                elif format[1][attr_i] == "int":
                    mol_i[format[0][attr_i]] = bytes_to_int(binData[pos_n:pos_n + 4])
                elif format[1][attr_i] == "str":
                    mol_i[format[0][attr_i]] = bytes_to_int(binData[pos_n:pos_n + 4])
                else:
                    error("[Error]: parseBin failed.")
                pos_n += 4
            frame_i["molecules"].append(mol_i)
        data_dict["frames"].append(frame_i)
    return data_dict


def ReadMasterMoleculeList(fileName, **kwargs):
    #--------------------------------------------------------------------------
    # MList = ReadMasterMoleculeList(fileInfo, varargin)
    # This function loads a .bin file containing a molecule list and converts
    # it into a matlab structure. This function only loads a master list, i.e.
    # the list corresponding to frame 0.
    #
    #--------------------------------------------------------------------------
    # Outputs:
    #
    # MList/structure array: This array contains a structure element for each
    # molecule.
    #
    # memoryMap/memory map structure: This structure contains information on
    # the dynamic link between matlab and the memory containing the given file
    #
    #--------------------------------------------------------------------------
    # Inputs:
    #
    # fileName/string or structure: fileName can be a string containing the
    # name of the file with its path or it can be a infoFile structure
    #
    #--------------------------------------------------------------------------
    # Variable Inputs:
    #
    # 'verbose'/boolean (true): print or hide function progress
    #
    # 'compact'/boolean (false): Toggles between a array of structures or a
    #   structure of arrays.  The later is much more memory efficient.
    #
    # 'ZScale'/positive (1): The value by which the Z data will be rescaled.
    #    Useful for converting nm to pixels.
    #
    # 'transpose/boolean (false): Change the entries from Nx1 to 1xN.
    #
    # 'fieldsToLoad'/cell (all fields): Load only the subset of fields provided
    #   in this option.
    #
    # 'loadAsStructArray'/boolean (false): Load the mList as a structure array
    #   object, which provides the convenience of the non-compact format
    #   without the memory overhead. See StructureArray.m for details.

    #--------------------------------------------------------------------------
    # Hardcoded variables
    #--------------------------------------------------------------------------
    format = [['x','y','xc','yc','h','a','w','phi','ax','bg','i','c',
               'density','frame','length','link','z','zc'],
              ['float', 'float', 'float', 'float', 'float', 'float', 'float',
               'float', 'float', 'float', 'float', 'int', 'int', 'int', 'int',
               'int', 'float', 'float', 'float']]

    headerSize = 16
    numEntries = 18
    entrySize = 4
    #--------------------------------------------------------------------------
    # Define default parameters
    #--------------------------------------------------------------------------
    transpose = False
    verbose = True
    ZScale = 1
    fieldsToLoad = format[0]
    loadAsStructArray = False
    if fileName:
        pass
    else:
        fileName = []
        error('STORM:invalidArguments - A valid path must be provided')

    #--------------------------------------------------------------------------
    # Parse Variable Input Arguments
    #--------------------------------------------------------------------------
    if len(kwargs)>1:
        for parameterName in kwargs:
            parameterValue = kwargs[parameterName]
            if parameterName == 'verbose':
                verbose = parameterValue
            elif parameterName == 'compact':
                compact = parameterValue
            elif parameterName == 'ZScale':
                ZScale = parameterValue
            elif parameterName == 'transpose':
                transpose = parameterValue
            elif parameterName == 'fieldsToLoad':
                fieldsToLoad = parameterValue
            elif parameterName == 'loadAsStructArray':
                loadAsStructArray = parameterValue
            else:
                error('[Error]: The parameter <',parameterName, '> is not recognized.')

    #--------------------------------------------------------------------------
    # Open File, read header, and determine file properties
    #--------------------------------------------------------------------------
    fid = open(fileName,'rb')
    if not fid:
        error('Problem opening file:', fileName)

    file_data = fid.read()
    data_dict = parseBinData(file_data)

    tag = data_dict["tag"]
    numFrames = data_dict["frame_num"]
    numMoleculesFrame0 = data_dict["frames"][0]["mol_num"]
    
    if verbose:
        print('-------------------------------------------------------------')
        print('Opening file:', fileName)
        print('Version:',tag)
        print('Contains:',numFrames, 'field')#
        print('Status:',file_data[4])
        print('Number of molecules in Frame 0:', numMoleculesFrame0)
        print('-------------------------------------------------------------')
    
    fid.close()

    MList = data_dict["frames"][0]["molecules"]

    DoThis = True
    if numMoleculesFrame0 >  20E6:
        print('[Warning]:File contains more than 20 million molecules !!!')

    #--------------------------------------------------------------------------
    # Confirm provided fields are valid
    #--------------------------------------------------------------------------
    fieldsToLoad = [f_i for f_i in fieldsToLoad if f_i in format[0]]
    
    #--------------------------------------------------------------------------
    # Handle empty MList case
    #--------------------------------------------------------------------------
    if numMoleculesFrame0 == 0:
        if transpose:
            emptyArray = np.zeros((0,1))
        else:
            emptyArray = np.zeros((1,0))

        MList = {}
        for f in fieldsToLoad:
            MList[f] = np.array(emptyArray, dtype=format[1][format.index(f)])

        return [MList] # Exit function
    

    #--------------------------------------------------------------------------
    # Rescale Z if necessary
    #--------------------------------------------------------------------------
    if ZScale != 1:
        for m_i in MList:
            if 'z' in  m_i:
                m_i['z'] = m_i['z']/ZScale
            if 'zc' in  m_i:
                m_i['zc'] = m_i['zc']/ZScale


    MList = [{f_i:ML_i[f_i] for f_i in fieldsToLoad} for ML_i in MList]

    return MList


def TransferInfoFileFields(fileStructs, **kwargs):
    # ------------------------------------------------------------------------
    # fileStruct = TransferInfoFileFields(fileStruct, **kwargs)
    # This function finds the .inf files associated with each file in
    #   fileStructs, loads it, and transfers specified fields to each entry in
    #   fileStructs. 
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # fileStructs/A array of structures with the following fields:
    #   --filePath: The path to the specific file.
    #--------------------------------------------------------------------------
    # Outputs
    # fileStructs/The same array of structures with additional fields from the
    #   corresponding info file structures added. 
    # binType/A string that specifies the bin type.  Only required if the
    #  default generateDaxName function is used.  
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # jeffmoffitt@gmail.com
    # September 10, 2014
    #--------------------------------------------------------------------------
    # Creative Commons License CC BY NC SA
    #--------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    
    # Parameters for parsing file names
    parameters = {}
    parameters['verbose'] = False
    parameters['infFieldsToAdd']=['Stage_X', 'Stage_Y']
    parameters['generateDaxName'] = lambda x: x["filePath"][:x["filePath"].find(x["binType"])-1]+'.dax'
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]
    
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if (len(kwargs) < 1) or ('filePath' not in fileStructs[0]):
        error('[Error]::invalidArguments - Improper structure provided.')

    # -------------------------------------------------------------------------
    # Printed updates
    # -------------------------------------------------------------------------
    if parameters["printedUpdates"]:
        print('--------------------------------------------------------------')
        timer = tic(99)
        print('Loading and transfering info files')
    
    # -------------------------------------------------------------------------
    # Loop over fileStructs
    # -------------------------------------------------------------------------
    for i in fileStructs:
        infFileName = parameters["generateDaxName"](i)
        infStruct = ReadInfoFile(infFileName, verbose=parameters["verbose"])
        
        # Add hardcoded fields
        i["infFilePath"] = infStruct["localPath"] +os.path.sep+ infStruct["localName"]
        i["imageH"] = infStruct["frame_dimensions"][0]
        i["imageW"] = infStruct["frame_dimensions"][1]
        
        # Add generic fields
        for j in parameters["infFieldsToAdd"]:
            i[j] = infStruct[j]
    
    # -------------------------------------------------------------------------
    # Printed updates
    # -------------------------------------------------------------------------
    if parameters["printedUpdates"]:
        print('...finished in ',toc(timer), 's')

    return fileStructs

def ReadInfoFile(fileName,**kwargs):
    #--------------------------------------------------------------------------
    # infoFile = ReadInfoFile(fileName, varargin)
    # This function returns a structure, info, containing the elements of an
    # .inf file.
    #--------------------------------------------------------------------------
    # Outputs:
    # info/struct: A structure array containing the elements of the info file
    #
    #--------------------------------------------------------------------------
    # Inputs:
    # fileName/string or cell array of strings ([]): A path to a .dax or .ini
    #   file
    # Field: explanation
    #                  localName: inf filename matlab found / should save as
    #                  localPath: where matlab found / should save this file
    #                   uniqueID: ?
    #                       file: full path to daxfile.
    #               machine_name: e.g. 'storm2'
    #            parameters_file: Full pathname of pars file used in Hal
    #              shutters_file: e.g. 'shutters_default.xml'
    #                   CCD_mode: e.g. 'frame-transfer'
    #                  data_type: '16 bit integers (binary, big endian)'
    #           frame_dimensions: [256 256]
    #                    binning: [1 1]
    #                 frame_size: 262144
    #     horizontal_shift_speed: 10
    #       vertical_shift_speed: 3.3000
    #                 EMCCD_Gain: 20
    #                Preamp_Gain: 5
    #              Exposure_Time: 0.1000
    #          Frames_Per_Second: 9.8280
    #         camera_temperature: -70
    #           number_of_frames: 10
    #                camera_head: 'DU897_BV'
    #                     hstart: 1
    #                       hend: 256
    #                     vstart: 1
    #                       vend: 256
    #                  ADChannel: 0
    #                    Stage_X: 0
    #                    Stage_Y: 5
    #                    Stage_Z: 0
    #                Lock_Target: 0
    #                   scalemax: 4038
    #                   scalemin: 0
    #                      notes: ''
    #--------------------------------------------------------------------------
    # Variable Inputs:
    # 'file'/string or cell array: The file name(s) for the .ini file(s) to load
    #   Path must be included.
    #
    # 'verbose'/boolean(true): Determines if the function hides progress
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Define default parameters
    #--------------------------------------------------------------------------
    infFileName = []
    verbose = False

    #--------------------------------------------------------------------------
    # Parse Required Input
    #--------------------------------------------------------------------------
    infFileName = fileName
    for parameterName in kwargs:
        parameterValue = kwargs[parameterName]
        if parameterName=='file':
            infFileName = parameterValue
        elif parameterName == 'verbose':
            verbose = parameterValue
        else:
            error('[Error]: The parameter <', parameterName, '> is not recognized.')
    
    #--------------------------------------------------------------------------
    # Get file if needed
    #--------------------------------------------------------------------------
    if len(infFileName) == 0:
        error("[Error]: Invalid file path - ReadInfoFile()")

    
    #--------------------------------------------------------------------------
    # Open Inf File
    #--------------------------------------------------------------------------
    if infFileName.endswith('.dax'):
        infFileName = infFileName[:-4] + '.inf'

    if not os.path.exists(infFileName):
        error("File does not exist:"+infFileName)
    # --------------------------------------------------------------------------
    # Read Inf File
    # --------------------------------------------------------------------------
    count = 0
    text = {}
    with open(infFileName) as fid:
        for line in fid:
            text[count] = line.strip()
            count = count + 1
    
    #--------------------------------------------------------------------------
    # Create Info File
    #--------------------------------------------------------------------------
    infoFile = {
    "localName":'',
    "localPath":'',
    "uniqueID":str(uuid.uuid4())[:8],
    "file":'',
    "machine_name":'',
    "parameters_file":'',
    "shutters_file":'',
    "CCD_mode":'',
    "data_type":'16 bit integers (binary, big endian)',
    "frame_dimensions":[0,0],
    "binning":[1,1],
    "frame_size":0,
    "horizontal_shift_speed":0,
    "vertical_shift_speed":0,
    "EMCCD_Gain":1,
    "Preamp_Gain":1,
    "Exposure_Time":1,
    "Frames_Per_Second":1,
    "camera_temperature":1,
    "number_of_frames":1,
    "camera_head":'',
    "hstart":1,
    "hend":256,
    "vstart":1,
    "vend":256,
    "ADChannel":0,
    "Stage_X":0,
    "Stage_Y":0,
    "Stage_Z":0,
    "Lock_Target":0,
    "scalemax":0,
    "scalemin":0,
    "notes":''
    }

    [infFilePath, name, extension] = [dirname(infFileName), basename(infFileName),os.path.splitext(infFileName)[1]]

    infoFile["localName"] = name
    infoFile["localPath"] = infFilePath
    infoFile["uniqueID"] = str(uuid.uuid4())[:8]
    #--------------------------------------------------------------------------
    # Parse each line and build ini structure
    #--------------------------------------------------------------------------
    for j in text:
        # Does the line contain a definition
        if "=" in text[j]:
            line_ls = text[j].split("=")

            #Prepare field name
            textName = line_ls[0].strip()
            textName = re.sub("[\{\(\[].*?[\)\]\}]", "", textName)
            textName = textName.strip()
            textName = re.sub(r'[^A-Za-z0-9 ]+', '', textName)
            fieldName = textName.replace(" ","_")

            #Prepare value
            value = line_ls[1].strip()  # Read value
            if ('x' in value) and (':' not in value):
                value1 = value.split("x")[0].strip()
                value2 = value.split("x")[1].strip()
                infoFile[fieldName] = [int(value1),int(value2)]
            else:
                fieldValue = value
                try:
                    infoFile[fieldName] = float(fieldValue)
                except:
                    infoFile[fieldName] = fieldValue
    
        elif 'information file for' in text[j]:  #If the line does not contain a definition, then the next line is the file name
            infoFile["file"] = text[j+1]

    if verbose:
        print('Loaded '+infFilePath+os.path.sep+name+'.inf')
    
    #--------------------------------------------------------------------------
    # Check frame dimensions
    #--------------------------------------------------------------------------
    if 0 in infoFile["frame_dimensions"]:
        print('STORM:corruptedInfoFile, Unexpected frame dimensions')
        infoFile["frame_dimensions"] = [infoFile["hend"] - infoFile["hstart"] + 1,
                                        infoFile["vend"] - infoFile["vstart"] + 1]
    
    return infoFile

        
def AlignFiducials(fiducialData,**kwargs):
    # ------------------------------------------------------------------------
    # [fiducialData] = AlignFiducials(fiducialData,varargin)
    # This function analyzes a series of raw conventional images in the 
    #   specified directory and creates a words structure which represents all
    #   of the identified words in the data.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # fiducialData/A structure array with elements equal to the number of
    #   images to align. This structure must contain an mList field. 
    #--------------------------------------------------------------------------
    # Outputs
    # fiducialData/The same structure array provided but with two new fields
    #   added.
    #   --transform. The transform that will bring each image into the
    #     reference frame of the first image. 
    #   --warpUncertainty. A matrix that describes the quality of the generated
    #     transform. 
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Alistair Boettiger
    # boettiger@fas.harvard.edu
    # Jeffrey Moffitt 
    # lmoffitt@mcb.harvard.edu
    # September 6, 2014
    #--------------------------------------------------------------------------
    # Based on previous functions by Alistair Boettiger
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.
    
    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['maxD'] = 8 # Maximum distance for fiducial tracking
    parameters['fiducialFrame'] = 1 # Frame to use for tracking
    parameters['fiducialWarp2Hyb1']= False
    parameters['verbose'] = True
    parameters['printedUpdates']= True
    
    parameters['reportsToGenerate']= []
    parameters['useSubFolderForCellReport']= True
    parameters['overwrite']= True
    parameters['figFormats']='png'
    parameters['saveAndClose']= False
    
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if len(kwargs) < 1 or len(fiducialData) < 2:
        error('[Error]:AlignFiducials() - Invalid fiducial data.')

    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]
    
    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    if parameters["printedUpdates"] and parameters["verbose"]:
        print('--------------------------------------------------------------')
        print('Analyzing fiducials...')
    
    # -------------------------------------------------------------------------
    # Handling optional variables
    # -------------------------------------------------------------------------
    if 'numHybs' not in parameters:
        parameters["numHybs"] = len(fiducialData)
    
    # -------------------------------------------------------------------------
    # Optional plotting
    # -------------------------------------------------------------------------
    jet = cm.get_cmap('jet')
    figHandle1 = ""
    figHandle2 = ""
    if 'fiducialReport1' in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]["fiducialReport1"]:
        figHandle1 = plt.figure(1,figsize=(6,6))
        ax1 = figHandle1.add_subplot()
        figHandle1_name = 'fiducialReport1_cell' + str(fiducialData[0]["cellNum"])
        clrmap1 = jet(np.linspace(0, 1, len(fiducialData)))
        r1 = 5 # radius for color wheels of aligned hybes

    if 'fiducialReport2' in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]["fiducialReport2"]:
        figHandle2 = plt.figure(2, figsize=(6, 6))
        ax2 = figHandle2.add_subplot()
        figHandle2_name = 'fiducialReport2_cell' + str(fiducialData[0]["cellNum"])
        clrmap2 = jet(np.linspace(0, 1, len(fiducialData)))
    
    # -------------------------------------------------------------------------
    # Build cell array of bead x,y positions
    # -------------------------------------------------------------------------
    fedPos = []
    w = []
    for fD_i in fiducialData:
        try:
            frame = [1,2]
            mList = [i for i in fD_i['mList'] if i['frame'] >= frame[0] and i['frame'] < frame[1]]
            tempPos = [[],[]]
            tempPos[0] = [mList_i["xc"] for mList_i in mList] #Kludge to force orientation of array
            tempPos[1] = [mList_i["yc"] for mList_i in mList]
            w.append([mList_i["w"] for mList_i in mList])
            fedPos.append(tempPos)
        except Exception as e:
            print(e)

    # -------------------------------------------------------------------------
    # Build transforms
    # -------------------------------------------------------------------------
    tformNN = [[]] * parameters["numHybs"]
    i = 0
    for fD_i in fiducialData:
        print("-> Processing...", i + 1, "/", len(fiducialData))
        parameters["fiducialNum"] = i
        try:
            fD_i["warpErrors"] = np.empty((5,))
            fD_i["warpErrors"][:] = np.nan
            if parameters["fiducialWarp2Hyb1"]: # warp all beads to first hybe
                [fD_i["tform"], tempErrors] = Warp2BestPair(fedPos[0],fedPos[i], **parameters)
            else:  # warp to previous hybe
                [tformNN[i], tempErrors] = Warp2BestPair(fedPos[max(i-1, 0)], fedPos[i], **parameters)
                fD_i["tform"] = maketform('composite',tformNN[0:i+1])

            fD_i["hasFiducialError"] = False
            fD_i["fiducialErrorMessage"] = []
            inds = list(range(0,(min(len(tempErrors), 5))))
            fD_i["warpErrors"][inds] = tempErrors[inds]
        except Exception as e:
            print(e)
            print('failed to warp data for hybe',i,'. Using previous positions')
            if i==0:
                fD_i["tform"] = maketform('affine',[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # don't warp if you can't warp
                tformNN[i]= maketform('affine',[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # don't warp if you can't warp
                fD_i["warpErrors"] = np.empty((5,))
                fD_i["warpErrors"][:] = np.nan
            else:
                fD_i["tform"] = fiducialData[i-1]["tform"] # don't warp if you can't warp
                tformNN[i]= maketform('affine',[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # don't warp if you can't warp
                fD_i["warpErrors"] = np.empty((5,))
                fD_i["warpErrors"][:] = np.nan
            fD_i["hasFiducialError"] = True
            fD_i["fiducialErrorMessage"] = e

        if parameters["verbose"]:
            print('  Alignment error =',fD_i["warpErrors"])

        # -------------------------------------------------------------------------
        # Plot Fiducial Report 1 Data: Bead Positions for Each Hyb
        # -------------------------------------------------------------------------
        if figHandle1 !="":
            tw = tforminv(fD_i["tform"],fedPos[i])
            xw, yw = tw[:, 0], tw[:, 1]
            ax1.plot(xw,yw,'.',color=clrmap1[i],markersize=1)
            theta = np.pi/len(fiducialData)*float(i+1)
            xl = np.asarray([xw-r1*np.cos(theta),xw+r1*np.cos(theta),[np.nan]*len(xw)]).T.reshape(-1,1)
            yl = np.asarray([yw-r1*np.sin(theta),yw+r1*np.sin(theta),[np.nan]*len(yw)]).T.reshape(-1,1)
            ax1.plot(xl,yl,color=clrmap1[i],lw=0.5)

        # -------------------------------------------------------------------------
        # Plot Fiducial Report 2 Data: Bead Widths for Each Hyb
        # -------------------------------------------------------------------------
        if figHandle2 !="":
            tw = tforminv(fD_i["tform"], fedPos[i])
            xw, yw = tw[:, 0], tw[:, 1]
            theta = np.pi/len(fiducialData)*(i+1.0)
            r2 = np.asarray(w[i])/80
            xl = np.asarray([xw - r2 * np.cos(theta), xw + r2 * np.cos(theta), [np.nan] * len(xw)]).T.reshape(-1, 1)
            yl = np.asarray([yw - r2 * np.sin(theta), yw + r2 * np.sin(theta), [np.nan] * len(yw)]).T.reshape(-1, 1)
            ax2.plot(xl,yl,color=clrmap2[i],lw=1)

        i += 1

    # -------------------------------------------------------------------------
    # Finalize report figures and save
    # -------------------------------------------------------------------------
    if 'figHandle1'!="":
        ax1.set_xlim(0,256)
        ax1.set_ylim(0,256)
        ax1.set_xlabel('pixels')
        ax1.set_ylabel('pixels')

    if parameters["saveAndClose"]:
        if parameters["useSubFolderForCellReport"]:
            if not os.path.exists(os.path.join(parameters["savePath"],"Cell_" + str(parameters["cellIDs"]))):
                os.mkdir(os.path.join(parameters["savePath"],"Cell_" + str(parameters["cellIDs"])))
            saveFile = os.path.join(parameters["savePath"],
                                    "Cell_" + str(parameters["cellIDs"]),
                                    figHandle1_name + "." + parameters["figFormats"])
            figHandle1.savefig(saveFile)
        else:
            saveFile = os.path.join(parameters["savePath"],
                                    figHandle1_name + "." + parameters["figFormats"])
            figHandle1.savefig(saveFile)
        plt.close(figHandle1)

    if 'figHandle2' != "":
        ax2.set_xlim(0, 256)
        ax2.set_ylim(0, 256)
        ax2.set_xlabel('pixels')
        ax2.set_ylabel('pixels')

    if parameters["saveAndClose"]:
        if parameters["useSubFolderForCellReport"]:
            if not os.path.exists(os.path.join(parameters["savePath"], "Cell_" + str(parameters["cellIDs"]))):
                os.mkdir(os.path.join(parameters["savePath"], "Cell_" + str(parameters["cellIDs"])))
            saveFile = os.path.join(parameters["savePath"],
                                    "Cell_"+ str(parameters["cellIDs"]),
                                    figHandle2_name+"."+parameters["figFormats"])
            figHandle2.savefig(saveFile)
        else:
            saveFile = os.path.join(parameters["savePath"],
                                    figHandle2_name + "." + parameters["figFormats"])
            figHandle2.savefig(saveFile)
        plt.close(figHandle2)
                
    return fiducialData,parameters
            
     
def Warp2BestPair(hybe1,hybe2, **kwargs):
    # Compute translation/rotation warp that best aligns the points in hybe1
    # and hybe2 by maximizing the alignment of the two points that show the
    # most mutually consistent x,y translation.
    # Copyright Presidents and Fellows of Harvard College, 2016.


    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['maxD']= 2
    parameters['useCorrAlign']=True
    parameters['fighandle']= []
    parameters['imageSize']= [256, 256]
    parameters['showPlots']= True
    parameters['verbose']= True

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]


    hybe1 = np.array(hybe1)
    hybe2 = np.array(hybe2)

    #--------------------------------------------------------------------------
    ## Main Function
    #--------------------------------------------------------------------------
    [matched1, matched2,parameters] = MatchFeducials(hybe1,hybe2,**parameters)

    if len(matched1) < 2:
        error('[Error]: Found fewer than 2 feducials, cannot compute warp')
    else:
        # Warp that maximizes MSD for best pair of beads
        # Compare the shift vectors, identify the pair that have the most similar
        shifts = np.array([hybe1[0,matched1] - hybe2[0,matched2],
                           hybe1[1,matched1] - hybe2[1,matched2]])
        [idx,dist] = knnsearch2d(shifts.T,shifts.T,2)
        shiftIdx  = np.argmin(dist[1,:])
        bestpair = np.array([[matched1[shiftIdx],matched2[shiftIdx]],
                             [matched1[idx[1,shiftIdx]],matched2[idx[1,shiftIdx]]]])
        [tform2,_,_,_,_] = WarpPoints(hybe1[:,bestpair[:,0]].T,hybe2[:,bestpair[:,1]].T,'translation rotation')
        tw = tforminv_x(tform2["tdata"]["T"],hybe2.T)
        xw, yw = tw[:,0],tw[:,1]

        if parameters["showPlots"]:
            if isinstance(parameters["fighandle"], matplotlib.figure.Figure):
                fig_x = parameters["fighandle"]
            elif parameters["fighandle"] == "" or len(parameters["fighandle"]) == 0:
                fig_x= plt.figure(10,figsize=(6, 6))
                parameters["fighandle"] = fig_x
            else:
                error('[Error]: parameters["fighandle"] setting is wrong.'
                      'parameters["fighandle"] can only be an empty string(''), empty liat([]) or an instance of matplotlib.figure.Figure')
            ax = fig_x.add_subplot()
            ax.plot(xw,yw,'k.', markersize=1)
            ax.plot(hybe1[0,:],hybe1[1,:],'bo',fillstyle='none',markersize=4,markeredgewidth=0.5)
            ax.set_title('Aligned by the pair of beads that best match')

            if not os.path.exists(os.path.join(parameters["savePath"], "savedImages")):
                os.makedirs(os.path.join(parameters["savePath"], "savedImages"), exist_ok=True)
            fig_x_name = os.path.join(parameters["savePath"],
                                      "savedImages/Aligned_bestPairs_c"+str(parameters["cellIDs"])+
                                      "_f"+str(parameters["fiducialNum"])+"."+parameters["figFormats"])
            fig_x.savefig(fig_x_name)
            plt.close(fig_x)


        [_,dist_PairWarp] = knnsearch2d(hybe1.T,tw,1)
        tform2["tdata"]["Tinv"][0,0] = 1
        dist_PairWarp = dist_PairWarp[0]
        dist_PairWarp[dist_PairWarp > parameters["maxD"]] = np.nan
        warpErrors = np.sort(dist_PairWarp)

    return [tform2, warpErrors]


def TransformImageData(imageData,fiducialData,**kwargs):
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # imageData/A structure array with elements equal to the number of
    #   images to align. This structure must contain an mList field.
    # fiducialData/A structure array with elements equal to the number of
    #   images to align. This structure must contain an mList field.
    #--------------------------------------------------------------------------
    # Outputs
    # imageData/A structure array with elements equal to the number of
    #   images to align. This structure must contain an mList field.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['verbose']=True
    parameters['printedUpdates'] = True

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    if parameters["printedUpdates"] and parameters["verbose"]:
        print('--------------------------------------------------------------')
        print('Shifting images')
    fieldsToTransferFrom = ['tform', 'warpErrors', 'hasFiducialError', 'fiducialErrorMessage', 'uID']
    fieldsToTransferTo = ['tform', 'warpErrors', 'hasFiducialError', 'fiducialErrorMessage', 'fidUID']
    for j in range(len(imageData)):
        xy_c= tforminv(
            fiducialData[j]["tform"],
            [[iD_mL["x"] for iD_mL in imageData[j]["mList"]],
             [iD_mL["y"] for iD_mL in imageData[j]["mList"]]]
        )
        for iD_mL_i in range(len(imageData[j]["mList"])):
            imageData[j]["mList"][iD_mL_i]["xc"] = xy_c[iD_mL_i, 0]
            imageData[j]["mList"][iD_mL_i]["yc"] = xy_c[iD_mL_i, 1]

        for k in range(len(fieldsToTransferFrom)):
            imageData[j][fieldsToTransferTo[k]] = fiducialData[j][fieldsToTransferFrom[k]]
    return imageData, parameters


def ReadDax(**kwargs):
    # --------------------------------------------------------------------------
    # [movie, infoFiles] = ReadDax(fileName, varargin)
    # This function loads a STORM movies from the dax file associated with the
    # provided .inf file
    # --------------------------------------------------------------------------
    # Outputs:
    # movies/LxMxN array: A 3D array containing the specified movie
    # infoFile: infoFile structure for the specified daxfile
    # infoFileRoi: modified infoFile corresponding to the daxfile
    # --------------------------------------------------------------------------
    # Input:
    # fileName/string or structure: Either a path to the dax or inf file or an
    #   infoFile structure specifying the dax file to load
    #
    # --------------------------------------------------------------------------
    # Variable Inputs:
    #
    # 'file'/string ([]): A path to the associated .inf file
    #
    # 'path'/string ([]): Default path to look for .inf files
    #
    # 'startFrame'/double  (1): first of movie to load.
    #
    # 'endFrame'/double ([]): last frame of the movie to load.  If empty will
    # be max.
    #
    # 'subregion'/double (zeros(4,1)):  [xmin, xmax, ymin, ymax] of the region
    # of the dax file to load.  Pixels indexed from upper left, as in images.
    #
    # 'infoFile'/info file structure ([]): An info file for
    # the files to be loaded.
    #
    # 'imageDimensions'/3x1 integer array ([]): The size of the movies to be
    # loaded.
    #
    # 'verbose'/boolean (true): Display or hide function progress
    #
    # 'orientation'/string ('normal'): Control the relative rotation of the data
    #   structure
    # --------------------------------------------------------------------------
    # Jeffrey Moffitt
    # jeffmoffitt@gmail.com
    # September 7, 2012
    #
    # Version 1.1
    # -------------------Updates:
    # 01/19/13: ANB
    # modified to allow arbitrary start and end frame to be specified
    # by the user.  Removed 'image_dimension' flag (this was non-functional)
    # and removed allFrames (this has become redundant)
    # -----------------------
    # 2/14/13: JRM
    # Minor fix to dax data type
    # -----------------------
    # ~12/15/13: ANB
    # ReadDax now respects binning options in dax file
    # ReadDax also computes how much memory it will take to load the requested
    # file and throws a warning if this exceeds a certain max value. Default
    # max is 1 Gb.  Warning allows user to continue, reduce frames, or abort.
    # -----------------------
    # 12/22/13: ANB
    # Added 'subregion' feature.
    # -------------------
    # ~08/01/15: ANB
    # fixed bug: data-type was hard-coded, should use what the info file
    # specifies.
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Hardcoded Variables
    # --------------------------------------------------------------------------
    quiet = 0
    orientationValues = ['normal', 'nd2']
    flags = ['infoFile', 'startFrame', 'endFrame', 'verbose', 'orientation', 'dataPath', 'allFrames', 'subregion']

    # --------------------------------------------------------------------------
    # Default Variables
    # --------------------------------------------------------------------------
    parameters = {}
    parameters['dataPath'] = []
    parameters['allFrames'] = False
    parameters['startFrame'] = 0
    parameters['endFrame'] = -1
    parameters['infoFile'] = []
    parameters['subregion'] = []
    parameters['verbose'] = True
    parameters['orientation'] = 'normal'

    maxMemory = 2 * 1024 * 1024 * 1024  # 2 Gb

    # --------------------------------------------------------------------------
    # Parse Input
    # --------------------------------------------------------------------------
    for k_i in kwargs:
        if k_i in flags:
            parameters[k_i] = kwargs[k_i]
        else:
            print("[Warning]: Parameter " + k_i + " is not reconized. Skipped.")

    dataPath = parameters['dataPath']
    allFrames = parameters['allFrames']
    startFrame = parameters['startFrame']
    endFrame = parameters['endFrame']
    infoFile = parameters['infoFile']
    subregion = parameters['subregion']
    verbose = parameters['verbose']
    orientation = parameters['orientation']

    infoFileRoi = []

    # --------------------------------------------------------------------------
    # Check parameter consistency
    # --------------------------------------------------------------------------
    if infoFile == []:
        error('[Error]: You did not specify info files.')

    fileName = infoFile
    # --------------------------------------------------------------------------
    # Load info files if needed
    # --------------------------------------------------------------------------
    infoFile = ReadInfoFile(infoFile, verbose=parameters["verbose"])

    if len(infoFile) == 0:
        print('Canceled, There is no infoFile. !!! STOP !!!')
        movie = []
        return movie, infoFile, infoFileRoi

    # --------------------------------------------------------------------------
    # Load Dax Files
    # --------------------------------------------------------------------------

    # --------- Determine number of frames to load
    framesInDax = infoFile["number_of_frames"]

    # parse now outdated 'allFrames' for backwards compatability
    if allFrames:
        endFrame = framesInDax
        startFrame = 0
    else:
        if endFrame < 0:
            endFrame = 0

    # parse startFrame and endFrame
    if endFrame < 0:
        endFrame = framesInDax
    if endFrame > framesInDax:
        if verbose:
            print(
                '[Warning]: input endFrame greater than total frames in dax_file.  Using all available frames after startFrame')
        endFrame = framesInDax

    framesToLoad = endFrame - startFrame + 1
    frameDim = [int(infoFile["frame_dimensions"][0] / infoFile["binning"][0]),
                int(infoFile["frame_dimensions"][1] / infoFile["binning"][1])]
    frameSize = frameDim[0] * frameDim[1]

    # Check memory requirements.  Ask for confirmation if > maxMemory.
    memoryRequired = frameSize * framesToLoad * 16 / 8
    DoThis = 1
    if memoryRequired > maxMemory:
        print("[Warning]: " + fileName + '  is > 2GB.\n Loading with warning...')

    # --------------------------------------------------------
    # Proceed to load specified poriton of daxfile
    # --------------------------------------------------------
    if DoThis:
        fileName = infoFile["localName"][:-4] + '.dax'
        if verbose:
            print('Loading ', infoFile["localPath"],"/", fileName,sep="")

        if 'little endian' in infoFile["data_type"]:
            binaryFormat = 'little'
            binaryFormat = '<'
        else:
            binaryFormat = 'big'
            binaryFormat = '>'

        # -----------------------------------------------------------------
        # Read all pixels from selected frames
        # -----------------------------------------------------------------
        if subregion == []:
            # MATLAB version:
            # fid = fopen([infoFile.localPath fileName]);
            # fseek(fid,(frameSize*(startFrame - 1))*16/8,'bof'); % bits/(bytes per bit)
            # dataSize = frameSize*framesToLoad;
            # movie = fread(fid, dataSize, '*uint16', binaryFormat);
            # fclose(fid);

            # Python version
            seek_pos = int(frameSize * startFrame * 2)  #bytes
            dataSize = int(frameSize * framesToLoad)

            dt = np.dtype("uint16")
            dt = dt.newbyteorder(binaryFormat)
            movie = np.fromfile(os.path.join(infoFile["localPath"], fileName), dtype = dt, count=dataSize,offset=seek_pos)

            try:  # Catch corrupt files
                if framesToLoad == 1:
                    movie = movie.reshape(frameDim)
                else:
                    if orientation == 'normal':
                        movie = np.transpose(movie.reshape(frameDim+[framesToLoad]), [1, 0, 2])
                    elif orientation == 'nd2':
                        movie = np.transpose(movie.reshape(np.fliplr(frameDim+[framesToLoad])), [1, 0, 2])
                    else:
                        pass
            except Exception as e:
                print('Serious error somewhere here...check file for corruption')
                movie = np.zeros(frameDim)

        # -----------------------------------------------------------------
        # Read the Indicated SubRegion using Memory Maps
        # -----------------------------------------------------------------
        else:
            # parse short-hand: xmin = 0 will start at extreme left
            #               ymax = 0 will go from ymin to the bottom
            xi = subregion[0]
            xe = subregion[1]
            yi = subregion[2]
            ye = subregion[3]
            if xi == 0:
                xi = 1
            if xe == 0:
                xe = frameDim[0]
            if yi == 0:
                yi = 1
            if ye == 0:
                ye = frameDim[1]
            # ------------------ arbitrary region ------------------------
            memoryMap = np.memmap(infoFile["localPath"]+os.path.sep+fileName,
                                  'Format', 'uint16',
                                  'Writable', False,
                                  'Offset', startFrame * frameSize * 2,
                                  'Repeat', framesToLoad * frameSize)

            [ri, ci, zi] = np.meshgrid([xi, xe], [yi, ye], [0, framesToLoad])
            inds = np.sub2indFast([frameDim[0], frameDim[1], framesInDax], ri[:], ci[:], zi[:])
            movie = memoryMap.Data[inds]
            movie = np.swapbytes(movie)
            xs = xe - xi + 1
            ys = ye - yi + 1
            movie = movie.reshape(xs, ys, framesToLoad)
            if 'normal' not in orientation:
                movie = np.permute(movie.reshape(xs, ys, framesToLoad), [2, 1, 3])

            infoFileRoi = infoFile
            infoFileRoi["hend"] = xs
            infoFileRoi["vend"] = ys
            infoFileRoi["frame_dimensions"] = [infoFile["hend"], infoFile["vend"]]
            infoFileRoi["file"] = os.path.join(infoFile["localPath"], infoFile["localName"][:-4] + '.dax')
            # --------------------------------------------------

        if verbose:
            print('Loaded', infoFile["localPath fileName"])
            print(f'{framesToLoad:3d}  {frameDim[0]:3d} * {frameDim[1]:3d} frames loaded')
    else:
        error('User aborted load dax due to memory considerations ')

    return movie, infoFile, infoFileRoi










