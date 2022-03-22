import os

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype

from utils.funcs import *


def LoadCodebook(codebookPath, **kwargs):
    # ------------------------------------------------------------------------
    # [codebook, header] = LoadCodebook(codebookPath, **kwargs)
    # This function creates a codebook structure from codebook file
    # --------------------------------------------------------------------------
    # Necessary Inputs
    # --codebookPath: A path to a valid codebook
    #
    # --------------------------------------------------------------------------
    # Outputs
    # --codebook: A structure array with the fields provided in the codebook file.
    # These must include name and barcode fields.
    # --header: A structure with fields containing the initial header information of the codebook.
    # Required fields are:
    #     --version: A string specifying the format version of the codebook
    #     --bit_names: A list of the names of the bits in the order in which they occur in the barcodes.
    # --------------------------------------------------------------------------
    # Original version (in MATLAB):
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # September 21, 2017
    # Copyright Presidents and Fellows of Harvard College, 2018.
    # --------------------------------------------------------------------------
    # This python version is developed by Ruifeng Hu from the Original version
    # 10-16-2021
    # huruifeng.cn@hotmail.com

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not os.path.exists(codebookPath):
        error('Error:invalidArguments - A valid path to a codebook is required')

    ## initialize the values:
    if "verbose" not in kwargs: kwargs["verbose"] = True
    if "barcodeConvFunc" not in kwargs: kwargs["barcodeConvFunc"] = ""

    arg_ls = ["verbose", "barcodeConvFunc"]
    parameters = ParseArguments(kwargs, arg_ls)
    # print(kwargs)

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        PageBreak()
        print('Loading codebook from:', codebookPath)

    # -------------------------------------------------------------------------
    # Open the file
    # -------------------------------------------------------------------------
    fid = open(codebookPath, 'r')
    # -------------------------------------------------------------------------
    # Load header
    # -------------------------------------------------------------------------
    header = {}
    line = fid.readline()
    while line:
        # Load line and split on comma
        line_ls = line.strip().split(',')

        # Remove whitespace
        stringParts = [i.strip() for i in line_ls]

        # If the line is name, id, barcode, then the header portion of the code book
        if stringParts[0] == 'name' and stringParts[1] == 'id' and stringParts[2] == 'barcode':
            break

        # Assign name value pairs
        if len(stringParts) == 2:
            header[stringParts[0]] = stringParts[1]
        else:
            header[stringParts[0]] = stringParts[1:]

        line = fid.readline()

    # -------------------------------------------------------------------------
    # Check header
    # -------------------------------------------------------------------------
    if not (("version" in header) and ("bit_names" in header)):
        error('Error:invalidArguments', 'The codebook is corrupt. Both a version and bit_names flag must be present.')

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    # print(header)
    if parameters["verbose"]:
        for f in header:
            data = header[f]
            if isinstance(data, list):
                data = ",".join(data)
            print('...', f, ':', data)

    # -------------------------------------------------------------------------
    # Build codebook
    # -------------------------------------------------------------------------
    # Switch based on version
    codebook = {'name': [], 'id': [], 'barcode': []}
    if header["version"] in ['1.0', '1']:
        line = fid.readline()
        while line:
            # Read line, split, and assign values
            line_ls = line.strip().split(',')

            # Remove whitespace
            stringParts = [i.strip() for i in line_ls]

            codebook["name"].append(stringParts[0])
            codebook["id"].append(stringParts[1])
            if parameters["barcodeConvFunc"]=="":
                codebook["barcode"].append(np.array(list(stringParts[2])))
            if callable(parameters["barcodeConvFunc"]):
                codebook["barcode"].append(parameters["barcodeConvFunc"](stringParts[2].replace(" ","")))

            line = fid.readline()
    else:
        error("Error: codebook version error.")

    # -------------------------------------------------------------------------
    # Close the file
    # -------------------------------------------------------------------------
    fid.close()

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        print('... loaded', len(codebook['name']), 'barcodes')

    return [codebook, header, parameters]


def LoadDataOrganization(dataOrgPath,**kwargs):
    # dataOrg = LoadDataOrganization(dataOrgPath, varargin)
    # This function opens a data organization file
    #
    #--------------------------------------------------------------------------
    # Necessary Inputs:
    #   dataOrgPath -- A valid path to a data organization file
    #--------------------------------------------------------------------------
    # Outputs:
    #   --None
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    #-----------------------------------------------------------------------------------------------------------------------------------
    # This function loads a data organization csv file

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}

    # Parameters for displaying progress
    parameters['verbose']  = False      # Display progress?

    # Parameters for handling internal delimiters in fields
    parameters['internalDelimiter'] = ';' # The internal delimiter for individual fields

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not os.path.exists(dataOrgPath):
        error('Error:invalidArguments - A valid path to the data organization file is required')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    #------------------------------------------------------------------------
    # Load data organization file
    # -------------------------------------------------------------------------
    if dataOrgPath.endswith(".csv"):
        # Load csv file using the table2struct approach
        dataOrg = pd.read_csv(dataOrgPath)
    else:
        error('[Error]:invalidFileType, Only csv files are supported.')

    # -------------------------------------------------------------------------
    # Handle internal delimiters
    # -------------------------------------------------------------------------
    possibleFields = {'frame', 'zPos'}; # Fields that could have internal delimiters
    for f in possibleFields:
        if f in dataOrg: # Confirm that the field exists
            if is_string_dtype(dataOrg[f]): # Check to see if it is a char--a flag that it could not be converted to numbers
                dataOrg[f] = dataOrg[f].apply(lambda x: [float(i) for i in x.split(";") if i !=""])


    # -------------------------------------------------------------------------
    # Build metadata structure
    # -------------------------------------------------------------------------
    metaData = {}
    # The number of data channels
    metaData["numDataChannels"] = len(dataOrg)

    # The number and location of z-stacks
    if 'zPos' in dataOrg:
        zPos_ls = []
        for i in dataOrg["zPos"]:
            if isinstance(i,list):
                zPos_ls += i
            elif isinstance(i,(int,str)):
                zPos_ls.append(i)
        uniqueZPos = np.unique(zPos_ls)
    else:
        uniqueZPos = 0
        # Add zPos to each dataOrg entry
        dataOrg["zPos"] = 0

    metaData["zPos"] = np.sort(uniqueZPos) # Order the z from small to large
    metaData["numZPos"] = len(uniqueZPos)

    # -------------------------------------------------------------------------
    # Display data organization file if requested
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        PageBreak()
        print('Loaded data organization file:',dataOrgPath);
        print('Found',metaData["numDataChannels"],"data channels")
        print('Found',metaData["numZPos"],'z-stacks')

    return [dataOrg, metaData, parameters]

def ReadBinaryFileHeader(filePath):
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not(filePath and os.path.exists(filePath)):
        error('[Error]:invalidArguments - A valid path must be provided.')

    # -------------------------------------------------------------------------
    # Open file
    # -------------------------------------------------------------------------
    data_df = pd.read_pickle(filePath)
    fileHeader = data_df.columns.tolist()
    return fileHeader

def ReadBinaryFile(filePath):
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not(filePath and os.path.exists(filePath)):
        error('[Error]:invalidArguments - A valid path must be provided.')

    # -------------------------------------------------------------------------
    # Open file
    # -------------------------------------------------------------------------
    data_df = pd.read_pickle(filePath)
    return data_df

def WriteBinaryFile(filePath, structArray, **kwargs):
    # ------------------------------------------------------------------------
    # WriteBinaryFile(filePath, struct, varargin)
    # This function writes a structure array to a custom binary file format
    # defined by the fields in the structure array.
    #
    # Structure fields may be arrays of any size and format but every instance
    # of this field in each structure in the array MUST be the same size and
    # type.
    #--------------------------------------------------------------------------
    # Necessary Inputs:
    #   filePath -- A valid path to the file that will be saved
    #   structArray -- The array to be saved.
    #--------------------------------------------------------------------------
    # Outputs:
    #   --None
    #--------------------------------------------------------------------------
    # File organization
    # Version number -- A uint8 that specifies the reader/writer version number
    # Corrupt -- A uint8 that is set to 1 when the file is opened and 0 to when
    #    closed. If the file is improperly closed, a 1 will indicate that it is
    #    corrupt
    # Number of entries -- A uint32 that specifies the number of entries
    # Header length -- A uint32 that specifies the length of the following
    # header
    # Header -- A character array that can be parsed to
    #    determine the layout of the file. Each entry is written as follows
    #    field name ,  field dimensions ,  field class ,
    #
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}

    # Parameters for displaying progress
    parameters['verbose']=False  # Display progress?
    parameters['overwrite']=True # Overwrite file? If not, append data
    parameters['append'] = False    # Append to file, but only if overwrite is not true

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not isinstance(structArray,(dict,list,pd.DataFrame)):
        error('[Error]]:invalidArguments - A path and a pd.DataFrame must be provided.');

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    # -------------------------------------------------------------------------
    # Parse structure
    # -------------------------------------------------------------------------
    # Determine fields and number of entries
    fieldsToWrite = structArray.columns.tolist()
    numEntries = structArray.shape[0]

    # -------------------------------------------------------------------------
    # Check to see if file exists
    # -------------------------------------------------------------------------
    isAppend = False
    if os.path.exists(filePath):
        if parameters["overwrite"] and not parameters["append"]: # If overwriting, delete file
            os.remove(filePath)
            if parameters["verbose"]:
                print('Deleting existing file:',filePath)
        elif not parameters["append"]:  # If not overwriting, and not appending generate error
            error('[Error]: existingFile - Found existing file.')
        else: # The file exists and the user has explicitly requested to append
            # Read previous file header
            fileHeader = ReadBinaryFileHeader(filePath)

            # Confirm that the file layout is identical
            if len(fileHeader) != len(fieldsToWrite) or not np.all([fieldsToWrite[i] == fileHeader[i] for i in range(len(fileHeader))]):
                error('[Error]:incompatibleHeaders - Cannot append binary data with a different organization')

            # Set the isAppend flag
            isAppend = True
        # Run checks

    # -------------------------------------------------------------------------
    # Handle the appending/writing scenarios
    # -------------------------------------------------------------------------
    if not isAppend:
        # storing data in pickle file format
        structArray.to_pickle(filePath)

    else: # Appending
        storedata = pd.read_pickle(filePath)
        storedata = storedata.append(structArray, ignore_index=True)
        storedata.to_pickle(filePath)

    return parameters



