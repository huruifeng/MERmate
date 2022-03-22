import os
import subprocess
from os.path import dirname, basename, splitext
from xml.dom import minidom

import numpy as np

from storm.mufit_analysis import mufit_analysis
from utils.funcs import error, PageBreak, tic, toc


def WriteDaoSTORMParameters(filePath, **kwargs):
    # ------------------------------------------------------------------------
    # [parameters, xmlObj] = CreateDaoSTORMParameters(varargin)
    # This function creates a structure with fields the contain all of the
    # parameters needed for 3D daoSTORM analysis.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # -- filePath: A string specifying the location of the file to write
    # (provide an empty array to just return the parameters structure without
    #  saving a file)
    #--------------------------------------------------------------------------
    # Outputs
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # jeffmoffitt@gmail.com
    # March 9, 2016
    #--------------------------------------------------------------------------
    # Creative Commons License CC BY NC SA
    #--------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    defaults = {}
    # Parameters controlling this function, not daoSTORM
    defaults['verbose']= False

    # Parameters describing the frames and AOI to analyze
    defaults['start_frame']= -1 # The initial frame for analysis
    defaults['max_frame']= -1   # The final frame for analysis (-1 implies all frames)

    defaults['x_start']= 0      # The first x pixel
    defaults['x_stop']= 2048       # The last x pixel
    defaults['y_start']= 0   # The first y pixel
    defaults['y_stop']= 2048    # The last y pixel

    # Parameters describing the fitting
    defaults['model'] = '2d' # {'2dfixed', '2d', '3d', 'Z'}
    #   2dfixed - fixed sigma 2d gaussian fitting.
    #   2d - variable sigma 2d gaussian fitting.
    #   3d - x, y sigma are independently variable,
    #          z will be fit after peak fitting.
    #   Z - x, y sigma depend on z, z is fit as
    #          part of peak fitting.

    defaults['iterations']= 20     # The number of iterations to perform
    defaults['threshold']= 100.0   # The minimum brightness
    defaults['sigma']= 1.0         # The initial guess for the width (in pixels)

    # Parameters describing the camera
    defaults['baseline']= 100.0    # The background term of the CCD
    defaults['pixel_size']= 160.0  # The pixel size
    defaults['orientation']='normal' #{'normal', 'inverted'}, 'normal'} # The orientation of the CCD

    # Parameters for multi-activator STORM
    defaults['descriptor'] = '1' # {'0', '1', '2', '3', '4'}

    # Parameters for peak matching
    defaults['radius']= 0   # Radius in pixels to connect molecules between frames
                                                        # 0 indicates no connection

    # Parameters for Z fitting
    defaults['do_zfit']= 0  # Should z fitting be performed
    defaults['cutoff']= 1.0

    defaults['wx_wo']= 238.3076
    defaults['wx_c']= 415.5645
    defaults['wx_d']= 958.792
    defaults['wxA']= -7.1131
    defaults['wxB']= 19.9998
    defaults['wxC']= 0.0
    defaults['wxD']= 0.0

    defaults['wy_wo']= 218.9904
    defaults['wy_c']= -310.7737
    defaults['wy_d']= 268.0425
    defaults['wyA']= 0.53549
    defaults['wyB']= -0.099514
    defaults['wyC']= 0.0
    defaults['wyD']= 0.0

    defaults['min_z']= -0.5
    defaults['max_z']= 0.5

    # Parameters for drift correction
    defaults['drift_correction']= 0 # Should drift correction be applied
    defaults['frame_step']= 8000
    defaults['d_scale']= 2

    # -------------------------------------------------------------------------
    # Define fields and types
    # -------------------------------------------------------------------------
    fieldsAndTypes = [
        ['start_frame', 'int'], ['max_frame', 'int'],
        ['x_start', 'int'], ['x_stop', 'int'],
        ['y_start', 'int'], ['y_stop', 'int'],
        ['model', 'string'], ['iterations', 'int'],
        ['baseline', 'float'], ['pixel_size', 'float'],
        ['orientation', 'string'], ['threshold', 'float'],
        ['sigma', 'float'], ['descriptor', 'string'],
        ['radius', 'float'], ['do_zfit', 'int'],
        ['cutoff', 'float'],
        ['wx_wo', 'float'], ['wx_c', 'float'],
        ['wx_d', 'float'], ['wxA', 'float'],
        ['wxB', 'float'], ['wxC', 'float'],
        ['wxD', 'float'],
        ['wy_wo', 'float'], ['wy_c', 'float'],
        ['wy_d', 'float'], ['wyA', 'float'],
        ['wyB', 'float'], ['wyC', 'float'],
        ['wyD', 'float'],
        ['min_z', 'float'], ['max_z', 'float'],
        ['drift_correction', 'int'], ['frame_step', 'int'],
        ['d_scale', 'int']]

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    parameters =defaults
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    # -------------------------------------------------------------------------
    # Check for requested file save
    # -------------------------------------------------------------------------
    if not filePath:
        return

    # -------------------------------------------------------------------------
    # Check for xml extension
    # -------------------------------------------------------------------------
    [_,fileExt] = os.path.splitext(filePath)
    if fileExt != '.xml':
        error('STORM:invalidArguments - Provided path must be to an xml file')

    # -------------------------------------------------------------------------
    # Create XML Object
    # -------------------------------------------------------------------------
    # Create the root element
    xmlObj = minidom.Document()
    xml = xmlObj.createElement('settings')
    xmlObj.appendChild(xml)
    # Loop over parameters fields and create individual nodes for each
    for f in fieldsAndTypes:
        localFieldAndType = f

        # Create node and define type
        node = xmlObj.createElement(localFieldAndType[0])
        node.setAttribute('type', localFieldAndType[1])

        # Find corresponding parameter and coerce value to string
        localValue = str(parameters[localFieldAndType[0]])

        node.appendChild(xmlObj.createTextNode(localValue))
        # Append to root element
        xml.appendChild(node)

    # -------------------------------------------------------------------------
    # Write xml
    # -------------------------------------------------------------------------
    xml_str = xmlObj.toprettyxml(indent="\t")
    with open(filePath, "w") as f:
        f.write(xml_str)

    if parameters["verbose"]:
        print('Wrote:',filePath)
    return [parameters, xml]


###################################################################################
def daoSTORM(filePaths, configFilePaths, **kwargs):
    # ------------------------------------------------------------------------
    # daoSTORM(filePaths, configFilePath, ...) is a light weight wrapper around daoSTORM
    #--------------------------------------------------------------------------
    # Necessary Inputs:
    #   filePath -- A cell array of filePaths to analyze or a single file path.
    #   configFilePaths -- The path to a daoSTORM configuration file or a cell
    #   array of such paths. If a cell array is provided it must match the
    #   length of the cell array of filePaths.
    #--------------------------------------------------------------------------
    # Outputs:
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # mListType -- A prefix that will be added before the mList tag (default is
    #   empty)
    # savePath -- The file path in which molecule lists will be saved (default
    #   is the data directory)
    # overwrite -- A boolean determining if existing files will be overwritten
    #   or ignored.
    # numParallel -- The number of parallel processes to launch. Default is 1.
    # hideterminal -- Hide the command windows that are launched?
    # waitTime -- The number of seconds to pause before querying the job queue
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # jeffmoffitt@gmail.com
    # March 11, 2016
    #--------------------------------------------------------------------------
    # Creative Commons License CC BY NC SA
    #--------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}

    # Parameters for saving mLists
    parameters['mListType']= ''        # A prefix to add before the mList tag
    parameters['savePath'] = ''        # The path where mLists will be saved (default, same path as filePath)
    parameters['overwrite']= True     # Overwrite existing files?
    parameters['tryToComplete'] = False # If files will not be overwritten, still call daoSTORM to see if they need to be completed (i.e. analysis was not previously finished)

    # Parameters for batch processing
    parameters['numParallel'] = 1     # The number of parallel processes to launch
    parameters['hideterminal']= False # Hide the analysis terminal
    parameters['waitTime'] = 15    # The number of seconds to pause between pooling job queue

    # Parameters for displaying progress
    parameters['verbose']= True       # Report progress?
    parameters['veryVerbose']= True   # Display all commands being called?
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not filePaths or not configFilePaths:
        error('matlabFunctions:invalidArguments - A path and a configuration file must be provided.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    # -------------------------------------------------------------------------
    # Coerce file paths to cell arrays
    # -------------------------------------------------------------------------
    if isinstance(filePaths,str):
        filePaths = [filePaths]
    if isinstance(configFilePaths,str):
        configFilePaths = [configFilePaths]
    if len(configFilePaths) == 1:
        configFilePaths = configFilePaths * len(filePaths)
    if len(configFilePaths) != len(filePaths):
        error('[error]:invalidArguments - The number of configuration files does not match the number of requested files')

    # -------------------------------------------------------------------------
    # Confirm that requested paths exist
    # -------------------------------------------------------------------------
    if not np.all([os.path.exists(i) for i in filePaths]):
        print('[error]:invalidArguments - Some requested file paths do not exist')
        # Display missing file paths to aid in troubleshooting
        if parameters["verbose"]:
            ind = np.nonzero([not os.path.exists(i) for i in filePaths])[0]
            for i in ind:
                print('... Missing:',filePaths[i])
        raise('[error]:invalidArguments - Some requested file paths do not exist')

    if not np.all([os.path.exists(i) for i in configFilePaths]):
        error('[error]:invalidArguments - Some requested configuration files do not exist')


    # -------------------------------------------------------------------------
    # Confirm that the same file is not analyzed twice (not yet supported)
    # -------------------------------------------------------------------------
    if len(filePaths) != len(np.unique(filePaths)):
        error('[error]:invalidArguments - Analysis of the same file twice is not yet supported.')

    # -------------------------------------------------------------------------
    # Handle prefix formatting
    # -------------------------------------------------------------------------
    if parameters["mListType"]=="":
        parameters["mListType"] = 'mList'

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        PageBreak()
        print('Analyzing',len(filePaths),'files')
        uniqueConfigFiles = np.unique(configFilePaths)
        if len(uniqueConfigFiles) == 1:
            print('...configuration file:',uniqueConfigFiles[0])
        else:
            print('...with',len(uniqueConfigFiles),'unique configuration files')
        print('...saving as',parameters["mListType"],'bin files')
        if parameters["savePath"]=="":
            print('...in the original data location')
        else:
            print('...here:',parameters["savePath"])

    # -------------------------------------------------------------------------
    # Create bin file paths
    # -------------------------------------------------------------------------
    binFilePaths = []
    indsToKeep = np.ones((len(filePaths),),dtype=np.bool)
    numDeleted = 0
    for f in filePaths:
        # Strip parts
        filePath = dirname(f)
        baseName = basename(f)
        fileName,fileExt = splitext(baseName)


        # Check for correct file extension
        if fileExt not in ['.dax', '.tif', '.tiff']:
            error('[error]:invalidFileExtension' + filePaths[f]+' contains an invalid extension.')

        # Create bin file path
        if parameters["savePath"]:
            binFilePath_f = os.path.join(parameters["savePath"], fileName + '_' + parameters["mListType"] + '.bin')
        else:
            binFilePath_f = os.path.join(filePath, fileName + '_' + parameters["mListType"] + '.bin')
        binFilePaths.append(binFilePath_f)

        # Check if it exists
        if os.path.exists(binFilePath_f):
            if parameters["overwrite"]:
                os.remove(binFilePath_f)
                numDeleted = numDeleted + 1
                if parameters["veryVerbose"]:
                    print('... Overwriting, deleted: ',binFilePath_f)
            elif not parameters["tryToComplete"]: # If tryToComplete is requested (and overwrite is not), keep these files and call daoSTORM, which will check to see if they need to be finalized
                indsToKeep[f] = False
                if parameters["veryVerbose"]:
                    print('... found and ignoring:',binFilePath_f)
    # Update list of data files and configuration files
    filePaths = np.array(filePaths)[indsToKeep]
    binFilePaths = np.array(binFilePaths)[indsToKeep]
    configFilePaths = np.array(configFilePaths)[indsToKeep]

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        print('...overwriting',numDeleted,'files')
        print('...ignoring',np.sum(~indsToKeep),'files')


    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        PageBreak()
        print('Starting running daoSTROM:',tic(1))
        batchTimer = tic(0)

    #--------------------------------------------------------------------------
    # Run analysis
    #--------------------------------------------------------------------------
    commands = {}
    for i in range(len(filePaths)):
        PageBreak()
        print("Analyzing: "+filePaths[i])
        mufit_analysis(filePaths[i],binFilePaths[i],configFilePaths[i])

    # -------------------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------------------
    if parameters["verbose"]:
        print('...completed at',tic(1))
        print('...in',toc(batchTimer),'s')

    return parameters






