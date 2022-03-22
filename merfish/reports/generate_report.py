import os.path

import matplotlib.pyplot as plt
import numpy as np

from merfish.scripts.affine import transformPointsForward
from utils.misc import parula_map


def GenerateGeoTransformReport(tforms, residuals, **kwargs):
    # ------------------------------------------------------------------------
    # [report, figHandles, parameters] = GenerateGeoTransformReport(tforms, residuals)
    # This function generates a report on the generated geometric
    # transformations
    #--------------------------------------------------------------------------
    # Necessary Inputs:
    #   tforms -- A cell array corresponding to geometric transform objects
    #       (affine2d())
    #   residuals -- A cell array of the distance vectors between control
    #       points
    #   inds -- A cell array of index pairs
    #
    #   All inputs can be either 1D cell arrays or 2D cell arrays, in the later
    #   case they should correspond to bits x fov.
    #--------------------------------------------------------------------------
    # Outputs:
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #
    #--------------------------------------------------------------------------
    # Example Calls
    #
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    #--------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Default Reports to Generate
    # -------------------------------------------------------------------------
    defaultReports = {}
    defaultReports['residualTransformError'] = [1,1,1700,400]
    defaultReports['residualTransformErrorByPosition']= [1,1,1700,400]
    defaultReports['transformSummary'] = [1,1,1230,400]

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters={}
    parameters['reportsToGenerate']=defaultReports
    parameters['edges']=np.array([np.arange(1,512+25,25),np.arange(1,512+25,25)])

    # Generic display parameters
    parameters['saveAndClose']=True
    parameters['overwrite']=True
    parameters['formats']='png'

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    # -------------------------------------------------------------------------
    # Determine display type
    # -------------------------------------------------------------------------
    if np.any(np.ndim(tforms)==1):
        displayMethod = '1D'
    else:
        displayMethod = '2D'

    report = {}
    # -------------------------------------------------------------------------
    # Calculate error
    # -------------------------------------------------------------------------
    # Determine if error properties will be calculated
    if 'residualTransformError' in parameters["reportsToGenerate"]:
        # Allocate error properties
        # Mean residuals
        muX = np.empty(tforms.shape)
        muX[:]=np.nan
        muY = np.empty(tforms.shape)
        muY[:] = np.nan

        # STD in residuals
        stdX = np.empty(tforms.shape)
        stdX[:] = np.nan
        stdY = np.empty(tforms.shape)
        stdY[:] = np.nan
        # Number of control points
        numCP = np.empty(tforms.shape)
        stdY[:] = np.nan

        # Calculate errors
        numElements = np.size(tforms)
        for i in range(numElements):
            localResiduals = residuals.flat[i]
            if np.size(localResiduals) > 0:
                muX.flat[i] = np.mean(localResiduals[:,0])
                muY.flat[i] = np.mean(localResiduals[:,1])
                stdX.flat[i] = np.std(localResiduals[:,0])
                stdY.flat[i] = np.std(localResiduals[:,1])
                numCP.flat[i] = localResiduals.shape[0]

        # Archive in report
        report["muX"] = muX
        report["muY"] = muY
        report["stdX"] = stdX
        report["stdY"] = stdY
        report["numCP"] = numCP

    # -------------------------------------------------------------------------
    # Calculate position dependence
    # -------------------------------------------------------------------------
    if'residualTransformErrorByPosition' in parameters["reportsToGenerate"]:
        # Calculate magnitude and angle for all residuals
        allResiduals = np.concatenate(residuals.flatten(order="F"),0)
        if isinstance(allResiduals[0,0],(list,np.ndarray)): # Handle 2D cells
            allResiduals = np.concatenate(residuals.flatten(order="F"),0)

        # errMag = np.sqrt(np.sum(allResiduals[:,0:1]*allResiduals[:,0:1],1))
        # errAngle = np.arcsin( allResiduals[:,0] / errMag)

        # Define edges for binning
        edges = parameters["edges"]

        # Allocate memory
        errX = np.zeros((len(edges[0])-1, len(edges[1])-1))
        errY = np.zeros((len(edges[0])-1, len(edges[1])-1))
        numValues = np.zeros((len(edges[0])-1, len(edges[1])-1))

        # Fill
        for i in range(len(edges[0])-1):
            for j in range(len(edges[1])-1):
                inds = (allResiduals[:,2] >= edges[0][i]) & \
                       (allResiduals[:,2] < edges[0][i+1]) & \
                       (allResiduals[:,3] >= edges[1][j]) & \
                       (allResiduals[:,3] < edges[1][j+1])
                errX[i,j] = np.mean(allResiduals[inds,0])
                errY[i,j] = np.mean(allResiduals[inds,1])
                numValues[i,j] = np.sum(inds)

        # Archive in report
        report["errX"] = errX
        report["errY"] = errY
        report["numValues"] = numValues

    # -------------------------------------------------------------------------
    # Calculate position dependence
    # -------------------------------------------------------------------------
    if 'transformSummary' in parameters["reportsToGenerate"]:

        # Allocate variables
        offsetX = np.empty(tforms.shape)
        offsetX[:] = np.nan
        offsetY = np.empty(tforms.shape)
        offsetY[:] = np.nan
        angle = np.empty(tforms.shape)
        angle[:] = np.nan
        scale = np.empty(tforms.shape)
        scale[:] = np.nan

        # Unit vector in x direction
        unitVector = np.empty((2,2))
        unitVector[0,:] = [0,0]
        unitVector[1,:] = [1,0]

        # Calculate errors
        numElements = np.size(tforms)
        for i in range(numElements):
            if len(tforms.flatten()[i])>0:
                movedPoints = transformPointsForward(tforms.flatten()[i], unitVector[:,0],unitVector[:,1])

                offsetX.flat[i] = movedPoints[0,0] - unitVector[0,0]
                offsetY.flat[i] = movedPoints[0,1] - unitVector[0,1]

                diffVector = np.diff(movedPoints,axis=0)

                scale.flat[i] = np.sqrt(np.sum(diffVector**2))
                degree_i = np.sum(diffVector*unitVector[1,:])/scale.flat[i]
                angle.flat[i] = np.rad2deg(np.arccos(degree_i))

        # Archive values
        report["offsetX"] = offsetX
        report["offsetY"] = offsetY
        report["scale"] = scale
        report["angle"] = angle

    # -------------------------------------------------------------------------
    # Display Transform Error Report
    # -------------------------------------------------------------------------
    figHandles = []
    if "reportPath" in parameters:
        reportPath = parameters["reportPath"]
    else:
        print("reportPath is not provided, Reports/Figures will not be saved!.")
        parameters["saveAndClose"]=False

    if "residualTransformError" in  parameters["reportsToGenerate"]:
        reportID = parameters["reportsToGenerate"]["residualTransformError"]
        file_name = 'Geometric Transformation Error Report'
        fig = plt.figure(file_name,figsize=(18,6))
        data = [muX, muY, numCP, stdX, stdY]
        titles = ['Residual X', 'Residual Y', 'Number of Points', 'STD X', 'STD Y']
        quantileRanges = [[0.1,0.9], [0.1,0.9], [0,1], [0,0.9], [0,0.9]]
        if displayMethod == '1D':
            for i in range(len(data)):
                fig.add_subplot(2,3,i+1)
                plt.imshow(data[i], cmap=parula_map)
                plt.xlabel('FOV')
                plt.ylabel(titles[i])
                localData = data[i].copy()
                limits = np.quantile(localData[:], quantileRanges[i])
                if np.diff(limits) > 0 and not np.any(np.isinf(limits)):
                    plt.ylim(limits)

        if displayMethod == '2D':
            for i in range(len(data)):
                plt.subplot(2,3,i+1)
                plt.imshow(data[i],cmap=parula_map,aspect='auto')
                plt.xlabel('FOV')
                plt.ylabel('Imaging Rounds')
                plt.title(titles[i])
                plt.colorbar()
                localData = data[i].copy()
                limits = np.quantile(localData[:], quantileRanges[i])
                if np.diff(limits) > 0 and not np.any(np.isinf(limits)):
                    plt.clim(limits[0],limits[1])
        plt.tight_layout()
        if parameters["saveAndClose"]:
            plt.savefig(os.path.join(reportPath,file_name+"."+parameters['formats']))
        plt.close()

    # -------------------------------------------------------------------------
    # Display Position Dependence of Transform Error Report
    # -------------------------------------------------------------------------
    if 'residualTransformErrorByPosition' in parameters["reportsToGenerate"]:
        file_name = 'Geometric Transformation Error Position Dependence Report'
        fig = plt.figure(file_name,figsize=(18,5))
        data = [errX, errY, numValues]
        titles = ['X', 'Y', 'Number']
        quantileRanges = [[.1,.9], [.1,.9], [0,1]]

        for i in range(len(data)):
            localData = data[i].copy()
            localData[np.isnan(localData)] = 0
            fig.add_subplot(1,3,i+1)
            plt.pcolor(edges[0][:-1]+np.diff(edges[0]), edges[1][:-1]+np.diff(edges[1]),localData,cmap=parula_map,shading='nearest')
            plt.xlabel('Position X (Pixels)')
            plt.ylabel('Position Y (Pixels)')
            plt.title(titles[i])
            plt.colorbar()
            limits = np.quantile(localData[:], quantileRanges[i])
            if np.diff(limits) > 0 and not np.any(np.isinf(limits)):
                plt.clim(limits[0], limits[1])

        plt.tight_layout()
        if parameters["saveAndClose"]:
            plt.savefig(os.path.join(reportPath, file_name + "." + parameters['formats']))
        plt.close()

    # -------------------------------------------------------------------------
    # Display properties of the transforms
    # -------------------------------------------------------------------------
    if 'transformSummary'in parameters['reportsToGenerate']:
        file_name='Geometric Transformation Summary'
        fig = plt.figure(file_name,figsize=(12,6))
        data = [offsetX, offsetY, scale, angle]
        titles = ['Offset X', 'Offset Y', 'Scale', 'Angle']
        zlabels = ['Pixels', 'Pixels', 'Fraction', 'Degrees']
        quantileRanges = [[.1,.9], [.1,.9], [.1,.9], [.1,.9]]
        if displayMethod == '1D':
            for i in range(len(data)):
                localData = data[i].copy()
                fig.add_subplot(2,2,i+1)
                plt.plot(localData,aspect="auto")
                plt.xlabel('FOV')
                plt.ylabel(titles[i])
                limits = np.quantile(localData[:], quantileRanges[i])
                if np.diff(limits) > 0 and not np.any(np.isinf(limits)):
                    plt.ylim(limits)
                plt.ylabel(zlabels[i])
        if displayMethod == '2D':
            for i in range(len(data)):
                fig.add_subplot(2,2,i+1)
                plt.imshow(data[i],cmap=parula_map, aspect='auto')
                plt.xlabel('FOV')
                plt.ylabel('Imaging Rounds')
                plt.title(titles[i])
                plt.colorbar()
                localData = data[i].copy()
                limits = np.quantile(localData[:], quantileRanges[i])
                if np.diff(limits) > 0 and not np.any(np.isinf(limits)):
                    plt.clim(limits[0], limits[1])
                # plt.zlabel(zlabels[i])
        plt.tight_layout()
        if parameters["saveAndClose"]:
            plt.savefig(os.path.join(reportPath, file_name + "." + parameters['formats']))
        plt.close()

    return report


