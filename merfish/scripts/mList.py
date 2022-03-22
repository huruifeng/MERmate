import os

import numpy as np
from matplotlib import pyplot as plt
from random import random
from skimage.transform import AffineTransform

from merfish.scripts.affine import fitGeoTrans, transformPointsForward_x
from utils.funcs import error, knnsearch2d
from utils.misc import ind2sub


def MLists2Transform(refList, mList, **kwargs):
    # ------------------------------------------------------------------------
    # [tforms, mList, residuals, inds, parameters] = MLists2Transform(refList, mList, varargin)
    # This function returns a set of geometric transforms based on a set of
    # control points found in each frame of both the reference mList, refList,
    # and the mList to be warped.
    #--------------------------------------------------------------------------
    # Necessary Inputs: 
    #   refList -- A molecule list with the following fields: x, y, and frame.
    #   mList -- A molecule list with the following fields: x, y, and frame.
    #       See ReadMasterMoleculeList for information on molecule lists.
    #       Molecule lists must be in the compact form.
    #--------------------------------------------------------------------------
    # Outputs: 
    #   tforms -- A cell array of geometric transform objects for each frame in
    #       the specified molecule lists
    #   mList -- The original moving mList but with xc and yc values updated to
    #       reflect the transformation. Note these are updated only if the 
    #       'applyTransform' flag is true. 
    #   residuals -- A cell array of the residual distances between control
    #       points after the transformation is applied. Each entry is a Nx4 set
    #       of points where the first two columns represent the residual error
    #       vectors and the last two the position of the original points.
    #       Note that these are updated only if the 'applyTransform' flag is 
    #       true.
    #   inds -- A cell array of the points in each frame of the mList that
    #       that were assigned to points in the refList
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    # 
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # jeffrey.moffitt@childrens.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    #--------------------------------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    
    # Parameters for the geometric transformation
    parameters['transformationType'] = "nonreflectivesimilarity" # {'nonreflectivesimilarity', 'similarity', 'affine', 'projective', 'polynomial'} # Type of geometric transformation
    parameters['polynomialOrder'] = 3 # Order of the polynomial fit
    
    # Parameters for control point association
    parameters['controlPointMethod'] = "nearestNeighbor" # {'nearestNeighbor', 'kNNDistanceHistogram'} # The method used to associate points in the refList and the mList
    parameters['distanceWeight'] = 0.25   # The weight applied to the STD of the distances for valid control point selection
    parameters['thetaWeight'] = 0.25     # The weight applied to the STD of the angles for valid control point selection
    parameters['histogramEdges'] = np.arange(-128,129)# The bin edges for the point different histogram
    parameters['numNN'] = 10               # The number of nearest neighbors to compute
    parameters['pairDistTolerance'] = 1    # The distance threshold to find paired points after crude shift (multiples of the histogramEdges step)
    
    # Apply transformation
    parameters['applyTransform'] = True
    
    # How to handle molecules in different frames
    parameters['ignoreFrames'] = False  # If false, combine all molecules in different frames
    parameters['transpose'] = False  # Whether to transpose data in X/Y or not
    
    # Reporting/Debugging: WILL BE REMOVED IN THE FUTURE
    parameters['debug'] = True
    parameters['displayFraction'] = 0.2
    parameters["debugFolder"] = ""
    
    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if len(np.setdiff1d(['x', 'y', 'frame'], list(refList.columns))) > 0 \
            or len(np.setdiff1d(['x', 'y', 'frame'], list(refList.columns))) > 0 :
        error('invalidArguments - Two valid molecule lists must be provided.')
    
    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k in kwargs:
        parameters[k] = kwargs[k]
    
    # -------------------------------------------------------------------------
    # Define default values
    # -------------------------------------------------------------------------
    if parameters["ignoreFrames"]:
        numFrames = 1
    else:
        numFrames = np.max(refList["frame"])

    residuals = {}
    inds = {}
    tforms = np.tile(AffineTransform(),(numFrames,)) # Default is no transformation
    
    # -------------------------------------------------------------------------
    # Loop over individual frames
    # -------------------------------------------------------------------------
    for f in range(numFrames):
        if parameters["ignoreFrames"]:
            if parameters["transpose"]:
                refPoints = np.stack([refList["x"].values,refList["y"].values])
                movPoints = np.stack([mList['x'].values,mList['y'].values])
            else:
                refPoints = np.stack([refList["x"].values,refList["y"].values],axis=-1)
                movPoints = np.stack([mList["x"].values,mList["y"].values],axis=-1) # Moving points: to warp
        else:
            if parameters["transpose"]:
                # Extract points points for each frame
                refPoints = np.stack([refList["x"][refList["frame"] == f].values,refList["y"][refList["frame"] == f].values])
                movPoints = np.stack([mList["x"][mList["frame"] == f].values, mList["y"][mList["frame"] == f].values]) # Moving points: to warp
            else:
                # Extract points points for each frame
                refPoints = np.stack([refList["x"][refList["frame"] == f].values, refList["y"][refList["frame"] == f].values],axis=-1)
                movPoints = np.stack([mList["x"][mList["frame"] == f].values, mList["y"][mList["frame"] == f].values],axis=-1) # Moving points: to warp
        
        # Check for empty mLists
        if len(refPoints) == 0:
            print('[Warning]: emptyMList - No molecules in reference list. Using default transformation.')
            residuals[f] = np.zeros((0,4))
            continue
        elif len(movPoints) == 0:
            print('[Warning]: emptyMList - No molecules in moving list. Using default transformation.')
            residuals[f] =  np.zeros((0,4))
            continue
    
        # Use different methods to identify control point pairs
        if parameters["controlPointMethod"]== 'nearestNeighbor':
            # Find the nearest neighbors and distances
            [idx, D] = knnsearch2d(movPoints,refPoints)

            # Find angle of vector between nearest neighbors
            theta = np.arcsin((movPoints[idx,1]-refPoints[:,1])/D )

            if not np.all(D==0):
                # Define thresholds for excluding outliers
                DBounds = np.median(D) + np.std(D)*parameters["distanceWeight"]*[-1,1]
                thetaBounds = np.median(theta)+ np.std(theta)*parameters["thetaWeight"]*[-1,1]

                # Define the control points to keep
                pointsToKeep = np.nonzero((D>=DBounds[0]) & (D<=DBounds[1]) & (theta >= thetaBounds[0]) & (theta <= thetaBounds[1]))[0]
            else: # If the lists match exactly
                pointsToKeep = np.arange(0,len(idx))

        elif parameters["controlPointMethod"]==  'kNNDistanceHistogram':
            # Calculate differences in X and Y for all neighborhoods of
            # Find the nearest neighbors and distances
            [idx,_] = knnsearch2d(movPoints,refPoints, k= parameters["numNN"])

            # Flatten indices
            idx1 = idx.flatten()
            idx2 = np.tile(np.arange(0,refPoints.shape[0]), (parameters["numNN"],))

            # Handle the case that there are less than 'K' points
            idx2 = idx2[0:len(idx1)]

            # Define distances
            diffX = movPoints[idx1,0] - refPoints[idx2,0]
            diffY = movPoints[idx1,1] - refPoints[idx2,1]

            # Calculate 3D histogram
            N,_,_ = np.histogram2d(diffX,diffY, bins=(parameters["histogramEdges"], parameters["histogramEdges"]))

            # Find peak position
            maxValue = np.max(N)
            [xpeak,ypeak] = np.unravel_index(np.argmax(N, axis=None), N.shape)

            # Determine offsets
            offset = [parameters["histogramEdges"][xpeak], parameters["histogramEdges"][ypeak]]

            # Assign control points from nearest neighbors
            [idx, D] = knnsearch2d(movPoints - np.tile(offset, (movPoints.shape[0],1)), refPoints)
            idx = idx[0]
            D = D[0]
            # Keep all points separated by less than the pixelation of the
            # crude shift
            pointsToKeep = np.nonzero(D <= parameters["pairDistTolerance"]*np.mean(np.diff(parameters["histogramEdges"])))[0]

            # Issue warnings
            if maxValue < np.mean(N) + 2*np.std(N):
                print('[Warning]: controlPoints - The maximum kNN offset is <2 times the std from the mean. Control point identification may be inaccurate')
    
        # Build transform
        try:
            if parameters["transformationType"]=='polynomial':
                tforms[f] = fitGeoTrans(movPoints[idx[pointsToKeep],:], refPoints[pointsToKeep, :], parameters["transformationType"],parameters["polynomialOrder"])
            else:
                tforms[f] = fitGeoTrans(movPoints[idx[pointsToKeep],:], refPoints[pointsToKeep, :], parameters["transformationType"])
        except:
            print('[Warning]: Did not find sufficient control points. Using default transformation')

        # Archive control point indices
        inds[f] = np.stack([pointsToKeep,idx[pointsToKeep]],axis=1) # Indices of reference points indices of moving points
        
        # Apply transformation
        if parameters["applyTransform"]:
            # Move points
            movedPoints = transformPointsForward_x(tforms[f], movPoints)
    
            # Store in xc and yc if they exist
            if 'xc' in mList and 'yc' in mList:
                if parameters["ignoreFrames"]:
                    mList["xc"] = movedPoints[:,0]
                    mList["yc"] = movedPoints[:,1]
                else:
                    mList["xc"][mList["frame"]==f] = movedPoints[:,0]
                    mList["yc"][mList["frame"]==f] = movedPoints[:,1]
            else:
                print('[Warning]: missingFields - Transformed points were not stored because the xc and yc fields were missing from the provided mList')
    
            residuals[f] = np.concatenate([movedPoints[idx[pointsToKeep],:] - refPoints[pointsToKeep,:], movPoints[idx[pointsToKeep],:]],axis=1)
            # Handle empty residuals
            if len(residuals[f]) == 0:
                residuals[f] = np.zeros((0,4))
        
        # Display progress if in debug mode
        if parameters["debug"]:
            if random() < parameters["displayFraction"]:
                figHandle = plt.figure('FOV '+str(f),figsize=(12,12))
                plt.plot(refPoints[:,0], refPoints[:,1], 'xr',markersize=4,markeredgewidth=0.5)
                plt.plot(movPoints[:,0], movPoints[:,1], 'go',fillstyle='none',markersize=6,markeredgewidth=0.5)

                movedPoints = transformPointsForward_x(tforms[f], movPoints)
                plt.plot(movedPoints[:,0], movedPoints[:,1], 'bs',fillstyle='none',markersize=4,markeredgewidth=0.5)
                plt.savefig(os.path.join(parameters["debugFolder"],"FOV_debug.png"))
                plt.close()

    # -------------------------------------------------------------------------
    # Flatten output if requested
    # -------------------------------------------------------------------------
    if parameters["ignoreFrames"]:
        if len(residuals)>0: # Handle case of no molecules in one of the lists
            residuals = residuals[0]
        tforms = tforms[0]

    return [tforms, mList, residuals, inds, parameters]


