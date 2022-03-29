import uuid

import numpy as np
import pandas as pd
from cv2 import floodFill
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist,pdist
from skimage import segmentation, measure
from skimage.measure import regionprops
from shapely import geometry

from utils.funcs import error
from utils.misc import inpolygon, polygon_area, sub2ind, inpolygon_Matlab

np.seterr(all='ignore')

class FoundFeature:
    # ------------------------------------------------------------------------
    # [fFeature, parameters] = FoundFeature(..., varargin)
    # This class is a container for segmented features in MERFISH data.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # jeffrey.moffitt@childrens.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    #--------------------------------------------------------------------------
    # This class is a wrapper around a found feature, i.e. a segmented cell, and
    # contains all the basic functionality associated with features

    version = '1.0'

    def __init__(self, **kwargs):
        # -------------------------------------------------------------------------
        # Define properties
        # -------------------------------------------------------------------------
        self.name = kwargs["name"] if ("name" in kwargs) and kwargs["name"] else ""    # A string name for this feature
        self.type = kwargs["type"] if ("type" in kwargs) and kwargs["type"] else ""                   # A string that can be used to distinguish different feature types
        self.verbose = kwargs["verbose"] if ("verbose" in kwargs) and kwargs["verbose"] else False             # A boolean that determines whether or not the class displays progress

        # Properties associated with the image
        self.image_size = kwargs["image_size"] if ("image_size" in kwargs) and kwargs["image_size"] else (0,0)     # The size of the original image (WxH)
        self.pixel_size = kwargs["pixel_size"] if ("pixel_size" in kwargs) and kwargs["pixel_size"] else 0     # The size of the pixel

        # Properties to allow identification of feature
        self.uID = kwargs["uID"] if ("uID" in kwargs) and kwargs["uID"] else ""             # A unique ID to this feature
        self.joinedUIDs =kwargs["joinedUIDs"] if ("joinedUIDs" in kwargs) and kwargs["joinedUIDs"] else []      # A cell array of all unique IDs joined to create this cell (if any)
        self.fovID =kwargs["fovID"] if ("fovID" in kwargs) and kwargs["fovID"] else []           # The ID of fov in which the cell appears
        self.feature_label = kwargs["feature_label"] if ("feature_label" in kwargs) and kwargs["feature_label"] else ""   # The element in the label matrix corresponding to this feature

        # Properties of the boundaries in fov coordinates
        self.num_zPos = kwargs["num_zPos"] if ("num_zPos" in kwargs) and kwargs["num_zPos"] else 0       # The number of z slices
        self.boundaries =kwargs["boundaries"] if ("boundaries" in kwargs) and kwargs["boundaries"] else ""      # A cell array of boundaries: A multidimensional cell if multiple fov have been grouped together

        # Properties of the boundaries in real coordinates
        self.abs_zPos = kwargs["abs_zPos"] if ("abs_zPos" in kwargs) and kwargs["abs_zPos"] else []          # An array of z positions
        self.abs_boundaries =kwargs["abs_boundaries"] if ("abs_boundaries" in kwargs) and kwargs["abs_boundaries"] else []  # A cell array of boundaries

        # Meta data associated with the features
        self.volume = kwargs["volume"] if ("volume" in kwargs) and kwargs["volume"] else 0             # The number of voxels in the feature
        self.abs_volume = kwargs["abs_volume"] if ("abs_volume" in kwargs) and kwargs["abs_volume"] else 0          # The total sample volume in the feature
        self.boundary_area  =kwargs["boundary_area"] if ("boundary_area" in kwargs) and kwargs["boundary_area"] else []         # An array of the number of pixels in each z stack
        self.abs_boundary_area =kwargs["abs_boundary_area"] if ("abs_boundary_area" in kwargs) and kwargs["abs_boundary_area"] else []      # An array of the area in each z stack
        self.is_broken = kwargs["is_broken"] if ("is_broken" in kwargs) and kwargs["is_broken"] else False       # Is the boundary broken?
        self.num_joined_features = kwargs["num_joined_features"] if ("num_joined_features" in kwargs) and kwargs["num_joined_features"] else 0 # The number of features joined together

        # Useful features for quickly indexing the feature
        self.feature_id = kwargs["feature_id"] if ("feature_id" in kwargs) and kwargs["feature_id"] else -1

        # Links to features contained or associated with this feature
        self.children =kwargs["children"] if ("children" in kwargs) and kwargs["children"] else []       # References to features that this feature 'owns'
        self.parents = kwargs["parents"] if ("parents" in kwargs) and kwargs["parents"] else []         # References to features that own this feature

        # Metadata associated with the cell
        self.metaData = kwargs["metaData"] if ("metaData" in kwargs) and kwargs["metaData"] else {}       # Reserved for misc meta data associated with the feature

    def createFoundFeature(self,labelMat, fovID, fovCenterPos, pixelSize, stageOrientation, boundingBox, zPos, featureLabel, **kwargs):
        # This class allows the construction of a defined feature based on
        # the pixels that are true within a 2D or 3D label matrix image
        # foundFeature = FoundFeature(labelMat, fovID, fovCenterPos,
        # pixelSize, stageOrientation, boundingBox, zPos, featureLabel)

        # -------------------------------------------------------------------------
        # Handle optional arguments
        # -------------------------------------------------------------------------
        # Define defaults
        parameters = {}

        # Parameters for parsing file names
        parameters['verbose'] = False      # Display progress of construction
        parameters['name'] =  ''          # A name for the OTTable instance

        # Parse varaible arguments with defaults
        for k_i in kwargs:
            parameters[k_i] = kwargs[k_i]

        # Transfer to object
        for k_i in parameters:
            if k_i in self.__dict__:
                self.__dict__[k_i] = parameters[k_i]



        # -------------------------------------------------------------------------
        # Transfer common information
        # -------------------------------------------------------------------------
        self.fovID = [fovID] if not isinstance(fovID,(list,np.ndarray)) else fovID
        self.abs_zPos = zPos
        self.feature_label = featureLabel
        self.image_size = labelMat.shape
        self.image_size = self.image_size[1:3]
        self.pixel_size = pixelSize
        self.num_zPos = labelMat.shape[0]

        # -------------------------------------------------------------------------
        # Prepare the object
        # -------------------------------------------------------------------------
        # Assign a uID to this found feature
        self.uID =  str(uuid.uuid4())

        # Prepare boundaries
        self.boundaries = np.empty((2,self.num_zPos),dtype=np.ndarray) # Two sets of boundaries to allow for joined features
        self.abs_boundaries =  np.empty((self.num_zPos,),dtype=np.ndarray)

        # -------------------------------------------------------------------------
        # Parse label matrix and build unique features
        # -------------------------------------------------------------------------
        # Loop through each z stack
        for z in range(self.num_zPos):

            # Find boundaries without holes
            # boundaries = np.argwhere(segmentation.find_boundaries(labelMat[z],mode="inner"))
            boundaries = measure.find_contours(labelMat[z], 0.9, fully_connected='high')

            # Determine if this boundary is empty
            if len(boundaries)==0:
                continue

            # Concatenate boundaries for multiple objects (i.e. connected in a different z plane)
            lBoundary = boundaries[0]
            for l in range(1,len(boundaries)):
                # lBoundary = np.concatenate((lBoundary,[[np.nan,np.nan]], boundaries[l]),0)
                lBoundary = np.concatenate((lBoundary, boundaries[l]), 0)


            # Flip X/Y coordinates to match with barcodes
            lBoundary = lBoundary[:, [1, 0]]

            # Add boundary to boundary list
            self.boundaries[0,z] = lBoundary

            # Convert boundary to absolute scale
            # Transform coordinate system to middle of image
            abs_lBoundary = lBoundary - np.tile([labelMat.shape[1]/2,labelMat.shape[2]/2], (lBoundary.shape[0],1))
            # Convert pixels to microns
            abs_lBoundary = abs_lBoundary * np.tile(pixelSize/1000*np.array(stageOrientation), (abs_lBoundary.shape[0],1))

            # Apply crop (bounding box is in microns)
            inInds = np.ones((abs_lBoundary.shape[0],))
            inInds[~ np.isnan(abs_lBoundary[:,0])] = (abs_lBoundary[~np.isnan(abs_lBoundary[:,0]),0] >= boundingBox[0])& \
                                                     (abs_lBoundary[~np.isnan(abs_lBoundary[:,0]),0] <= (boundingBox[0] + boundingBox[2])) & \
                                                     (abs_lBoundary[~np.isnan(abs_lBoundary[:,0]),1] >= boundingBox[1]) & \
                                                     (abs_lBoundary[~np.isnan(abs_lBoundary[:,0]),1] <= (boundingBox[1] + boundingBox[3]))
            inInds = inInds.astype(bool)
            # Check to see if the only remaining true in ind is due to a
            # nan flag
            if np.sum(inInds) == np.sum(np.isnan(abs_lBoundary[:,1])):
                inInds = np.zeros((abs_lBoundary.shape[0],),dtype=bool)

            # Determine start and stop of line segments
            startInds = np.nonzero(inInds & ~np.roll(inInds,-1,axis=0))[0]+1 # The index of the start of the break
            numBreaks = len(startInds)

            # Handle case that the boundary has no breaks
            if len(startInds)==0:
                startInds = [0]

            # Shift the coordinate system to the location of the fov
            abs_lBoundary = abs_lBoundary + np.tile(fovCenterPos, (abs_lBoundary.shape[0],1))

            # Skip construction of boundaries if nothing survived the crop
            if np.all(~inInds) or numBreaks > 1: # Discard anything that has been cropped away
                self.is_broken = True
                continue

            # Shift the absolute boundaries and index out the remaining
            # boundaries
            abs_lBoundary = np.roll(abs_lBoundary, -startInds[0]+1, axis=0)
            abs_lBoundary = abs_lBoundary[np.sum(~inInds):,:]

            # Add absolute/cropped boundary to boundary list
            self.abs_boundaries[z] = abs_lBoundary

            # Mark the feature as broken or not
            self.is_broken = (self.is_broken) | (numBreaks > 0)

        # -------------------------------------------------------------------------
        # Calculate the metadata associated with the feature
        # -------------------------------------------------------------------------
        self.CalculateProperties()


    # -------------------------------------------------------------------------
    # Calculate Feature Centroid
    # -------------------------------------------------------------------------
    def CalculateCentroid(self):
        # This method calculates the centroid of the feature boundary in
        # absolute coordinates
        #
        # centroid = self.CalculateCentroid(obj)

        # Calculate the x,y centroid for each boundary
        centroid = np.zeros((3,))
        muCentroid = np.zeros((2, self.num_zPos))
        for z in range(self.num_zPos):
            lBoundary = self.abs_boundaries[z]
            muCentroid[:,z] = np.nanmean(lBoundary,0)

        # Weight the x/y centroids by relative boundary area
        centroid[:2] = np.nansum(muCentroid*np.tile(self.abs_boundary_area, (2,1)),1)/np.nansum(self.abs_boundary_area)

        # Weight the z centroid by the area of each boundary
        centroid[2] = np.nansum(self.abs_zPos*self.abs_boundary_area)/np.nansum(self.abs_boundary_area)

        return centroid

    # -------------------------------------------------------------------------
    # Plot found feature in absolute coordinates on provide axis handle
    # -------------------------------------------------------------------------
    def Plot(self, zInd=None, axisHandle=None):
        # Plot the found feature in the provided zInd into the provided
        # axes handle
        #
        # lineHandles = self.Plot(zInd, axisHandle)

        # Create a figure and axis if not provided
        if axisHandle == None:
            figHandle = plt.figure()

        # Define the z ind if not provided
        if zInd== None:
            zInd = 1

        # If multiple objects are provided (i.e. this method called from a
        # class array), then loop over then and plot
        lineHandles = []
        for i in range(len(self)):
            localBoundaries = self.abs_boundaries[zInd]
            if localBoundaries.size > 0:
                lineHandles.append(plt.plot(localBoundaries[:,0],localBoundaries[:,1]))
        return lineHandles


    # -------------------------------------------------------------------------
    # Calculate morphological properties of the feature
    # -------------------------------------------------------------------------
    def CalculateMorphology(self):
        # This method calculates morphological properties of the feature
        #
        # [eccentricity, hullRatio, numRegions] = self.CalculateMorphology(obj)

        # Allocate memory
        eccentricity = np.zeros((self.num_zPos,))
        hullRatio = np.zeros((self.num_zPos,))
        numRegions = np.zeros((self.num_zPos,))
        edges = [[],[]]
        # Loop over all z plane
        for z in range(self.num_zPos):
            # Extract boundary
            lBoundary = self.abs_boundaries[z]

            # Break if is empty
            if lBoundary.size == 0:
                continue

            # Convert to pixel values
            lBoundary = np.round(lBoundary/(self.pixel_size/1000))

            # Remove nan
            isBadIndex = np.isnan(lBoundary[:,0])
            lBoundary = lBoundary[~isBadIndex,:]

            # Calculate the values to convert to a binary image
            edges[0] = np.unique(lBoundary[:,0])
            edges[1] = np.unique(lBoundary[:,1])

            # Confirm a valid boundary
            numUniquePixels = [len(e_i) for e_i in edges]
            if np.any(numUniquePixels <= 1):
                continue

            # Compute the 2D image of boundary
            image = np.histogram2d(lBoundary, bins=edges)>0

            # Fill the holes in the image
            image = ndimage.binary_fill_holes(image)

            # Compute the properties
            props = regionprops(image)

            # Keep the eccentricity of the largest region
            eccentricity[z] = props[0].eccentricity

            # Keep the ratio of the area to the convex area for the largest
            # region
            hullRatio[z] = props[0].area/props[0].convex_area

            # Record the number of regions
            numRegions[z] = len(props)
        return  eccentricity, hullRatio, numRegions

    # -------------------------------------------------------------------------
    # Calculate the distance to the feature
    # -------------------------------------------------------------------------
    def DistanceToFeature(self, position):
        # This method calculates the distance between a position and the
        # boundaries of the local feature (in absolute coordinates)
        #
        # dist = self.DistanceToFeature(position)

        # Check necessary input
        if  position.shape!= (1,3):
            error('[Error]:invalidArguments - A 1 x 3 position vector must be provided.')

        # Determine the proper z index
        zInd = (position[0,2] >= [self.abs_zPos]) & (position[0,2] < [self.abs_zPos[1:],np.inf])

        # Calculate distance to boundary
        dist = np.min(np.sqrt(np.sum((self.abs_boundaries[zInd] - np.tile(position[0,0:2],(self.abs_boundaries[zInd].shape[0],1)))**2 )))

        return dist

    # -------------------------------------------------------------------------
    # Determine if a point is within the feature
    # -------------------------------------------------------------------------
    def IsInFeature(self, position):
        # This method returns a boolean specifying if the point or position
        # is within the feature.
        #
        # inFeature = self.IsInFeature(position)

        # Check necessary input
        if position.shape != (1, 3):
            error('[Error]:invalidArguments - A 1 x 3 position vector must be provided.')

        # Determine the proper z index
        zInd = position[0,2] >= [self.abs_zPos] & position[0,2] < [self.abs_zPos[1:] ,np.inf]

        # Calculate distance to boundary
        inFeature = inpolygon(position[0,0], position[0,1],self.abs_boundaries[zInd][:,1], self.abs_boundaries[zInd][:,1])

        return inFeature
    # -------------------------------------------------------------------------
    # Assign feature id
    # -------------------------------------------------------------------------
    def AssignFeatureID(self, featureID):
        # Assign a feature id to the object, must be a positive integer
        #
        # self.AssignFeatureID(featureID)

        # Check required input
        if featureID < 0 or (np.round(featureID) != featureID):
            error('[Error]:invalidArguments - A positive integer feature ID must be provided.')

        # Assign the feature id
        self.feature_id = featureID


    # -------------------------------------------------------------------------
    # Check overlap between this feature and another feature
    # -------------------------------------------------------------------------
    def DoesFeatureOverlap(self, fFeature):
        # This method checks to see if any portion of the provided fFeature
        # overlaps with any portion of the current feature. This comparison
        # is done using boundaries in the absolute (real world) coordinates
        #
        # doesOverlap = self.DoesFeatureOverlap(fFeature)

        # Check to confirm that the provided fFeature is a a foundFeature
        if not isinstance(fFeature, FoundFeature):
            error('[Error]:invalidArgument - A valid foundFeature object must be provided')

        # Check to confirm that the zPositions are equivalent between these
        # two features
        if not np.all(self.abs_zPos == fFeature.abs_zPos):
            print('[Warning]:differentCoordinates - The two FoundFeature objects do not share the same z coordinate system.')

        # Initialize doesOverlap
        doesOverlap = False

        # Loop through the z positions
        for z in range(len(self.abs_zPos)):
            # Identify corresponding z postion in the other feature
            zInd = np.nonzero(fFeature.abs_zPos == self.abs_zPos[z])[0]

            # Handle the case that this position does not exist in both
            # features
            if len(zInd) == 0:
                continue

            # Extract boundaries for these z planes
            absBoundary1 = self.abs_boundaries[z]
            absBoundary2 = fFeature.abs_boundaries[zInd[0]]

            # Handle the position that either boundary is empty
            if len(absBoundary1)==0 or len(absBoundary2)==0:
                continue

            # Determine the points that are within or overlapping
            in_on_x, on_x = inpolygon_Matlab(absBoundary2[:,0], absBoundary2[:,1], absBoundary1[:,0], absBoundary1[:,1])

            # Find if any points are in (but not shared between the boundaries)
            doesOverlap = np.any(in_on_x[~on_x])

            # Exit as soon as any overlap is found
            if doesOverlap:
                return doesOverlap

        return doesOverlap

        # -------------------------------------------------------------------------
        # Check overlap between this feature and another feature
        # -------------------------------------------------------------------------

    def doesContainsFeature(self, fFeature):
        # This method checks to see if the provided fFeature is in current feature.
        #
        # isInterior = self.isInteriorFeature(fFeature)

        # Check to confirm that the provided fFeature is a a foundFeature
        if not isinstance(fFeature, FoundFeature):
            error('[Error]:invalidArgument - A valid foundFeature object must be provided')

        # Check to confirm that the zPositions are equivalent between these
        # two features
        if not np.all(self.abs_zPos == fFeature.abs_zPos):
            print(
                '[Warning]:differentCoordinates - The two FoundFeature objects do not share the same z coordinate system.')

        # Initialize doesOverlap
        deosContains = False

        # Loop through the z positions
        for z in range(len(self.abs_zPos)):
            # Identify corresponding z postion in the other feature
            zInd = np.nonzero(fFeature.abs_zPos == self.abs_zPos[z])[0]

            # Handle the case that this position does not exist in both
            # features
            if len(zInd) == 0:
                continue

            # Extract boundaries for these z planes
            absBoundary1 = self.abs_boundaries[z]
            absBoundary2 = fFeature.abs_boundaries[zInd[0]]

            # Handle the position that either boundary is empty
            if len(absBoundary1) == 0 or len(absBoundary2) == 0:
                continue
            absBoundary1 = geometry.Polygon(absBoundary1)
            absBoundary2 = geometry.Polygon(absBoundary2)
            deosContains =absBoundary1.contains(absBoundary2)

            # Exit as soon as any overlap is found
            if deosContains:
                return deosContains

        return deosContains

    def isValidFeature(self):

        # Initialize
        isValid = True

        # Loop through the z positions
        for z in range(len(self.abs_zPos)):
            # Extract boundaries for these z planes
            absBoundary = self.abs_boundaries[z]

            # Handle the position that either boundary is empty
            if len(absBoundary) == 0:
                continue
            absBoundary = geometry.Polygon(absBoundary)

            isValid = absBoundary.is_valid

            # Exit as soon as any overlap is found
            if not isValid:
                return isValid

        return isValid

    # -------------------------------------------------------------------------
    # Calculate area/volume properties of the feature
    # -------------------------------------------------------------------------
    def CalculateProperties(self):
        # Calculate the area, the volume, and the bounding boxes for both
        # pixel and absolute coordinate values
        #
        # self.CalculateProperties()

        # Prepare the object
        self.boundary_area = np.zeros((self.num_zPos,))
        self.abs_boundary_area = np.zeros((self.num_zPos,))

        # Determine the thickness of the optical slice (if appropriate)
        if self.num_zPos > 1:
            zSliceThickness = np.mean(np.diff(self.abs_zPos))
        else:
            zSliceThickness = 1

        # Loop over all z positions
        for z in range(self.num_zPos):
            # Loop over all fov associated with each pixel coordinate
            # system boundary
            for f in range(len(self.fovID)):
                # Extract pixel coordinate boundary
                lBoundary = self.boundaries[f,z]

                # Handle case of empty boundary
                if not isinstance(lBoundary,np.ndarray) or len(lBoundary)==0:
                    continue

                # Compute metadata on this boundary (in pixel coordinates)
                self.boundary_area[z] = self.boundary_area[z] + polygon_area(lBoundary[~np.isnan(lBoundary[:,0]),0], lBoundary[~np.isnan(lBoundary[:,0]),1])
                self.volume = self.volume + self.boundary_area[z]

            # Extract absolute coordinate boundary
            abs_lBoundary = self.abs_boundaries[z]

            # Handle case of empty boundary
            if not isinstance(abs_lBoundary,np.ndarray) or len(abs_lBoundary) == 0:
                continue

            # Determine meta data for absolute boundary
            self.abs_boundary_area[z] = polygon_area(abs_lBoundary[~np.isnan(abs_lBoundary[:,0]),0], abs_lBoundary[~np.isnan(abs_lBoundary[:,0]),1])
            self.abs_volume = self.abs_volume + self.abs_boundary_area[z]*zSliceThickness

    # -------------------------------------------------------------------------
    # Implement/overload a copy command to make a deep copy of the object
    # -------------------------------------------------------------------------
    def copy(self):
        # Make a deep copy of the object (ignoring handles)

        # Make a new found feature object
        cpObj = FoundFeature()

        # Find all properties associated with the old object
        fields = list(self.__dict__.keys())

        # Remove the uID
        fields = np.setdiff1d(fields, 'uID')

        # Transfer fields
        for f in fields:
            cpObj.__dict__[f] = self.__dict__[f]

        # Create new uID
        cpObj.uID = str(uuid.uuid4())
        return cpObj

    # -------------------------------------------------------------------------
    # Determine the penalty associated with joining two found features
    # -------------------------------------------------------------------------
    def CalculateJoinPenalty(self, featureToJoin=None):
        # Calculate the penalty associated with joining two feature objects
        # or of joining an object with itself. The penalty is defined as
        # the average distance between the broken ends of all of the
        # corresponding boundaries

        # Examine required input
        if featureToJoin==None:
            joiningSelf = True
        elif not isinstance(featureToJoin, FoundFeature):
            error('[Error]:invalidArguments - An instance of FoundFeature must be provided')
        else:
            joiningSelf = False

        # Examine the features to see if they can be joined
        if not joiningSelf:
            # Confirm that the features are broken
            if  not (self.is_broken and featureToJoin.is_broken):
                error('[Error]:invalidArguments - Found Features must have a broken boundary to be capable of being joined')

            # Check that the two objects have the same z coordinate system
            if not np.all(self.abs_zPos == featureToJoin.abs_zPos):
                error('[Error]:invalidArguments - To be joined, the two FoundFeature objects must have the same z coordinate system')
        else:
            if not self.is_broken:
                error('[Error]:invalidArguments', 'Found Features must have a broken boundary to be capable of being joined')

        # Handle the case that the penalty is being calculated for the
        # object itself
        if joiningSelf:
            penalty = 0
            numZSlices = 0
            for z in range(self.num_zPos):
                # Extract absolute coordinate system boundaries
                abs_boundary = self.abs_boundaries[z]

                # Handle empty boundaries for this slice
                if len(abs_boundary)==0:
                    continue

                # Compute start to end distance
                DES = pdist([abs_boundary[-1,:], abs_boundary[0,:]])[0]

                # Accumulate penalty
                penalty = DES + penalty

                # Increment the number of z slices
                numZSlices = numZSlices + 1

            # Normalize by the number of z slices
            penalty = penalty/numZSlices

            # Return the penalty value
            return penalty

        # Continue with the case that there are two features to compare
        penalty = 0
        numZSlices = 0
        for z in range(self.num_zPos):

            # Extract boundaries
            boundary1 = self.abs_boundaries[z]
            boundary2 = featureToJoin.abs_boundaries[z]

            # If either is empty, skip this feature
            if boundary1.size==0 or boundary2.size==0:
                continue

            # Compute the distance
            DES = pdist([boundary1[-1,:], boundary2[0,:]])[0] + pdist([boundary1[0,:], boundary2[-1,:]])[0]
            DEE = pdist([boundary1[-1,:], boundary2[-1,:]])[0] + pdist([boundary1[0,:], boundary2[0,:]])[0]

            # Determine the minimum of the two
            minD = np.min([DES,DEE])

            # Accumulate penalty
            penalty = penalty + minD

            # Increment number of z slices
            numZSlices = numZSlices + 1

        # Normalize the penalty
        penalty = penalty/numZSlices

        # Handle the case that neither boundary set existed in the same z
        # plane
        if numZSlices == 0:
            penalty = np.inf

        return penalty

    # -------------------------------------------------------------------------
    # Join two FoundFeature objects
    # -------------------------------------------------------------------------
    def JoinFeature(self, featureToJoin=None):
        # Join two found feature objects (or one found feature object
        # with itself
        #
        # joinedFeature = self.JoinFeature() # Join the feature with itself
        # joinedFeature = self.JoinFeature(featureToJoin) # Join the
        # feature with another feature

        # Examine required input
        if featureToJoin == None:
            joiningSelf = True
        elif not isinstance(featureToJoin, FoundFeature):
            error('[Error]:invalidArguments - An instance of FoundFeature must be provided')
        else:
            joiningSelf = False

        # Examine the features to see if they can be joined
        if not joiningSelf:
            if not (self.is_broken and featureToJoin.is_broken):
                error('[Error]:invalidArguments - Found Features must have a broken boundary to be capable of being joined')

            # Check that the two objects have the same z coordinate system
            if not np.all(self.abs_zPos == featureToJoin.abs_zPos):
                error('[Error]:invalidArguments - To be joined, the two FoundFeature objects must have the same z coordinate system')
        else:
            if not self.is_broken:
                error('[Error]:invalidArguments - Found Features must have a broken boundary to be capable of being joined')

        # Make copy of the current object
        joinedFeature = self.copy()

        # Handle the case that we are joining this feature
        if joiningSelf:
            # Transfer the uID of the previous feature
            joinedFeature.joinedUIDs.append(self.uID)

            # Loop over z positions
            for z in range(self.num_zPos):

                # Extract boundary
                boundary1 = joinedFeature.abs_boundaries[z]

                # Escape if the boundary is empty
                if boundary1.size==0:
                    continue

                # Fill edges
                endDist = np.sqrt(np.sum((boundary1[-1,:] - boundary1[0,:])**2))
                numPoints = int(np.round(endDist/(self.pixel_size/1000)))
                boundary1ToBoundary1 = np.stack((np.linspace(boundary1[-1,0], boundary1[0,0], numPoints+2), # X positions
                                                np.linspace(boundary1[-1,1], boundary1[0,1], numPoints+2)),1)    # Y positions
                boundary1ToBoundary1 = boundary1ToBoundary1[1:-1,:] # Trim off replicate points

                joinedFeature.abs_boundaries[z] = np.concatenate((boundary1, boundary1ToBoundary1),0)

        else: # Handle the case that we are joining two features

            # Transfer the uID of the previous feature
            joinedFeature.joinedUIDs.append(self.uID)
            joinedFeature.joinedUIDs.append(featureToJoin.uID)

            # Remove pixel boundaries and fov information from joined
            # feature
            joinedFeature.boundaries = np.empty((2,self.num_zPos),dtype=np.ndarray)
            joinedFeature.fovID = [self.fovID,featureToJoin.fovID]

            # Loop over z positions
            for z in range(self.num_zPos):
                # Extract boundaries
                boundary1 = self.abs_boundaries[z]
                boundary2 = featureToJoin.abs_boundaries[z]

                # Exit if either is empty
                if boundary1.size==0 or boundary2.size==0:
                    continue

                # Compute the distance
                DES = pdist([boundary1[-1,:], boundary2[0,:]])[0] + pdist([boundary1[0,:], boundary2[-1,:]])[0]
                DEE = pdist([boundary1[-1,:], boundary2[-1,:]])[0] + pdist([boundary1[0,:], boundary2[0,:]])[0]

                # If the end better matches to the end, then one boundary
                # needs to be inverted
                if DEE < DES:
                    boundary2 = np.flipud(boundary2)

                # Fill gaps: boundary 2 to boundary 1
                endDist = np.sqrt(np.sum((boundary2[-1,:] - boundary1[0,:])**2))
                numPoints = int(np.round(endDist/(self.pixel_size/1000)))
                boundary2ToBoundary1 = np.stack(
                    (np.linspace(boundary2[-1,0], boundary1[0,0], numPoints+2),  # X positions
                    np.linspace(boundary2[-1,1], boundary1[0,1], numPoints+2)),1) # Y positions
                boundary2ToBoundary1 = boundary2ToBoundary1[1:-1,:] # Trim off replicate points

                # Fill gaps: boundary 1 to boundary 2
                endDist = np.sqrt(np.sum( (boundary1[-1,:] - boundary2[0,:])**2))
                numPoints = int(round(endDist/(self.pixel_size/1000)))
                boundary1ToBoundary2 = np.stack(
                    (np.linspace(boundary1[-1,0], boundary2[0,0], numPoints+2),  # X positions
                    np.linspace(boundary1[-1,1], boundary2[0,1], numPoints+2)),1)    # Y positions
                boundary1ToBoundary2 = boundary1ToBoundary2[1:-1,:] # Trim off replicate points

                # Update the absolute boundary in the joined object
                joinedFeature.abs_boundaries[z] = np.concatenate((boundary1, boundary1ToBoundary2,
                                                                  boundary2, boundary2ToBoundary1),0)

                # Update the pixel coordinate boundaries
                joinedFeature.boundaries[0,z] = self.boundaries[0,z]
                joinedFeature.boundaries[1,z] = featureToJoin.boundaries[0,z]

            # Update the meta data associated with the joined feature
            joinedFeature.CalculateProperties()

        # Update the number of joined features
        joinedFeature.num_joined_features = len(joinedFeature.fovID)
        joinedFeature.is_broken = False

        return joinedFeature

    # -------------------------------------------------------------------------
    # Determine if this feature falls within a fov
    # -------------------------------------------------------------------------
    def InFov(self, fovIDs=[]):
        # Determine if the feature falls within any of the specified fovIDs
        #
        # isInFov = self.InFov(fovIDs)

        # Check required input
        if not isinstance(fovIDs,(np.ndarray,list)) or len(fovIDs)==0:
            error('[Error]:invalidArguments - A set of fov IDs must be provided')

        # Determine if the fovIDs for this feature are within the specified
        # fovIDs
        isInFov = np.any(np.isin(self.fovID, fovIDs))
        return isInFov

# -------------------------------------------------------------------------
    # Dilate an absolute boundary
    # -------------------------------------------------------------------------
    def DilateBoundary(self, zIndex, dilationSize):
        # Produce a dilated boundary useful for determining if an RNA is
        # within a cell or not
        #
        # boundary = self.DilateBoundary(zIndex, dilationSize)

        # Extract the absolute boundary
        oBoundary = self.abs_boundaries[zIndex]

        # Compute the arc length
        s = np.sqrt(np.sum((oBoundary - np.roll(oBoundary,-1,axis=0))**2,1))

        # Compute the derivatives
        dx = np.gradient(oBoundary[:,0])/s
        ddx = np.gradient(dx)/s
        dy = np.gradient(oBoundary[:,1])/s
        ddy = np.gradient(dy)/s

        # Compute the curvature
        num = dx * ddy - ddx * dy
        denom = dx * dx + dy * dy
        denom = np.sqrt(denom)
        denom = denom * denom * denom
        kappa = num / denom
        kappa[denom < 0] = np.nan

        # Handle the case of differential directions to the curves
        signValues = [1,-1]

        # Use the area to determine if we have selected the correct sign
        area = polygon_area(oBoundary[:,0], oBoundary[:,1])

        # Loop over sign values
        for i in range(len(signValues)):
            # Create the normal vector
            normal = np.stack((ddx, ddy),1)
            normal = normal/np.tile(np.sqrt(np.sum(normal**2, 1)).reshape(-1,1), (1,2))
            normal = normal*np.tile(np.sign(kappa).reshape(-1,1), (1,2)) # Position it correctly

            # Handle when the vector is not defined
            noVectorInds = np.all(normal==0,1)
            normal[noVectorInds,:] = np.nan

            # Fill in all nan values with linear interpolation
            normal =  pd.DataFrame(normal).interpolate(limit_direction="both").to_numpy()

            # Provide a slight filter
            normal = 1/3*(np.roll(normal,-1,axis=0) + normal + np.roll(normal,1,axis=0))

            # Renormalize
            normal = normal/np.tile(np.sqrt(np.sum(normal**2, 1)).reshape(-1,1), (1,2))

            # Provide the new boundary
            dBoundary = oBoundary + dilationSize*signValues[i]*normal

            # Calculate new area
            newArea = polygon_area(dBoundary[:,0], dBoundary[:,1])

            # If it is larger the sign is correct, break
            if newArea > area:
                break

        return dBoundary

    # -------------------------------------------------------------------------
    # Generate a pixel mask for the specified z indices
    # -------------------------------------------------------------------------
    def GeneratePixelMask(self, fovID, zIndices):
        # Produce a binary mask (2D or 3D) for the boundaries in the given
        # fov and with the specified zIndices
        #
        # mask = self.GeneratePixelMask(fovID, zIndices)

        # Confirm that the provided values are valid
        if not np.isin(fovID, self.fovID):
            error('[Error]:invalidArgument - The requested fov does not contain the desired feature')
        if not np.all(np.isin(zIndices, np.arange(0,self.num_zPos))):
            error('[Error]:invalidArgument - The requested z indices are not present in this feature')

        # Allocate mask memory
        mask = np.zeros((self.image_size[0], self.image_size[1], len(zIndices)))

        # Determine the local fovID
        localFovID = np.nonzero(self.fovID == fovID)[0]

        # Build this mask for the specified z indices
        for z in range(len(zIndices)):
            # Build the outline mask
            localMask = np.zeros((self.image_size))

            # Extract boundary
            lBoundary = self.boundaries[localFovID, zIndices[z]]
            lBoundary = np.stack(lBoundary[0],0)
            # Remove nan values
            lBoundary = lBoundary[~np.isnan(lBoundary[:,0]),:]
            lBoundary = np.round(lBoundary).astype(np.int)
            # Handle empty case
            if lBoundary.size==0:
                continue

            # Add boundaries to mask
            localMask.flat[sub2ind(self.image_size, lBoundary[:,1], lBoundary[:,0],order="F")] = True
            localMask = localMask.T
            # Build the full mask by filling in the holes in this image
            mask[:,:,z] = ndimage.binary_fill_holes(localMask)
        return mask

    # -------------------------------------------------------------------------
    # Export the found feature to a table object
    # -------------------------------------------------------------------------
    def Feature2Table(self,fObj):
    # Output all properties of a the feature as a table. This output functionality will eventually allow features to be saved/read from a csv format.
    # oTable = self.Feature2Table()

        # Determine the number of zPos
        numZPos = len(fObj[1].abs_zPos)

        # Create the cell arrays for these boundaries
        boundariesX = np.tile('', (len(fObj), numZPos))
        boundariesY = np.tile('', (len(fObj), numZPos))

        # Fill these items
        uIDs = self.uID
        featureIDs = np.concatenate(self.feature_id,0)
        isBrokenFlags = np.concatenate(self.is_broken,0)
        numJoinedFeatures = np.concatenate(self.num_joined_features,0)
        absVolumes = np.concatenate(self.abs_volume,0)

        fovIDs = self.fovID[0]

        # Fill the boundary cells
        for z in range(numZPos):
            for o in range(len(fObj)):
                localBoundary = fObj[o].abs_boundaries[z]
                boundaryX = np.array((1, 2*localBoundary.shape[0]))
                # boundaryX(1:2:end) = arrayfun(@num2str, localBoundary(:,1), 'UniformOutput', false)
                boundaryX[0:2:] = [str(i) for i in localBoundary[:,0]]
                boundaryX[0:2:] = np.tile('', (1,localBoundary.shape[0]))
                boundariesX[o,z] = np.concatenate(boundaryX,1)
                boundaryY = np.array((1, 2*localBoundary.shape[0]))
                # boundaryY(1:2:end) = arrayfun(@num2str, localBoundary(:,2), 'UniformOutput', false)
                boundaryY[0:2:] = [str(i) for i in localBoundary[:,1]]
                boundaryY[1:2:] =  np.tile('', (1,localBoundary.shape[0]))
                boundariesY[o,z] = np.concatenate(boundaryY,1)

        # Create the table
        oTable =pd.DataFrame()

        # Add metadata
        oTable.loc[:,"feature_uID"] = uIDs
        oTable.loc[:,"feature_ID"] = featureIDs
        oTable.loc[:,"fovID"] = fovIDs
        oTable.loc[:,"is_broken"] = isBrokenFlags
        oTable.loc[:,"num_joined_features"] = numJoinedFeatures
        oTable.loc[:,"abs_volume"] = absVolumes

        # Add boundaries
        for z in range(numZPos):
            oTable.loc[:,'abs_x_boundary_'+str(z)] = boundariesX[:,z]
            oTable.loc[:,'abs_y_boundary_'+str(z)] = boundariesY[:,z]

        return oTable




