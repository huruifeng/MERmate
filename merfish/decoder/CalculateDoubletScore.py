import os

import numpy as np
import pandas as pd

from utils.funcs import tic, toc, PageBreak

np.seterr(all='ignore')

def CalculateDoubletScore(mDecoder):
    # Calculate Doublet Score
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # September 21, 2017
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2018.
    # -------------------------------------------------------------------------
    # Purpose: 1) Calculate and export various properties of found features
    # that could be used to identify potential segmentation errors
    # -------------------------------------------------------------------------

    PageBreak()
    print('Calculate doublet score...')

    ## Load the found features
    foundFeatures = mDecoder.GetFoundFeatures()

    ## Extract the feature uids
    feature_uID = [fF.uID for fF in foundFeatures]

    ## Load the barcode metadata
    barcodeMetadataPath = os.path.join(mDecoder.normalizedDataPath,'reports','barcode_metadata.csv')
    print('Loading barcode metadata')
    localTimer = tic(0)

    barcodeTable = pd.read_csv(barcodeMetadataPath,index_col=0,header=0)
    print('...found',len(barcodeTable),' barcodes')

    # Cut the barcodes to keep only those within features
    barcodeTable = barcodeTable.loc[barcodeTable.in_feature == 1, :]
    print('...cutting to',len(barcodeTable),' barcodes within features')

    ## Determine properties associated with these barcodes
    numBarcodes = len(mDecoder.codebook["barcode"])

    ## Loop over features and compute numbers and mean positions
    counts = np.zeros((len(feature_uID), numBarcodes))
    comX = np.full((len(feature_uID), numBarcodes),np.nan)
    varX = np.full((len(feature_uID), numBarcodes),np.nan)
    comY = np.full((len(feature_uID), numBarcodes),np.nan)
    varY = np.full((len(feature_uID), numBarcodes),np.nan)

    for f in range(len(feature_uID)):
        # Identify the index of this feature (unnecessary)
        local_feature_id = [fF.feature_id for fF in foundFeatures if fF.uID==feature_uID[f]]

        # Extract the local barcode table
        localBarcodeTable = barcodeTable.loc[barcodeTable.feature_id == local_feature_id[0],:]
        abs_position = np.array([x.strip("[]").split() for x in localBarcodeTable.abs_position.values],dtype=np.float)
        # Compute the counts
        counts[local_feature_id,:] = np.bincount(localBarcodeTable.barcode_id,minlength=numBarcodes)

        # Compute the center of mass for each barcode id
        ids = np.unique(localBarcodeTable.barcode_id)
        comX[local_feature_id,ids] = [np.nanmean(abs_position[np.where(localBarcodeTable.barcode_id == i)[0],0]) for i in ids]
        comY[local_feature_id,ids] = [np.nanmean(abs_position[np.where(localBarcodeTable.barcode_id == i)[0],1]) for i in ids]

        # Compute variance
        varX[local_feature_id,ids] = [np.nanvar(abs_position[np.where(localBarcodeTable.barcode_id == i)[0],0]) for i in ids]
        varY[local_feature_id,ids] = [np.nanvar(abs_position[np.where(localBarcodeTable.barcode_id == i)[0],1]) for i in ids]

        # Display progress
        if not np.mod(f+1, 100):
            print('...completed',f+1,'of',len(foundFeatures))
            print('...in',toc(localTimer),'s')
            localTimer = tic(0)


    ## Create the save path if needed
    savePath = os.path.join(mDecoder.normalizedDataPath,'reports')
    if not os.path.exists(savePath):
        os.makedirs(savePath,exist_ok=True)

    ## Save these data
    np.savetxt(os.path.join(savePath,'barcode_counts.csv'), counts,delimiter=",")
    np.savetxt(os.path.join(savePath,'barcode_center_of_mass_X.csv'), comX,delimiter=",")
    np.savetxt(os.path.join(savePath,'barcode_center_of_mass_Y.csv'), comY,delimiter=",")
    np.savetxt(os.path.join(savePath,'barcode_center_of_mass_var_X.csv'), varX,delimiter=",")
    np.savetxt(os.path.join(savePath,'barcode_center_of_mass_var_Y.csv'), varY,delimiter=",")

