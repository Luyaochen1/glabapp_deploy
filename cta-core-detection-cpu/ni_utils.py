#
# Utility functions for NeuroImaging
# author: Luca Giancardo  
#

import nilearn.image as ni_img
import numpy as np
import pandas as pd
import os
import  skimage.transform as sk_tr 
import skimage as ski

import utils

# constants
DATA_PREFIX_DIR = '/collab/gianca-group/lgiancardo/stroke/data45/stroke_LHC/' # for mask

# Define constants for empty spaces to be removed
xEnd = -36
yStart = 37
yEnd = -34
zStart = 41
zEnd = -56

# Load the Nifti files:
def loadNifti(fileIn):
    return ni_img.load_img(fileIn)

def loadSubjGt(fileIn):
    """
    retrieve valid subjects from Excel
    :param fileIn:
    :return:
    """

    patFr = pd.read_excel(fileIn)
    # Assume all images are usable for now
    # validLbl = patFr['Usable Images?'].str.lower() == 'yes'
    # patFr = patFr[validLbl]

    # patID
    patFr['patID'] = patFr['Study ID'].str.lower()

    # binarize outcome
    patFr['stroke'] = (patFr['Stroke?'].str.lower() == 'yes').astype(int)
    patFr['lvo'] = patFr['LVO']

    # fillin NaN
    # patFr.loc[patFr['Stroke Volume'].isna(), 'Stroke Volume'] = 0
    # patFr.loc[patFr['Penumbra Volume'].isna(), 'Penumbra Volume'] = 0

    return patFr

def loadNpyAndpreProcessVol4(leftBrain, rightBrain, maskIn, extractVessels = True, normNoVess=False, dataType=np.float16):
    """
    preprocess brain hemispheres 
    normNoVess: normaliza data according to normLim. Do not use if extractVessels = True

    v. 4
    """
    #-Constants
    valOutsideMask = 0
    normLim = [0, 100]
    #-
    def normVol(maskTmp):
        # saturate data 
        maskTmp[maskTmp < normLim[0]] = normLim[0]
        maskTmp[maskTmp > normLim[1]] = normLim[1]
        # normalize from 0 to 1 
        maskTmp = maskTmp - normLim[0]
        maskTmp = maskTmp / (normLim[1]-normLim[0])

        return maskTmp
        
    # Extract vessels 
    if extractVessels:
        valOutsideMask = 0.5

        def extractVesselAndNorm(brainIn):
            thTop = 100.
            thVess = 50.
            brainIn = brainIn - thVess
            brainIn[brainIn<0] = 0
            brainIn = np.clip(brainIn, a_min=None, a_max=(thTop-thVess))
            # normalize
            brainIn = brainIn / (thTop-thVess)
            brainIn = brainIn - 0.5

            return brainIn
        leftBrain = extractVesselAndNorm(leftBrain)
        rightBrain = extractVesselAndNorm(rightBrain)

    # Optionally apply mask
    maskedLeft = leftBrain.copy()
    maskedRight = rightBrain.copy()
    if type(maskIn) != bool:
        #==== Compute mask boundary planes
        xStart_mask = 2
        xEnd_mask = 0
        yStart_mask = 0
        yEnd_mask = 0
        zStart_mask = 0
        zEnd_mask = 0

        for x in range(-1, -28, -1):
            if np.sum(maskIn[x, :, :]) > 0:
                xEnd_mask = x
                break
            
        for y in range(73):
            if np.sum(maskIn[:, y, :]) > 0:
                yStart_mask = y
                break
                
        for y in range(-1, -72, -1):
            if np.sum(maskIn[:, y, :]) > 0:
                yEnd_mask = y
                break
            
        for z in range(20):
            if np.sum(maskIn[:, :, z]) > 0:
                zStart_mask = z
                break
                
        zEnd_mask = -1
        #====
        maskedLeft[maskIn == 0] = valOutsideMask
        maskedRight[maskIn == 0] = valOutsideMask

        # normalize data 
        if normNoVess and not extractVessels:            
            maskedLeft = normVol(maskedLeft)
            maskedRight = normVol(maskedRight)

        maskedLeft = maskedLeft[xStart_mask:xEnd_mask, yStart_mask:yEnd_mask, zStart_mask:zEnd_mask]
        maskedRight = maskedRight[xStart_mask:xEnd_mask, yStart_mask:yEnd_mask, zStart_mask:zEnd_mask]
    else:
        # if full brain just normalize 
        if normNoVess and not extractVessels:
            maskedLeft = normVol(maskedLeft)
            maskedRight = normVol(maskedRight)

    # Store sample
    left = np.expand_dims(maskedLeft, axis=[0,-1])
    right = np.expand_dims(maskedRight, axis=[0,-1])
    # left = maskedLeft
    # right = maskedRight

    return [left.astype(dataType), right.astype(dataType)]



if __name__ == "__main__":
    conf = utils.readConf( 'res/configuration.json' )

    # outDir = c['baseDir'] + '/' + c['dirNifti'] + '/' + patID + '/'



    pass