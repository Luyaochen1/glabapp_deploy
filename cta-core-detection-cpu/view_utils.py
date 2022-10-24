#
# Utility functions to view neuroimaging volumes
# author: Luca Giancardo  
#

import numpy as np
import random
import pandas as pd
import pylab as pl
import nilearn.image as ni_img
import nilearn.datasets as ni_data
import nilearn.plotting as ni_pl
import matplotlib.pyplot as plt
import nibabel as nib


def joinBrain(leftArr, rightArr):
    fullBrain = np.vstack((np.flip(rightArr, axis=0), leftArr))
    fullBrainNf = nib.Nifti1Image(fullBrain, affine=np.eye(4))        

    return fullBrainNf

def joinBrainArr(leftArr, rightArr):
    fullBrain = np.vstack((np.flip(rightArr.astype(np.float32), axis=0), leftArr.astype(np.float32)))

    return fullBrain


def loadBrainHalves(fileIn):
    brainDict = np.load(fileIn, allow_pickle=True)

    # Get hemispheres and resample
    leftBrain = brainDict.item().get('leftBrain')
    rightBrain = brainDict.item().get('rightBrain')
    
    return [leftBrain, rightBrain]


def plotSplitBrain(leftArr, rightArr, 
                    typeIn='slice',offsetRatio = 0.1,slicePerRow = 10, colorbar=True, 
                    existFig=None, existAx= None,
                    **args):
    """
    type: slice, anat' 
    existFig= set existing figure object. set None to generate automatically
    existAx= list of two axes create with something like:  fig, axs = plt.subplots(2, figsize=(40,8)). set None to generate automatically
    """
    display_modeTmp = 'z'
    

    r= joinBrain(leftArr.astype(np.float32),rightArr.astype(np.float32))

    if typeIn=='slice':
        # f1 = plt.figure(figsize=(20,3));
        # f2 = plt.figure(figsize=(20,3));
        # ni_pl.plot_img( r, display_mode=display_modeTmp, cut_coords=np.linspace(0,10,10),figure=f1, colorbar=True );
        # ni_pl.plot_img( r, display_mode=display_modeTmp, cut_coords=np.linspace(11,20,10),figure=f2, colorbar=True );
        
        # compute max min according to offset
        minZ = int(offsetRatio * r.shape[2])
        maxZ = r.shape[2]-int(offsetRatio * r.shape[2])
        midZ = int(minZ+((maxZ-minZ)/2.))
        
        slicesIdx0 = np.round(np.linspace(minZ, midZ, slicePerRow)).astype(int)
        slicesIdx1 = np.round(np.linspace(midZ, maxZ, slicePerRow)).astype(int)

        if existFig==None:
            fig, axs = plt.subplots(2, figsize=(40,8))
        else:
            assert(len(existAx)==2)
            fig = existFig
            axs = existAx
        ni_pl.plot_img( r, display_mode=display_modeTmp, cut_coords=slicesIdx0,figure=fig, axes=axs[0], colorbar=colorbar, **args )
        ni_pl.plot_img( r, display_mode=display_modeTmp, cut_coords=slicesIdx1,figure=fig, axes=axs[1], colorbar=colorbar, **args )
    else: # anat
        if existFig==None:
            fig = plt.figure(figsize=(12,4))
        else:
            fig = existFig
        
        ni_pl.plot_anat( r, figure=fig, **args )

    return fig
        

def plotBrain(fullArr, typeIn='slice',offsetRatio = 0.1,slicePerRow = 10, colorbar=True, existFig=None, existAx= None, **args):
    """
    plot full brain
    typeIn: slice, anat 
    existFig= set existing figure object. set None to generate automatically
    existAx= list of two axes create with something like:  fig, axs = plt.subplots(2, figsize=(40,8)). set None to generate automatically

    """
    
    fullBrainNf = nib.Nifti1Image(fullArr.astype(np.float32), affine=np.eye(4))   

    return plotBrainNf(fullBrainNf,typeIn,offsetRatio ,slicePerRow, colorbar,   existFig= existFig, existAx=existAx, **args)

        
def plotBrainNf(fullBrainNf, typeIn='slice', offsetRatio = 0.1,slicePerRow = 10, colorbar=True,existFig=None, existAx= None, **args):
    """
    plot full brain from a NF image
    typeIn: slice, anat 
    existFig= set existing figure object. set None to generate automatically
    existAx= list of two axes create with something like:  fig, axs = plt.subplots(2, figsize=(40,8)). set None to generate automatically

    """
    display_modeTmp = 'z'
    

    if typeIn=='slice':
        zMinMaxWorldVec = fullBrainNf.affine @ np.array([[0,0,0,1],[0,0,fullBrainNf.shape[2]-1,1]]).T

        minW, maxW = zMinMaxWorldVec[2,:]

        # # compute max min according to offset
        # minZ = int(offsetRatio * fullBrainNf.shape[2])
        # maxZ = fullBrainNf.shape[2]-int(offsetRatio * fullBrainNf.shape[2])
        # midZ = int(minZ+((maxZ-minZ)/2.))

        # # print(minZ,maxZ,midZ)
        
        # slicesIdx0 = np.round(np.linspace(minZ, midZ, slicePerRow)).astype(int)
        # slicesIdx1 = np.round(np.linspace(midZ, maxZ, slicePerRow)).astype(int)

        # fig, axs = plt.subplots(2, figsize=(40,8))
        # ni_pl.plot_img( fullBrainNf, display_mode=display_modeTmp, cut_coords=slicesIdx0,figure=fig, axes=axs[0], colorbar=colorbar, **args )
        # ni_pl.plot_img( fullBrainNf, display_mode=display_modeTmp, cut_coords=slicesIdx1,figure=fig, axes=axs[1], colorbar=colorbar, **args )

        # compute max min according to offset
        minZ = minW+(offsetRatio * (maxW-minW))
        maxZ = maxW - (offsetRatio * (maxW-minW))
        midZ = (minZ+((maxZ-minZ)/2.))
        
        slicesIdx0 = np.round(np.linspace(minZ, midZ, slicePerRow)).astype(int)
        slicesIdx1 = np.round(np.linspace(midZ, maxZ, slicePerRow)).astype(int)

        if existFig==None:
            fig, axs = plt.subplots(2, figsize=(40,8))
        else:
            assert(len(existAx)==2)
            fig = existFig
            axs = existAx
        
        # Network input
        ni_pl.plot_img( fullBrainNf, display_mode=display_modeTmp, cut_coords=slicesIdx0,figure=fig, axes=axs[0],  colorbar=colorbar,    **args )
        ni_pl.plot_img( fullBrainNf, display_mode=display_modeTmp, cut_coords=slicesIdx1,figure=fig, axes=axs[1],  colorbar=colorbar,   **args )

    else: # anat
        if existFig==None:
            fig = plt.figure(figsize=(12,4))
        else:
            fig = existFig
        ni_pl.plot_anat( fullBrainNf, figure=fig )

    return fig

def plotSplitBrain2(leftArr, rightArr):
    display_modeTmp = 'z'
    
    f1 = plt.figure(figsize=(20,3))
    f2 = plt.figure(figsize=(20,3))
    f3 = plt.figure(figsize=(20,3))
    f4 = plt.figure(figsize=(20,3))
    r= joinBrain(leftArr.astype(np.float32),rightArr.astype(np.float32))

    ni_pl.plot_img( r, display_mode='z', cut_coords=np.linspace(0,10,10),figure=f1, colorbar=True )
    ni_pl.plot_img( r, display_mode='z', cut_coords=np.linspace(11,20,10),figure=f2, colorbar=True )
    ni_pl.plot_img( r, display_mode='z', cut_coords=np.linspace(21,30,10),figure=f3, colorbar=True )
    ni_pl.plot_img( r, display_mode='z', cut_coords=np.linspace(31,40,10),figure=f4, colorbar=True )

def plotSplitBrain3(leftArr, rightArr, typeIn='slice',offsetRatio = 0.2,slicePerRow = 8, colorbar=True, **args):
    """
    type: slice, anat' 
    """
    display_modeTmp = 'z'
    

    r= joinBrain(leftArr.astype(np.float32),rightArr.astype(np.float32))


    if typeIn=='slice':
        # compute max min according to offset
        minZ = int(offsetRatio * r.shape[2])
        maxZ = r.shape[2]-int(offsetRatio * r.shape[2])
        midZ = int(minZ+((maxZ-minZ)/2.))
        
        slicesIdx0 = np.round(np.linspace(minZ, midZ, slicePerRow)).astype(int)
        slicesIdx1 = np.round(np.linspace(midZ, maxZ, slicePerRow)).astype(int)

        fig, axs = plt.subplots(2, figsize=(20,4))
        # Network input
        ni_pl.plot_img( r, display_mode=display_modeTmp, cut_coords=slicesIdx0,figure=fig, axes=axs[0], colorbar=True,  **args )
        ni_pl.plot_img( r, display_mode=display_modeTmp, cut_coords=slicesIdx1,figure=fig, axes=axs[1], colorbar=True,   **args )

        axs[0].set_title('ax0')
        axs[1].set_title('ax1')


    else: # anat
        f1 = plt.figure(figsize=(12,4))
        ni_pl.plot_anat( r, figure=f1, **args )
        
