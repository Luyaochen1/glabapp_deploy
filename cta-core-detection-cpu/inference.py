#
# Implements functionality to preprocess the data and run pretrained inference model to predict core  
# author: Luca Giancardo  
#

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

import tensorflow as tf
import pandas as pd
import numpy as np
import utils
import ni_utils
import view_utils
import skimage as ski
import nibabel as nib
import scipy.ndimage as nd_img
import threading

import ml.cnn_net


class CorePredictor:
    INITIAL_WEIGHT_FILENAME = 'res/paper2020-cor-ord-exp1_weights.27-val_core_mse109.79.hdf5'
    INPUT_DIM = (73, 182, 133, 1) # dimension of half brain
    CLIP_UP_TH = 100
    CLIP_LOW_TH = 0
    # models shared beetween objects
    MODEL = None
    MODEL_WEIGHTS = None
    SAL_VIS = None

    SEM_INIT = threading.Semaphore() # semophore for the initialization

    def __init__(self, modelWeigths=INITIAL_WEIGHT_FILENAME) -> None:
        """[summary]

        Args:
            modelWeigths ([type], optional): [description]. Defaults to INITIAL_WEIGHT_FILENAME.
        """
        CorePredictor.SEM_INIT.acquire()
        if CorePredictor.MODEL_WEIGHTS != modelWeigths:
            CorePredictor.MODEL_WEIGHTS = modelWeigths
            ## see 
            # dice-core-rapid-fiv-GradCAM-final.ipynb from stroke repository
            # and  



            # load model/weights
            OUTCOME_CRITERIAS = ['core', 'pen']
            CorePredictor.MODEL  = ml.cnn_net.sNetVggNetWithSkipConnBNsOrd(CorePredictor.INPUT_DIM,
                                                                            depthBefore=3,
                                                                            depthAfter=2,
                                                                            activation='relu',
                                                                            nFilters=24, 
                                                                            nConv=3,
                                                                            globAvgPool=True,
                                                                            addDenseLayerNeurons=15,
                                                                            tasksLst=OUTCOME_CRITERIAS )

            CorePredictor.MODEL.load_weights( CorePredictor.MODEL_WEIGHTS )

            # load saliency visualizer 
            CorePredictor.SAL_VIS = ml.cnn_net.SaliencyVisualizer(CorePredictor.MODEL, rmSoftMax=True)
        CorePredictor.SEM_INIT.release()

        # assign shared class attributes to object attributes  
        self._modelWeigths = CorePredictor.MODEL_WEIGHTS
        self._salVis = CorePredictor.SAL_VIS
        self._model  = CorePredictor.MODEL
        # Init input/output
        self.initInputOutput()

    def initInputOutput(self):
        self.brainNi = None
        self.brainArr = None
        self.lBrainArr = None # left hemisphere input
        self.rBrainArr = None # right hemisphere input
        self.brainLoaded = False
        self.actArr = None
        self.mOut = {'prob': None} # numerical output from the model

    def normBrainForSalVis(self,brainHalfArr):
        """Brain normalization for saliency visualizer

        Args:
            brainHalfArr ([type]): [description]

        Returns:
            [type]: [description]
        """
        print(np.max(brainHalfArr.ravel()))

        normVals = [0, np.max(brainHalfArr.ravel())] 
        resN = np.clip(brainHalfArr, normVals[0],   normVals[1])
        resN /= normVals[1]
        
        resN *= 100

        resN=resN.astype(int)
        resN2=nd_img.gaussian_filter(resN, sigma=1.5, truncate=4.0)
        
        return resN2

    def loadAlignedBrainFromNpy(self, brainNpyFile) -> None:
        """Load the hemispheres from npy file. Note that in this case left and right hemispheres could be flipped during the visualization.

        Args:
            brainNpyFile (str): location of npy file. Requires 'leftBrain' and 'rightBrain' keys containign the required  matrices as numpy Arrays
        """
        self.initInputOutput()

        brainDict = np.load(brainNpyFile, allow_pickle=True)
        # print(FILEPATH)

        # Get hemispheres and resample
        self.lBrainArr = brainDict.item().get('leftBrain')
        self.rBrainArr = brainDict.item().get('rightBrain')

        # join
        self.brainArr = view_utils.joinBrainArr( self.lBrainArr, self.rBrainArr  )
        # create a full nifti brain assuming identity affine matrix
        self.brainNi = view_utils.joinBrain( self.lBrainArr, self.rBrainArr  )

        self.brainLoaded = True

    def loadAlignedBrainFromNifti(self, brainNiFile) -> None:
        """Load aligned hemispheres from Nifti file

        Args:
            brainNpyFile (str): location of npy file. Requires 'leftBrain' and 'rightBrain' keys containign the required  matrices as numpy Arrays
        """
        self.initInputOutput()

        self.brainNi = nib.load(brainNiFile)
        self.brainArr = self.brainNi.get_fdata()
        # clip and save
        self.brainArr = np.clip(self.brainArr, self.CLIP_LOW_TH, self.CLIP_UP_TH)

        self.brainNi = nib.Nifti1Image(self.brainArr, affine= self.brainNi.affine ) 
        h = self.brainArr.shape[0]

        # Get hemispheres and resample
        self.lBrainArr = self.brainArr[int(h/2):h, :, :]
        self.rBrainArr = np.flip(self.brainArr[0:int(h/2), :, :], axis=0)

        self.brainLoaded = True

    def normAndInfer(self) -> bool:
        """run secondary preprocessing / normalization and inference of loaded brain

        Returns:
            bool: True if successfull
        """

        if not self.brainLoaded:
            print('No brain loaded')
            return False

        # run secondary preprocessing and normalization before network input
        res = ni_utils.loadNpyAndpreProcessVol4(self.lBrainArr, self.rBrainArr,maskIn=False, extractVessels=False, normNoVess=True)
        [redCoreLamL,redCoreLamR, prob] = self._salVis.getGradCam(res,classIdx=None,outcomeIdx=0, onlyPos=True) #outcomeIdx=0 core, outcomeIdx=1 pen

        #-- Set contralateral to 0 and normalize, using CTA values (v.2)
        # estimate and normalize best map
        bestCoreMap = None
        if ( np.sum(redCoreLamR) > np.sum(redCoreLamL)  ):
            bestCoreMap = self.normBrainForSalVis(redCoreLamR)
            print('<-')
        else:
            bestCoreMap = self.normBrainForSalVis(redCoreLamL)
            print('->')
        # detect map corresponding to lowest HU on CTA
        resL, resR = res[0][0,:,:,:,0], res[1][0,:,:,:,0]
        thCtaMap = 50
        if ( np.mean(resR[bestCoreMap>thCtaMap].ravel()) < np.mean(resL[bestCoreMap>thCtaMap].ravel())  ):
            redCoreLamL = bestCoreMap * 0
            redCoreLamR = bestCoreMap
        else:
            redCoreLamL = bestCoreMap
            redCoreLamR = bestCoreMap * 0
        #--

        self.actArr = view_utils.joinBrainArr( redCoreLamL, redCoreLamR.astype(np.float32) )
        self.mOut['prob'] = prob

        return True


    def outputSummary2D(self, outFile) -> None:
        """Output a 2D summary of input and ouput

        Args:
            outFile (string): output file ending as png

        Returns:
            None
        """
        if  self.actArr is None:
            print('No brain was inferred')
            return False

        # adapted from paper2020-seg-fiv-gradcam_final.py
        nBrainPlots = 2
        fig, axs = plt.subplots(2*nBrainPlots, figsize=(40,8*nBrainPlots))

        typeVis='slice'
        lstAxId=0 
        view_utils.plotBrainNf( self.brainNi, typeIn=typeVis, colorbar=True, existFig=fig, existAx=axs[lstAxId:lstAxId+2], cmap='gray', annotate=True);
        lstAxId +=2
        actNi = nib.Nifti1Image(self.actArr, affine=self.brainNi.affine)
        view_utils.plotBrainNf( actNi, typeIn=typeVis, existFig=fig, existAx=axs[lstAxId:lstAxId+2], vmin=0, vmax=100, annotate=False);
        lstAxId +=2

        plt.savefig(outFile, dpi=150, bbox_inches='tight')

        # release memory
        fig.clf()    
        plt.close()
        del actNi
        gc.collect()



    def outputAsNifti(self, outFile) -> None:
        """Output a 3D core as nifti 

        Args:
            outFile (string): output file 

        Returns:
            None
        """
        if  self.actArr is None:
            print('No brain was inferred')
            return False



        actNi = nib.Nifti1Image(self.actArr, affine=self.brainNi.affine) 
        actNi.to_filename(outFile)
