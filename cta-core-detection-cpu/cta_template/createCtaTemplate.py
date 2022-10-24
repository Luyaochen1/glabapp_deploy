'''
Script to generate a CTA-specific template

'''

import numpy as np
from tqdm import tqdm
import nibabel as nib
import glob

# directory containing CTA images to use as template (not included in the repository)
INPUT_DIR = '/data/giancardo-group/lgiancardo/stroke/stroke/dataset_orig/brain/pre_processed_FSL/saturated_resizedTo0.5_nifti/'
# pattern to recognise CTA files
CTA_FILES = 'brainNetworkInput_sub-*.nii.gz'

OUTPUT_TEMPLATE = 'cta_template_20220408.nii.gz' # 733 images


filesList = glob.glob(INPUT_DIR + '/'+ CTA_FILES , recursive=0)

#== create brain Accumulator
brainNi = nib.load(filesList[0])
brainArr = brainNi.get_fdata()
arrType = brainNi.get_data_dtype()
brainAff = brainNi.affine
#==

newShape = list(brainArr.shape) + [len(filesList)]
brainArrAcc = np.zeros( newShape, arrType)

#== Load brain accumulator
i = 0
for ctaFile in tqdm(filesList):
    brainNi = nib.load(ctaFile)
    brainArrAcc[:,:,:,i] = brainNi.get_fdata()

    i += 1
#==


# compute me
brainMeanArr = np.mean( brainArrAcc, axis=3 )

resNi = nib.Nifti1Image(brainMeanArr, brainAff)

nib.save(resNi, OUTPUT_TEMPLATE)