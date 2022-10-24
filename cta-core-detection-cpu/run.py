#
# Example to run the pipeline
#

# load GPU configuration (modify as needed)

# import gpu_configuration 

import inference as inf # load main library



# init core prediction object
corePred = inf.CorePredictor() 

# corePred.loadAlignedBrainFromNpy( 'exampleData/leftAndFlippedRightBrain_sub-0151.npy' ) # load aligned brain as NPY file
corePred.loadAlignedBrainFromNifti( 'exampleData/ctaAligned_sub-0150.nii.gz' ) # load aligned brain as Nifti file


# Run normalization and ML model
corePred.normAndInfer()

# Output examples
corePred.outputSummary2D('exampleData/example2dOuput_sub-0150.png') # 2D image
corePred.outputAsNifti('exampleData/example3dOuput_sub-0150.nii.gz') # 3D image


