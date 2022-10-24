#
# Example to run the pipeline concurrently
#


import gpu_configuration # load GPU configuration (modify as needed)

import inference as inf # load main library
import threading
import multiprocessing

# Create two functions to run concurrently

def initLoadAndPred1():
    # init core prediction object
    corePred1 = inf.CorePredictor() 
    # corePred.loadAlignedBrainFromNpy( 'exampleData/leftAndFlippedRightBrain_sub-0151.npy' ) # load aligned brain as NPY file
    corePred1.loadAlignedBrainFromNifti( 'exampleData/ctaAligned_sub-0151.nii.gz' ) # load aligned brain as Nifti file
    # Run normalization and ML model
    corePred1.normAndInfer()
    # Output examples
    corePred1.outputSummary2D('exampleData/example2dOuput_sub-0151.png') # 2D image

def initLoadAndPred2():
    corePred2 = inf.CorePredictor() 
    corePred2.loadAlignedBrainFromNifti( 'exampleData/ctaAligned_sub-0150.nii.gz' ) # load aligned brain as Nifti file
    corePred2.normAndInfer()
    corePred2.outputSummary2D('exampleData/example2dOuput_sub-0150.png') # 2D image



# Test with multiple threads
thr1 = threading.Thread(target=initLoadAndPred1, args=(), kwargs={})
thr2 = threading.Thread(target=initLoadAndPred2, args=(), kwargs={})

# run concurrently with threads
thr1.start()
thr2.start()


# Test with multiple processes
proc1 = multiprocessing.Process(target=initLoadAndPred1)
proc2 = multiprocessing.Process(target=initLoadAndPred2)

# run concurrently with processes
proc1.start()
proc2.start()
