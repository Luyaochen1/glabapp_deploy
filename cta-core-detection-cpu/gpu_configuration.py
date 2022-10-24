#
# Configure GPU settings for the whole project
# author: Luca Giancardo  
#

import sys,os

# # disable GPU
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]=""; 
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# enable GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="6"; 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
