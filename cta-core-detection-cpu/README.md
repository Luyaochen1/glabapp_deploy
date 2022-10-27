# CTA Core Detection Pipeline

See run.py for running example
Remeber to edit gpu_configuration.py to reflect your configuration

exampleData directory contains examples for 
- Aligned 3D CTA brain (Nifti): ctaAligned_sub-0151.nii.gz
- Aligned 3D CTA brain (npy): ctaAligned_sub-0151.npy (formerly leftAndFlippedRightBrain_sub-0151.npy)
- Example 2D output: example2dOuput_sub-0151.png
- Example 3D output: example3dOuput_sub-0151.nii.gz

# Additional Files to run CTA  detection as a celery job queue

[predict_config.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/cta-core-detection-cpu/predict_config.py): the configuriaotn file for job queue ( including the security key to redis server)

predict_worker.py(https://github.com/Luyaochen1/glabapp_deploy/blob/main/cta-core-detection-cpu/predict_worker.py)  - define the main funciton to pick up the job queue and run prediction

predict_celery.py(https://github.com/Luyaochen1/glabapp_deploy/blob/main/cta-core-detection-cpu/predict_celery.py)  - an abstract fuction provide funciton defination only for submitting job or lauch a celery monitoring tools

# Run the celery job queue service

### celery job queue service

```
celery  -A predict_worker.client worker  -D --loglevel=INFO --concurrency=2
```

Here,

Following -A is the function name - the function decorated by @client.task in  predict_worker.py 
"worker" -  the celery queue name
-D  - make the celery as a service
--concurrency=2  - 2 concurrent processes to handle the job at the backend


### celery job queue monitor

```
nohup flower -A predict_celery.client flower --port=5555 &
```

Here,

Following -A is the function name - the function decorated by @client.task in  predict_celery.py 




