# Steps to implement a redis-celery-flask job queue system

This is a step-by-step sample of implementing a python job queue with redis, celery and Flask ( using a CTA core detction program by Giancardo Lab ). 

## introduciton

The implementation is Docker-based. This document will cover the step-by-step installation for Celery and Flask, while the REDIS comes from a prebuilt docker image.

There are three Docker containers to build:

### Redis :

Redis is an in-memory data structure store. In our cases, it acts as job queue database: the Flask application will create a task and put it into the Redis job queue. And then, the Celery application will pick up the job from job queue and call a predefined program to run the job.


### Flask : 

A python flask application to provide front end web application. It contains the blow funcitons
- A GUI for the user to upload those files to be processed
- Genereate an session ID and create an input folder with this session ID 
- save uploaded files to the input folder  
- insert the job to the job queu ( with input folder and email address as parameter)
- check the status of job processing and report error message
- Genereate the papaya web display for the output query

### Celery : 

A python celery application to provide the backend function to process the request. It contains the blow funcitons
- Pickup the job and get job parameter (input folder and email address) 
- Get files to be processed from the input folder 
- Call the core funciton to pcoess the input files
- Save output to the output folder
- Send an email when job finishs
- Allow user to check job the status 

A high-level diagram is like this:

<picture>
<img alt="high-level diagram" src="https://github.com/Luyaochen1/glabapp_deploy/blob/main/High-level-diagram.JPG">
</picture>   

## Prerequisite 

### 1. Docker server

Docker is the only software required to run the implementation. Please refer to the below link for the docker server installation.

https://docs.docker.com/desktop/install/linux-install/
 
In the rest part of this documentation, we will assument that user alreay have the sudo right to run the docker commnad.
 
### Docker network  

This documents assums the three containers we listed above is running on the same docker network and IP address was assigned dynamaciily.  For a production deployment, the fixed internal IP address of redis is recommended. Please refer to the Docker network instruction about how to set up the fixed IP address of the container.

Also, we assumene the firewall will allow the user to access the docker port we assigned. Please check witt IT for the firewall rules.

## create the redis container

```

docker run --name myredis -d redis redis-server --save 60 1 --loglevel warning

```
 
## Setup the Application container (Celery Container)

All the below commands are required to run inside the docker container of the application.

### 1.Install prerequisites for celery 

```
python3 -m pip install celery_req.txt

```

The versions specified in the requirements files are not restricted and can be changed according to the other dependency 

### 2.Warp and test the algorithm ( prediction program)

There is original program that can run the prediction. cta-core-detection/run.py

It read the input data from exampleData/ctaAligned_sub-0151.nii.gz and produce output files exampleData/example2dOuput_sub-0150.png and exampleData/example3dOuput_sub-0150.png

```
import gpu_configuration # load GPU configuration (modify as needed)

import inference as inf # load main library



# init core prediction object
corePred = inf.CorePredictor() 

corePred.loadAlignedBrainFromNifti( 'exampleData/ctaAligned_sub-0151.nii.gz' ) # load aligned brain as Nifti file


# Run normalization and ML model
corePred.normAndInfer()

# Output examples
corePred.outputSummary2D('exampleData/example2dOuput_sub-0150.png') # 2D image
corePred.outputAsNifti('exampleData/example3dOuput_sub-0150.nii.gz') # 3D image

```

The wrapped program (cta-core-detection/predict_worker.py) needs to archive :
 - Specify a folder to hold the input and output data:  /papaya_web/data
 - Modify the predict function: "initLoadAndPred(input_file)"  will run the prediction and create the output files according to the input file name
 - An E-mail function: when the job is done, it can send an alert e-mail
 - A core function "process_images(session_id,email)", where:  
   session_id is the index to specify where is the input file located : /papaya_web/data/session_id
   the function will loop each file in side /papaya_web/data/session_id to run the prediction program 
   it will send the email to the address specified by email parameter
 - a funciton decorator "@client.task" to tell celery program that it is acutally a celery funciton can process data in the job queue. 

The wrapped program will load the redis configuration from predict_config.py

```
SECRET_KEY = 'change_it_immediately'
CELERY_BROKER_URL = 'redis://10.100.0.5:6379/0'
CELERY_RESULT_BACKEND = 'redis://10.100.0.5:6379/0'
```

The configuration is shared by the celery job queue definition program and job submission program.

The wrapped program can be tested via the below code.

```
import predict_worker
import time

start_time = time. time()
predict_worker.process_images('test','')   # There is a sample input file in /papaya_web/data/test
stop_time = time. time()
print('##### process time ', stop_time - start_time)

```

### 3.Start up celery as a backend program 



Run the below command from the application container

```
celery  -A predict_worker.client worker  -D --loglevel=INFO --concurrency=2

nohup flower -A predict_celery.client flower --port=5555 &

```

The first line will create a celery job queue call "predict_worker.process_images" - this is the function specified by decorator  "@client.task"

The second line will launch a monitor program (flower) at port 5555. We can check the job queue status and error message there. The flower will use an abstracted function defined by predict_celery.py

```
# predict_celery.py
import os
from celery import Celery
import time
import predict_config

from flask import Flask

app = Flask(__name__)
app.config.from_object("predict_config")

# set up Abstracted celery client
client = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)

@client.task
def process_images(session_id,email):
    return session_id

```

### 4.Test job submission with celery

The job submission is running from another folder ( folder:celery_test .  Actually job submission may happen  on another server; here is just a test)

The folder at least needs two files :
 - predict_config.py has the key and redis server url
 - predict_worker.py the abstracted job queue program defination
 
To add a job to job queue, run
```
from predict_worker import process_images

r = process_images.delay('test','test@uth.tmc.edu')

print('Submitted async {}: {}',r)

```
Here "test" is the folder name with input file(s). 


## Front-end program 

This sample deployment does not include a front-end program. Normally, a front-end program will be responsible for :
- create an input folder ( by session ID ) 
- upload files to the input folder  
- call celery job submission program ( add input folder to the job queue)
- check the status and process error message







