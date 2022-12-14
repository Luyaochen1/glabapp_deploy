# Steps to create a python based job queue system with redis, celery and flask

This is a step-by-step guide of implementing a python based job queue with redis, celery, and flask ( with a CTA core detection algothium by Giancardo Lab ). 

## Introduction

The implementation is Docker-based. This document will cover the step-by-step installation for Celery job server and Flask web server, while the REDIS database server comes from a prebuilt docker image.

A high-level diagram is like this:

<picture>
<img alt="high-level diagram" src="https://github.com/Luyaochen1/glabapp_deploy/blob/main/High-level-diagram-2.JPG">
</picture>   

There are three Docker containers to build:

### Redis Database Server:

Redis is an in-memory data structure store. In our cases, it acts as a job queue database: the Flask application will create a task and put it into the Redis job queue. And then, the Celery application will pick up the job from the job queue and call a predefined program to run the job.

Both Celery and Flask need a security key to access it.

### Flask Web Server : 

A python flask application server to provide front-end web application. The web GUI service for user to upload a file to process: a job will be created and insert to Redis database. 

It contains the blow functions,

- A GUI (Nginx+flask) for the user to upload those files to be processed
- Generate an session ID and create an input folder with this session ID 
- save uploaded files to the input folder  
- insert the job to the job queue ( with input folder and email address as  parameters)
- check the status of job processing and report an error message
- Generate the papaya web display for the output query

### Celery Job Server : 

A Celery backend service who pick up the job from the queue and execute it with a predfine funciton ( the image process algothium) 

It contains the blow functions,

- Pickup the job and get the job parameters (input folder and email address) 
- Get files to be processed from the input folder 
- Call the core function to process the input files
- Save output to the output folder
- Send an email when job finishes
- Allow user to check job the status 


The function flow is described as 

<picture>
<img alt="high-level diagram" src="https://github.com/Luyaochen1/glabapp_deploy/blob/main/Flow-Flask-Celery-Redis.png">
</picture>   



## Prerequisite 

### Docker server

Docker is the only software required to run the implementation. Please refer to the below link for the docker server installation.

https://docs.docker.com/desktop/install/linux-install/
 
In the rest part of this documentation, we will assume that the user already has the sudo right to run the docker command.
 
### Docker network  

This document assums the three containers we listed above are running on the same docker network and IP address was assigned dynamically.  For a production deployment, the fixed internal IP address of redis server is recommended. Please refer to the Docker network instruction about how to set up the fixed IP address of the container.

The Flask web server will use internal port 8080 for its web applications. In this example, the port will be mapped to port 7605 of the server.

The Celery will use port internal 5555 for its monitoring program. In this example, the port will be mapped to port 7606 of the server.

The Redis will use port internal 7379 for its database access. This port is not accessible from external.

Also, if you run these containers on a server, we assume the firewall will allow the user to access port 7605 and 7606. Please check with IT for the firewall rules. Also, you must know the server IP address for the user to access the web pages.

In our example, the docker server IP address is 129.106.31.204, and the web application can be accessed by http://129.106.31.204:7605/cta/ 

If you try it on your own machine (windows/Mac laptop or PC), you can replace the server IP address with localhost. Then, the web application can be accessed by http://localhost:7605/cta/ 


## Create Docker containers

### clone source code from Github

```
cd  <your working folder>  # any folder the user has the full access
git clone https://github.com/Luyaochen1/glabapp_deploy.git
```

## create the redis database server container

Run the below command to create the redis container
```
sudo docker run --name glabapps_redis_test -d redis redis-server --save 60 1 --loglevel warning

```

After we start up the redis container. We need to find out its IP address ( to be used for further configuration) 

Run 
```
sudo docker inspect glabapps_redis_test

```
and search for "IPAddress" in the output.

In our example, we will use 172.17.0.7 as the internal IP address of the Redis server.


 
## Create the Celery job server container  

Run the below command to create the celery container. The container is built based on a python 3.8.15 image.
```
cd   <your working folder>
sudo docker run -it -d -p7606:5555 -v$(pwd)/glabapp_deploy:/glabapp_deploy --name=glabapps_celery_test  python:3.8.15 bash
```

and run the below command to enter the Celery container

```
sudo docker exec -it glabapps_celery_test bash
root@xxxxxxxx:/#
```

and then, run the below Linux commands to install the requirements

```
# install a text editor to change some setup later
apt update -y && apt install nano -y

# go to the home folder for celery application 
cd /glabapp_deploy/cta-core-detection-cpu

# install required python packages for CTA detections 
pip install -r requirements.txt

# install required python packages for Celery 
pip install -r celery_req.txt

```

Now, we can try to run the original CTA detection program ot make sure the required packages are installed.

```
python run.py
......
......
avg grad 0 tf.Tensor(2.9693632e-05, shape=(), dtype=float32)
avg grad 1 tf.Tensor(-2.9693632e-05, shape=(), dtype=float32)
(1, 10, 23, 17, 24)
4.8814295e-06
->
root@92a9aa239f8e:/glabapp_deploy/cta-core-detection-cpu#
```

To start the celery job service, we need to check the configuration file and ensure the redis address is pointed to the IP address of the redis container. And the same time, update the base URL of the flask server with the actual server IP address( you can use localhost if you test it from a local PC/MAC). 

Also, to ensure security, change the SECRET_KEY if required. 

The same setup will be used for all the other applications.

```
# next 3 lines is about accessin the redis server
SECRET_KEY = 'change_it_immediately'
CELERY_BROKER_URL = 'redis://172.17.0.7:6379/0'
CELERY_RESULT_BACKEND = 'redis://172.17.0.7:6379/0'
# specify where to hold the job data
IMG_FOLDER = '/glabapp_deploy/papaya/data/'
# these lines is help the email body to generate the correct URLs for image viewer
SITE_URL = 'http://129.106.31.204:7605'
EMAIL_SENDER  = 'admin@cta.uth.tmc.edu'
```

The next step is to start the celery server job service and monitor service

```
# start a celery job service
celery  -A predict_worker.client worker  -D --loglevel=INFO --concurrency=2

# start a job monitor service
nohup flower -A predict_celery.client flower --port=5555 &

# exist the celery docker and go to the next step
exit
```

by entering the URL http://129.106.31.204:7606  (repalce 129.106.31.204 with your server IP or localhost if run locally), we can access the celery monitor tool. 

We will test the celery job queue function after creating the flask web server container


## Create the Flask web server container  

Run the below command to create the flask container. The container is built based on a python 3.8.15 image.
```
cd   <your working folder>
sudo docker run -it -d -p7605:8080 -v$(pwd)/glabapp_deploy:/glabapp_deploy --name=glabapps_flask_test  python:3.8.15 bash

```

and run the below command to enter the flask container

```
sudo docker exec -it glabapps_flask_test bash
root@xxxxxxxx:/#
```

and then, run the below Linux commands to install the requirements

```
#  install an editor and nginx program
apt update -y && apt install nano nginx -y

# go to the base folder for flask
cd /glabapp_deploy/flask

# install python packages
pip install -r flask_req.txt 

# change the owner of programs so that flask service can then update them
chown www-data:www-data /glabapp_deploy/flask -R
chown www-data:www-data /glabapp_deploy/papaya/data -R

# go to the flask program folder
cd /glabapp_deploy/flask/cta

# create a folder for logs
mkdir /var/log/uwsgi

# modfify the IP address configurations
nano /glabapp_deploy/flask/cta/predict_config.py
```
Here, the IP addresses in [/glabapp_deploy/flask/predict_config.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/flask/cta/predict_config.py) must follow the Redis server IP address we mentioned earlier in this document.


And then, we can start the services required for Flask web server. 

```
#start the flask service
./uwsgi_start.sh 

# copy the ngxin config files
cp /glabapp_deploy/nginx/cta-detect /etc/nginx/sites-enabled/.
cp /glabapp_deploy/nginx/nginx.conf /etc/nginx/.
rm /etc/nginx/sites-enabled/default

# test nginx setup
nginx -t

#run nginx as a service
service nginx start 

``` 
Now, CTA Detection application can be accessed by URL http://129.106.31.204:7605/cta/    (repalce 129.106.31.204 with your server IP or localhost if run locally) ;  we can also go back to the celery monitor tool (http://129.106.31.204:7606) to check the job status.

Try to upload a image file to test the application. 



## Coding instruction 

Please refer to the comments of each program for the coding instructions 

There is just a list of related programs and highlight their functions. 

### Celery job server 

#### [/glabapp_deploy/cta-core-detection-cpu/predict_worker.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/cta-core-detection-cpu/predict_worker.py)

predict_worker.py contains the core function process_images(session_id,email)  to:

- decide working folder as base_folder + session_id
- for each file in the working folder, run the prediction program
- send the email after the job done

process_images(session_id,email) has a function decorator @client.task. 
When starting the celery service, "-A predict_worker.client" parameter goes to the "@client.task" function of predict_worker.py to process the job queue request. 

#### [/glabapp_deploy/cta-core-detection-cpu/predict_celery.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/cta-core-detection-cpu/predict_celery.py)

predict_celery.py is an abstract function that runs the celery monitor service. It has the same function definition, but actually, it does nothing. Having this abstract function is to avoid the system do something when loading the monitor program.

#### [/glabapp_deploy/cta-core-detection-cpu/predict_config.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/cta-core-detection-cpu/predict_config.py)

This is the configuration file we discussed above to hold the radis server IP address, port, and security key, and the urls for generating the email body.


### Flask web server 

#### [/glabapp_deploy/flask/cta/main.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/flask/cta/main.py)
This is the flask program to provide the GUI and API service. it uses the below code to insert a job into redis job queue.
```
from predict_worker import process_images

r = process_images.delay(session_id,email_address)
```


#### [/glabapp_deploy/flask/predict_worker.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/flask/cta/predict_worker.py)

predict_worker.py is an abstract function. Having this abstract function here is to simplify the coding: the flask program just need to know the interface to the job processor, not the function itself. 

#### [/glabapp_deploy/flask/predict_config.py](https://github.com/Luyaochen1/glabapp_deploy/blob/main/flask/cta/predict_config.py)

This is the configuration file we discussed above to hold the radis server IP address, port, and security key: it shall be the same as the one on the celery server.

#### [/glabapp_deploy/flask/cta/templates/upload.html](https://github.com/Luyaochen1/glabapp_deploy/blob/main/flask/cta/templates/upload.html)

This is the actual web page template when you enter the http://server_IP_address:server_port/cta/.

This web page template has many other functions not discussed in this document. Here is just a list for reference,
- Styles 
- Cookie for showing privacy alert
- Cookie to limit the number of uploads a day
- Javascript to show the upload status
- Javascript to show the job process upload status
- Cookie to limit that only the user who submits the job can download the result
 

