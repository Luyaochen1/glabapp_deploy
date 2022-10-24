#####################################
# This program defines the core function 
# to process a job in the queue
#
######################################

import os
from celery import Celery
import time
import predict_config

from flask import Flask

# celery will use app.config to pass the redis server setup
app = Flask(__name__)
app.config.from_object("predict_config")

import os,re,datetime
import time
import uuid
from datetime import datetime
import json


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

import sys
 
# not use GPU for testing
#import gpu_configuration 

# define the runing parameters

img_folder = app.config['IMG_FOLDER']
site_url  = app.config['SITE_URL']
email_sender = app.config['EMAIL_SENDER']


import inference as inf # load main library

# function to predict one file
def initLoadAndPred(input_file):
    print('prcoess start - {} '.format (input_file) )
    global corePred

    try:
        print( 'corePred object: ', corePred)
    except NameError:
        corePred_exists = False
    else:
        corePred_exists = True
    if not corePred_exists :
        corePred = inf.CorePredictor()

    corePred.loadAlignedBrainFromNifti( input_file ) # load aligned brain as Nifti file

    corePred.normAndInfer()

    result_2d =  input_file.replace('.nii.gz','_result_2d.png')
    result_3d =  input_file.replace('.nii.gz','_result_3d.nii.gz')
    corePred.outputSummary2D(result_2d) # 2D image
    corePred.outputAsNifti(result_3d) # 2D image
    print('prcoess stop - {} '.format (input_file) )
    return [result_2d,result_3d]


def send_email_alert ( email, sessoion_id):
        # create message osinstance
        msg = MIMEMultipart()
        # setup the parameters of the message
        msg['From'] =  email_sender
        msg ['To'] =   email
        msg['Subject'] = "CTA Detection job - Do not reply"

        message_text = """
    Your submitted job is finished.
    This is the link to your submitted CTA dection job:
    {}/cta/status/{}
        """.format ( site_url, sessoion_id)

        msg.attach(MIMEText(message_text, 'plain'))
        server = smtplib.SMTP('smtp.uth.tmc.edu')
#        server.starttls()
        server.sendmail(msg['From'], email, msg.as_string())
        server.quit()

        return

# set up celery client
client = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)

# celery decorator - to tell celery service that  process_images is used to handle the quest send to "client" defined by app.config['CELERY_BROKER_URL']
@client.task
def process_images(session_id,email):
    file_input_path = img_folder+session_id
    files_output = []
    files_input = []
    for file in os.listdir(file_input_path):
        if  '_result_2d' not in file and '_result_3d' not in file and '.html' not in file:
            files_input.append(file_input_path + '/' + file)
            files_output.append(initLoadAndPred(file_input_path + '/' + file) )
    print(files_output)
    print("Thread Job Done !!")

    if email :
        send_email_alert(email,session_id)

    resp ={}
    resp['message'] = 'Job Done'
    resp['session_id'] = session_id
    resp['input_files'] = files_input
    resp['result_files'] = files_output
    resp['process_id'] = os.getpid()
    return json.dumps(resp)
