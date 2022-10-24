# this program used to tell celery monitor about the job define. 
# it is a dummy function, no executable feature is rueiqred. 

import os
from celery import Celery
import time
import predict_config

from flask import Flask

app = Flask(__name__)
app.config.from_object("predict_config")

# set up celery client
client = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)

@client.task
def process_images(session_id,email):
    return session_id
