from flask import Flask

#UPLOAD_FOLDER = 'static/uploads/'

#UPLOAD_FOLDER = '/papaya_web/input/'
CUSTOM_OUTPUT_PATH =  '/glabapp_deploy/papaya/data/'

app = Flask(__name__)
app.secret_key = "secret key"
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['custom_output_PATH'] = CUSTOM_OUTPUT_PATH
