#######################################
#  This is the main funciton for flask
#
######################################

import os,re
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import time
import uuid
from datetime import datetime
import requests
import hashlib

import predict_config

ALLOWED_EXTENSIONS = set(['gz'])

img_folder = predict_config.IMG_FOLDER
site_url  = predict_config.SITE_URL
email_sender = predict_config.EMAIL_SENDER


hash_h = hashlib.blake2b(digest_size=32, person=b'QygcxlOd6YnheN3')

from predict_worker import process_images

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/cta/file_upload', methods=['POST'])
def upload_file():

        session_id = datetime.now().strftime("%Y%m%d") + '-' +  str(uuid.uuid4())

        hash_h = hashlib.blake2b(digest_size=32, person=b'QygcxlOd6YnheN3')
        hash_h.update('session_id'.encode())
        session_id_h = hash_h.hexdigest()

        print(session_id)
        print(request.files)

        # check if the post request has the file part
        if 'file' not in request.files:
                resp = jsonify({'message' : 'No file part in the request'})
                resp.status_code = 400
                return resp

        reply_message=''
        load_files = set()
        file_save_path = img_folder+session_id
        if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)

        for file in request.files.getlist('file'):

                if file.filename == '':
                        reply_message +=  'No file selected for uploading \n'
                if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(file_save_path, filename))
                        load_files.add (filename)
                        reply_message += 'File {} successfully uploaded \n'.format(filename)
                else:
                        reply_message += 'File {} not allowed \n'.format(file.filename)
        resp = jsonify({'message' : reply_message},{'uploaded_files' : list(load_files)},{'session_id':session_id},{'session_id_h':session_id_h})
        print(resp)
        resp.status_code = 201
        return resp

def replace_index_file(session_id,newimgfileinput,newimgfileoutput):

        papaya_index_temp_file = predict_config.PAPAYA_FOLDER + '/index_templdate_session.html'
        papaya_index_file = predict_config.PAPAYA_FOLDER +'/data/{}/index_{}.html'.format(session_id,newimgfileoutput)

        # Safely read the input filename using 'with'
        with open(papaya_index_temp_file) as f:
                print('get index')
                s = f.read()
        # Safely write the changed content, if found in the file
        with open(papaya_index_file, 'w') as f:
                s = s.replace('XXXoutputXXX', newimgfileoutput)
                s = s.replace('XXXinputXXX', newimgfileinput)
                f.write(s)


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib


def send_email_alert ( email, sessoion_id):
	# create message osinstance
	msg = MIMEMultipart()
	# setup the parameters of the message
	msg['From'] =  email_sender
	msg ['To'] =   email
	msg['Subject'] = "CTA Detection job - Do not reply"
	
	message_text = """
    This is the link to your submitted CTA dection job:
    {}/cta/status/{}

	""".format ( site_url,sessoion_id)
	
	msg.attach(MIMEText(message_text, 'plain'))
	server = smtplib.SMTP('smtp.uth.tmc.edu')
	#server.starttls()
	server.sendmail(msg['From'], email, msg.as_string())
	server.quit()

	return


@app.route('/cta/')
def upload_form():
	return render_template('upload.html')

@app.route('/cta/', methods=['POST'])
def upload_image():

	print(request.files)
	print(request.form.get('email'))

	session_id = datetime.now().strftime("%Y%m%d") + '-' +  str(uuid.uuid1())
	file_save_path = img_folder+session_id
	
	hash_h = hashlib.blake2b(digest_size=32, person=b'QygcxlOd6YnheN3')
	hash_h.update(session_id.encode())
	session_id_h = hash_h.hexdigest()

	if 'files[]' not in request.files:
		flash('No file uploaded')
		return redirect(request.url)

	print(file_save_path, ' -- created')

	if not os.path.exists(file_save_path):
		os.makedirs(file_save_path)

	files = request.files.getlist('files[]')
	file_names = []
	messages=[]
	flash('session id: ' + session_id)
	messages.append('session id: ' + session_id)
	for file in files:
		filename = secure_filename(file.filename)
		if allowed_file(file.filename):
			file_names.append(filename)
			file.save(os.path.join(file_save_path, filename))
#			print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('{} successfully uploaded'.format(filename))
			messages.append('{} successfully uploaded'.format(filename))
		else:
			flash('{} is not a valid file'.format(filename))
			messages.append('{}  is not a valid file'.format(filename))
	
	email =  request.form.get('email')

# call to insert job request to celery queue	
	r = process_images.delay(session_id,email)

	print('Submitted async {}: {}',r, file_names)

	
	if email :
		send_email_alert(email,session_id)

#	return render_template('upload.html', filename=[], session_id=session_id)
	return jsonify({'message':messages, 'session_id':session_id,'file_names':file_names,'session_id_h':session_id_h})


@app.route('/cta/display/<path>/<filename>')
def display_image(path,filename):
        #print('display_image filename: ' + filename)
        return redirect(url_for('custom_output', filename=path +'/' + filename), code=301)

# Custom static data
@app.route('/cta/custom_output/<path:filename>')
def custom_output(filename):
    return send_from_directory(app.config['custom_output_PATH'], filename)

#  Check job status Summary
@app.route('/cta/status_summary/<session_id>')
def check_status_summary(session_id):

        file_input_path = img_folder+session_id
        filenames_input = []
        filenames_2d = []
        filenames_3d = []
        for file in os.listdir(file_input_path):
                if  '_result_2d' in file and 'html' not in file:
                        filenames_2d.append(file)
                if  '_result_3d' in file and 'html' not in file:
                        filenames_3d.append(file)
                if  '_result_2d' not in file and '_result_3d' not in file and 'html' not in file:
                        filenames_input.append(file)
        return jsonify({'inputs':len(filenames_input),'outputs_2d': len(filenames_2d), 'outputs_3d': len( filenames_3d), 'session_id': session_id})

#  Check job status (dummy)
@app.route('/cta/status')
def check_status_dummy():
    return "200"



#  Check job status
@app.route('/cta/status/<session_id>')
def check_status(session_id):

        file_input_path = img_folder+session_id
        filenames_input = []
        filenames_2d = []
        filenames_3d = []
        for file in os.listdir(file_input_path):
                if  '_result_2d' in file and 'html' not in file:
                        filenames_2d.append(file)
                if  '_result_3d' in file and 'html' not in file:
                        filenames_3d.append(file)
                if  '_result_2d' not in file and '_result_3d' not in file and 'html' not in file:
                        filenames_input.append(file)
        print(filenames_input)
        print(filenames_2d)
        print(filenames_3d)

        for file in  filenames_input :
                replace_index_file(session_id,file,file.replace('.nii.gz','_result_3d.nii.gz'))

        if len(filenames_input) == len(filenames_2d) and len(filenames_input) == len(filenames_3d) :
                finished = 'Finished'
        else:
                finished = ''
        return render_template('download.html', filenames_2d = filenames_2d, filenames_3d = filenames_3d,  session_id=session_id, finished = finished)

@app.route('/cta/check_hash', methods=['POST'])
def session_hash_check():
    content  = request.json
    print(content)    
    session_id=content['session_id']

    hash_h = hashlib.blake2b(digest_size=32, person=b'QygcxlOd6YnheN3')
    hash_h.update(content['session_id'].encode())
    session_id_h = hash_h.hexdigest()
    print(session_id,session_id_h)
    if session_id_h == content['session_id_h']:
        print('passed')
        return jsonify({'result': 'passed'})
    else:
        return jsonify({'result': 'failed'})


if __name__ == "__main__":
    app.run( host='0.0.0.0',port=8080)
