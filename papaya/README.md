### This folder holds the code for papaya viewer and the data for images

#### Papaya Viewer

Papaya is a pure JavaScript medical research image viewer, supporting DICOM and NIFTI formats. Please refer to  https://mangoviewer.com/papaya.html

#### data fokder

For each job, the flask web server will create a unique ID as the job folder. The Celery job server will then pick up each file in the folder, process it, and save the result into the same file.
