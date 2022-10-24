# glab-apps-gui
A production collection for glabapps.uth.edu

# branch for flask / nginx 
code located in flask container:
/app

# Start ( see  branch dockercompose):
base image : luca_flask:20220422

# start up app/service:

cd /app/cta

./uwsgi_start.sh

service nginx restart

# reset / restart flask app

cd /app/cta

touch touch-gui.ini
