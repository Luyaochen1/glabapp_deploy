Please see setup instruciont aobut installing the Flask service


# start up app/service:

cd (where the source code located)/flask/cta

./uwsgi_start.sh

service nginx restart

# reset / restart flask app after code change 

cd (where the source code located)/flask/cta

touch touch-gui.ini
