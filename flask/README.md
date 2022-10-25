Please see setup instruciont aobut installing the Flask service

# Mange the services on Flask

Nginx - a web serice to pick up the user request for static web contents and recirect dynamic request and APIs to Flask
Flask - a Python applicaiotn server to handle the dynamic web requests or APIs

### start up the services:

cd (where the source code located)/flask/cta

```
./uwsgi_start.sh   # star the flask

service nginx restart  # start / restart the nginx
```

# reset / restart flask app after code change 

cd (where the source code located)/flask/cta

touch touch-gui.ini
