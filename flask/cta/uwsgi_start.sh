uwsgi --ini  cta_gui.ini --die-on-term --honour-stdin  --touch-reload=touch-gui.ini --master --daemonize /var/log/uwsgi/cta-gui.log
