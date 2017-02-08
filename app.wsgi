
import sys
sys.path.insert(0,"/home/shuoliu/Research/TF/tf-image-classification")
from app inport app as application

<VirtualHost *>
    ServerName tf-image-classification

    WSGIDaemonProcess tf-image-classification user=shuoliu group=shuoliu threads=5 python-path=/home/shuoliu/anaconda2/envs/tf/lib/python2.7/site-packages
    WSGIScriptAlias /tf-image-classification /var/www/tf-image-classification/tf-image-classification.wsgi

    <Directory /var/www/tf-image-classification>
        WSGIProcessGroup tf-image-classification
        WSGIApplicationGroup %{GLOBAL}
        Order deny,allow
        Allow from all
    </Directory>
</VirtualHost>