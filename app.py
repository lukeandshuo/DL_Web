import os
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import PIL.Image as Image
import cStringIO as StringIO
import urllib
import exifutil
import sys
import cv2
from Classification import Classification
from multiprocessing import Process,Manager
import tensorflow as tf
#add face verification path
sys.path.append("/home/shuoliu/Research/TF/FaceVerification/openface/demos")
from verification import FaceVerification

#add object detection path
sys.path.append("/home/shuoliu/Research/TF/ObjectDetection/")
from ObjectDetection import ObjectDetection
sys.path.append("")
REPO_DIRNAME = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(REPO_DIRNAME,'tmp/caffe_demos_uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_IMAGE_EXTENSIONS = set(['png',  'jpg', 'jpeg'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer =urllib.urlopen(imageurl).read()
        ext = os.path.splitext(imageurl)[1].strip('.')

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
            ,section="Classification"
        )

    logging.info('Image: %s', imageurl)
    # using multiprocessing to avoid out of memory on GPU
    with Manager() as manager:
        ret = manager.dict()
        p = Process(target=app.clf.classify_image,args=(string_buffer,ext,ret))
        p.start()
        p.join()
        result = ret['result']
    print "url",result
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl, section="Classification")

#
# @app.route('/classify_upload', methods=['POST'])
# def classify_upload():
#     try:
#         # We will save the file to disk for possible data collection.
#         imagefile = flask.request.files['imagefile']
#         filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
#             werkzeug.secure_filename(imagefile.filename)
#         filename = os.path.join(UPLOAD_FOLDER, filename_)
#         imagefile.save(filename)
#         logging.info('Saving to %s.', filename)
#         string_buffer = filename
#         ext = os.path.splitext(filename)[1].strip('.')
#
#     except Exception as err:
#         logging.info('Uploaded image open error: %s', err)
#         return flask.render_template(
#             'index.html', has_result=True,
#             result=(False, 'Cannot open uploaded image.')
#         )
#     # using multiprocessing to avoid out of memory on GPU
#     with Manager() as manager:
#         ret = manager.dict()
#         p = Process(target=app.clf.classify_image,args=(string_buffer,ext,ret))
#         p.start()
#         p.join()
#         result = ret['result']
#     print result
#
#     return flask.render_template(
#         'index.html', has_result=True, result=result,
#         imagesrc=embed_image_html(filename) # TODO
#     )


@app.route('/face_url', methods=['GET'])
def face_url():
    imgurl1 = flask.request.args.get('face1')
    print imgurl1
    imgurl2 = flask.request.args.get('face2')
    print imgurl2
    try:
        # string_buffer =urllib.urlopen(imgurl1).read()
        # ext = os.path.splitext(imgurl1)[1].strip('.')
        url_buffer1 = urllib.urlopen(imgurl1).read()
        img1 = url_to_img(url_buffer1)
        if img1 == None:
            raise ValueError("error image")
        filename1_ = str(datetime.datetime.now()).replace(' ', '_') + "img1.png"
        filename1 = os.path.join(UPLOAD_FOLDER, filename1_)

        url_buffer2 = urllib.urlopen(imgurl2).read()
        img2 = url_to_img(url_buffer2)
        if img2 == None:
            raise ValueError("error image")
        filename2_ = str(datetime.datetime.now()).replace(' ', '_') + "img2.png"
        filename2 = os.path.join(UPLOAD_FOLDER, filename2_)

        cv2.imwrite(filename1,img1)
        cv2.imwrite(filename2,img2)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
            , section="FV"
        )

    logging.info('Image: %s', imgurl1)
    # using multiprocessing to avoid out of memory on GPU
    with Manager() as manager:
        ret = manager.dict()
        p = Process(target=app.face.verification,args=(filename1,filename2,ret))
        p.start()
        p.join()
        result = ret['result']
        drawImg1 = ret['drawImg1']
        drawImg2 = ret['drawImg2']
    print result
    return flask.render_template(
        'index.html', has_result=True, result=result, drawImg1=embed_cv_image_html(drawImg1),drawImg2=embed_cv_image_html(drawImg2), section="FV")


@app.route('/od_url', methods=['GET'])
def od_url():
    imgurl = flask.request.args.get('image')
    print imgurl
    try:
        # string_buffer =urllib.urlopen(imgurl1).read()
        # ext = os.path.splitext(imgurl1)[1].strip('.')
        url_buffer1 = urllib.urlopen(imgurl).read()
        img = url_to_img(url_buffer1)
        if img == None:
            raise ValueError("error image")
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + "img.png"
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        cv2.imwrite(filename,img)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
            , section="OD"
        )

    logging.info('Image: %s', imgurl)
    # using multiprocessing to avoid out of memory on GPU
    with Manager() as manager:
        ret = manager.dict()
        p = Process(target=app.od.detect,args=(filename,ret))
        p.start()
        p.join()
        result = ret['result']
        drawImg = ret['drawImg']
    print result
    return flask.render_template(
        'index.html', has_result=True, result=result, drawImg=embed_cv_image_html(drawImg), section="OD")

def url_to_img(url_buffer):
    image = np.asarray(bytearray(url_buffer),dtype="uint8")
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    return image

def embed_cv_image_html(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img)
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def embed_image_html(imagename):
    """Creates an image embedded in HTML base64 format."""
    image = exifutil.open_oriented_im(imagename)
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    app.clf = Classification()
    app.face = FaceVerification()
    app.od = ObjectDetection()

    # cv2.imshow("luke",img)
    # cv2.waitKey(0)
    # img1 = "luke1.jpg"
    # img2 = "luke2.jpg"
    # Same,drawImg1,drawImg2=app.face.verification(img1,img2)
    # if Same:
    #     print "same"
    # else:
    #     print "different"
    # drawImg = np.concatenate((drawImg1,drawImg2),axis=1)
    # cv2.imshow("face",drawImg)
    # cv2.waitKey(3000)
    # Initialize classifier + warm start by forward for allocation
    # with Manager() as manager:
    #     ret = manager.dict()
    #     p = Process(target=app.clf.classify_image,args=("test.jpg","jpg",ret))
    #     p.start()
    #     p.join()
    #     print ret
    #warm up
    # for i in range(2):
    #     app.clf.classify_image("test.jpg",'jpg')

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
