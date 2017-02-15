import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time
import sys
import urllib2
import os
slim = tf.contrib.slim
from synset import *

from datasets import imagenet
from nets import resnet_v1
from preprocessing import vgg_preprocessing

class Classification(object):

    def __init__(self,checkpoints_dir = 'checkpoints/'):
        self.checkpoints_dir = checkpoints_dir
        self.valid_ext = ['png','jpg','jpeg','gif']
        self.g = tf.Graph()
    def _compute(self):
        pass
    def classify_image(self,image_string,ext='png',ret = None):

        if ext not in self.valid_ext:
            # print "wrong image formatg"
            ret['result'] = (False,"please input valid image format", "png,jpg,jpeg,gif")
            return ret['result']
        try:

            image_size = resnet_v1.resnet_v1.default_image_size
            with self.g.as_default():
                #if image is from local then read file firstly
                if os.path.splitext(image_string)[1].strip(".") in self.valid_ext:
                    # print "image from local"
                    image_string = tf.read_file(image_string)
                if ext == "jpeg" or "jpg":
                    # print "jpg"
                    image = tf.image.decode_jpeg(image_string, channels=3)
                if ext == "png":
                    image = tf.image.decode_png(image_string,channels=3)
                if ext == 'gif':
                    image = tf.image.decode_gif(image_string,channels=3)
                processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
                processed_images = tf.expand_dims(processed_image, 0)
                # print "1"
                # Create the model, use the default arg scope to configure the batch norm parameters.
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    logits, _ = resnet_v1.resnet_v1_50(processed_images, num_classes=1000, is_training=False)
                probabilities = tf.nn.softmax(logits)

                init_fn = slim.assign_from_checkpoint_fn(
                    os.path.join(self.checkpoints_dir, 'resnet_v1_50.ckpt'),
                    slim.get_model_variables())
                # print "2"
            with self.g.as_default():
                with tf.Session() as sess:
                    init_fn(sess)
                    starttime = time.time()
                    np_image, probabilities = sess.run([image, probabilities])
                    endtime = time.time()
                    probabilities = probabilities[0,0,0,0:]
                    sorted_inds = np.argsort(probabilities)[::-1]
            indices = sorted_inds[:5]
            preditions = synset[indices]
            meta = [(p,'%.5f'% probabilities[i]) for i, p in zip(indices,preditions)]
            ret['result']=(True,meta,'%.3f' % (endtime-starttime))
            return ret['result']
        except Exception as err:
            # print "error"
            ret['result'] = (False,"someting went wrong when classifying the image,", "Maybe try another one?")
            return ret['result']

if __name__ =='__main__':
    url ="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    image_string = urllib2.urlopen(url).read()
    ext = os.path.splitext(url)[1].strip('.')
    cls = Classification()
    re,meta,t=cls.classify_image(image_string,ext)
    # if re:
    #     print "time:",t
    #     for i in meta:
    #         print in
