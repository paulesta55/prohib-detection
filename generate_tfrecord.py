from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import pandas as pd
import sys
import os
import cv2
sys.path.append("../../models/research")
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('label', '', 'Name of class label')
flags.DEFINE_string('img_path', '', 'Path to the images')
FLAGS = flags.FLAGS


def create_tf_example(element,path):

    with tf.gfile.GFile(os.path.join(path, '{}'.format(element[0])), 'rb') as fid:
        encoded_jpg = fid.read()
  # TODO(user): Populate the following variables from your example.
    im = cv2.imread(os.path.join(path, '{}'.format(element[0])))
    height =  800# Image height
    width = 1360 # Image width
    filename = element[0].encode('utf8') # Filename of the image. Empty if image is not from file
    image_format = b'jpg' # b'jpeg' or b'png'
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    for i in element[1]['x1']:
        xmins.append(i/width)

    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    for k in element[1]['x2']:
        xmaxs.append(k/width)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)

    for l in element[1]['y1']:
        ymins.append(l/height)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)   
    for m in element[1]['y2']:
        ymaxs.append(m/height)
        classes_text.append('red-circle'.encode('utf8'))
        print(classes_text)
	classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    data = pd.read_csv(FLAGS.csv_input,sep=";")
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    data.columns = ["img", "x1", "y1", "x2", "y2","id"]
    # TODO(user): Write code to read in your dataset to examples variable
    for el in data.groupby('img'):
        tf_demo = create_tf_example(el,path)
        writer.write(tf_demo.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == '__main__':
    tf.app.run()
