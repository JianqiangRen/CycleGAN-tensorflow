# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import tensorflow as tf
import argparse
import os
import cv2
import numpy as np
from module import generator_resnet

parser=argparse.ArgumentParser()
parser.add_argument("--ckpt",dest='ckpt',type=str,required=True)
parser.add_argument("--input",dest='input',type=str)
parser.add_argument("--output",dest='output',type=str)
args=parser.parse_args()

def load_test_data(image_path, fine_size=256):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (fine_size, fine_size))
    img = img/127.5 - 1
    return img


if __name__=="__main__":
    f = tf.gfile.FastGFile(args.ckpt, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, name='')
    
    sess = tf.InteractiveSession(graph=persisted_graph)
    
    content = tf.get_default_graph().get_tensor_by_name("input:0")
 
    output = tf.get_default_graph().get_tensor_by_name("output:0")
    
    content_feed = load_test_data(args.input)
    content_feed = np.expand_dims(content_feed, 0)
 
    output_value = sess.run(output, feed_dict={content: content_feed})[0]
    output_value = (output_value +1.0)*127.5
    
    print(np.min(output_value))
    
    cv2.imwrite(os.path.join(args.output,  os.path.basename(args.input)), np.column_stack(((content_feed[0] +1.0)*127.5,output_value ))  )
    print('saved {}'.format(args.output))
    sess.close()

    