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
from collections import namedtuple

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt', dest='ckpt', type=str)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')

args = parser.parse_args()



def freeze(ckpt_path):
    
    
    OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                                 gf_dim df_dim output_c_dim')
    options = OPTIONS._make((args.batch_size, args.fine_size,
                                  args.ngf, args.ndf, args.output_nc))
  
  
    inp_content_image = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='input')
 
    
    out_image = generator_resnet(inp_content_image, options,name='generatorA2B')
    out_image = tf.identity(out_image, name='output')
    
    init_op = tf.global_variables_initializer()
    
    restore_saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        restore_saver.restore(sess, ckpt_path)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                        output_node_names=['output'])
        
        path = os.path.dirname(ckpt_path)
        with open(path + '/cyclegan.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


if __name__ == '__main__':
    ckpt_path = args.ckpt
    freeze(ckpt_path)
    print('freeze done')