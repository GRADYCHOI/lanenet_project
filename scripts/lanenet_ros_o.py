#!/usr/bin/env python3

import argparse
import os.path as os

import sys
#sys.path.append('/aveesSSD/lanenet_ws/')

import time
import math
import tensorflow as tf
import numpy as np
import cv2

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

from lanenet_ros.msg import Lane_Image
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

class lanenet_detector():
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic')
        self.output_image = rospy.get_param('~output_image')
        self.output_lane = rospy.get_param('~output_lane')
        self.weight_path = rospy.get_param('~weight_path')
        self.use_gpu = rospy.get_param('~use_gpu')
        self.lane_image_topic = rospy.get_param('~lane_image_topic')

        self.init_lanenet()
        self.bridge = CvBridge()
        sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher(self.output_image, Image, queue_size=1)
        self.pub_laneimage = rospy.Publisher(self.lane_image_topic, Lane_Image, queue_size=1)
        print('\n__init__\n')


    def init_lanenet(self):
        print(self.weight_path)
        LOG.info('Start reading image and preprocessing')
#        t_start = time.time()
#        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#        image_vis = image
#        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
#        image = image / 127.5 - 1.0
#        LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))
    
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    
        net = lanenet.LaneNet(phase='test', cfg=CFG)
        binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    
        postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    
        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
    
        self.sess = tf.Session(config=sess_config)
    
        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
    
        # define saver
#        saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=self.weight_path)
        print('======saver=======') 


##        with sess.as_default():
##            saver.restore(sess=sess, save_path=self.weight_path)
##    
##            t_start = time.time()
##            loop_times = 500
##            for i in range(loop_times):
##                binary_seg_image, instance_seg_image = sess.run(
##                    [binary_seg_ret, instance_seg_ret],
##                    feed_dict={input_tensor: [image]}
##                )
##            t_cost = time.time() - t_start
##            t_cost /= loop_times
##            LOG.info('Single image inference cost time: {:.5f}s'.format(t_cost))
##    
##            postprocess_result = postprocessor.postprocess(
##                binary_seg_result=binary_seg_image[0],
##                instance_seg_result=instance_seg_image[0],
##                source_image=image_vis
##            )
##            mask_image = postprocess_result['mask_image']
##    
##            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
##                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
##            embedding_image = np.array(instance_seg_image[0], np.uint8)
##    
##            plt.figure('mask_image')
##            plt.imshow(mask_image[:, :, (2, 1, 0)])
##            plt.figure('src_image')
##            plt.imshow(image_vis[:, :, (2, 1, 0)])
##            plt.figure('instance_image')
##            plt.imshow(embedding_image[:, :, (2, 1, 0)])
##            plt.figure('binary_image')
##            plt.imshow(binary_seg_image[0] * 255, cmap='gray')
##            plt.show()
##    
##        sess.close()
    
#        '''
#        initlize the tensorflow model
#        '''
#
#        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
#        phase_tensor = tf.constant('test', tf.string)
#        net = lanenet.LaneNet(phase=phase_tensor)
#        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')
#
#        # self.cluster = lanenet_cluster.LaneNetCluster()
#        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()
#
#        saver = tf.train.Saver()
#        # Set sess configuration
#        if self.use_gpu:
#            sess_config = tf.ConfigProto(device_count={'GPU': 1})
#        else:
#            sess_config = tf.ConfigProto(device_count={'CPU': 0})
#        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
#        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
#        sess_config.gpu_options.allocator_type = 'BFC'
#
#        self.sess = tf.Session(config=sess_config)
#        saver.restore(sess=self.sess, save_path=self.weight_path)


    def img_callback(self, data):
    	try:
    		cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    	except CvBridgeError as e:
    		print(e)
    #        cv2.namedWindow("ss")
    #        cv2.imshow("ss", cv_image)
    #        cv2.waitKey(0)
    	original_img = cv_image.copy()
    	resized_image = self.preprocessing(cv_image)
    	mask_image = self.inference_net(resized_image, original_img)
    	out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, "bgr8")
    	self.pub_image.publish(out_img_msg)

    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
#        cv2.namedWindow("ss")
#        cv2.imshow("ss", image)
#        cv2.waitKey(1)
        return image

    def inference_net(self, img, original_img):
        binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                        feed_dict={self.input_tensor: [img]})

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_img
        )
        # mask_image = postprocess_result['mask_image']
        mask_image = postprocess_result
        mask_image = cv2.resize(mask_image, (original_img.shape[1],
                                                original_img.shape[0]),interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.addWeighted(original_img, 0.6, mask_image, 5.0, 0)
        return mask_image


if __name__ == '__main__':
    # init args
#    rospy.init_node('lanenet_node')
    print('\ninit_node\n')
    lanenet_detector()
    rospy.spin()



