#!/usr/bin/env python3

import argparse
import os.path as os

import sys
sys.path.append('/home/choiin/lanenet_ws/src/lanenet_ros/')

import time
import datetime
import math
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import PIL
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

from lanenet_ros.msg import Lane_Image, Lane, Curve
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

from tensorflow.python.client import device_lib

count = 0
num = 1
img_np = np.zeros((720,1280,3))
img_np2 = np.zeros((720,1280,3))
roi_x = 50
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
        sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=5)
        self.pub_image = rospy.Publisher(self.output_image, Image, queue_size=5)
#        self.pub_laneimage = rospy.Publisher(self.lane_image_topic, Lane_Image, queue_size=1)
#        self.pub_laneimage = rospy.Publisher(self.output_image, CompressedImage, queue_size=1)
#        self.pub_lane = rospy.Publisher(self.output_lane, Image, queue_size=1)


    def init_lanenet(self):
        print(self.weight_path)
        device_lib.list_local_devices()

    
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    
        net = lanenet.LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='LaneNet')
    
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    
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


    def img_callback(self, data):
    	global count
    	global num
    	global img_np
    	global img_np2
    	#global original_img
    	try:
    		cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    		count += 1 
    	except CvBridgeError as e:
            print(e)
    	cv_image = cv2.resize(cv_image, (1280,720))
    	original_img = cv_image.copy()
#    	roi_image = cv_image.copy()
#    	roi_image = roi_image[50:256, 0:512]
#    	roi_image = cv2.resize(roi_image, (512, 256), interpolation=cv2.INTER_LINEAR)
#-------------origin image sup-------------
#    	if count > 3 :
#            cv_image = cv2.addWeighted(cv_image, 0.6, img_np, 0.5, 0)
#    	img_np = original_img.copy()
#    	copy_img = img_np.copy()
#    	if count > 2 :
#            img_np = cv2.addWeighted(img_np, 0.6, img_np2, 0.5, 0)
#    	img_np2 = copy_img.copy() 
    	#cv2.imshow("(t-1) + (t-2)", img_np)
    	#cv2.imshow("cv", cv_image)
#---------------resize sup---------------
    	resized_image = self.preprocessing(cv_image)
    	if count > 3 :
            resized_image = cv2.addWeighted(resized_image, 0.6, img_np, 0.5, 0)
    	img_np = resized_image.copy()
    	copy_img = img_np.copy()
    	if count > 2 :
            img_np = cv2.addWeighted(img_np, 0.6, img_np2, 0.5, 0)
    	img_np2 = img_np.copy()
    	img_np2 = copy_img.copy()
#    	cv2.imshow("roi", resized_image)
#---------------------------------------
    	mask_image = self.postprocessing(resized_image, original_img)
#    	mask_image = self.postprocessing(resized_image, roi_image)
#    	if count % 3 == 0:
#            cv2.imwrite("/home/choiin/original_frame/%d.jpg" %num, original_img)
#            cv2.imwrite("/home/choiin/superposition_frame/%d.jpg" %num, resized_image)
#            num = num+1 
#    	image_publisher = self.image_publish(mask_image)
        


        


    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0 #normalization
        # ROI method 1 - black box
        mask = np.zeros(image.shape[:2], dtype = "uint8") 
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.rectangle(mask, (cX - 255, cY - 75), (cX + 255, cY + 127), 255, -1)
        image = cv2.bitwise_and(image, image, mask = mask)
        # ROI method 2 - cut and stretch
#        image = image[50:256, 0:512]
#        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)

#        cv2.imshow("image", image)
        return image

    def postprocessing(self, img, original_img):    #inference_net
        binary_seg_image, instance_seg_image = self.sess.run(
				[self.binary_seg_ret, self.instance_seg_ret],
				feed_dict={self.input_tensor: [img]}
		) # this process 0.04~0.05s
#        binary_seg_image = cv2.resize(binary_seg_image, (512, 206))
#        binary_seg_image = cv2.copyMakeBorder(binary_seg_image, 50, 0, 0 ,0, cv2.BORDER_CONSTANT, value = [0,0,0])
#        instance_seg_image = cv2.resize(instance_seg_image, (512, 206))
#        instance_seg_image = cv2.copyMakeBorder(instance_seg_image, 50, 0, 0 ,0, cv2.BORDER_CONSTANT, value = [0,0,0])
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_img
        ) # 0.8 ~ 1.5 s
       
        mask_image = postprocess_result['mask_image']
        number_line = postprocess_result['line_count']

        #print("line number : ", number_line)
#        mask_image = cv2.resize(mask_image, (512, 206))
#        mask_image = cv2.copyMakeBorder(mask_image, 50, 0, 0 ,0, cv2.BORDER_CONSTANT, value = [0,0,0])
        mask_image = cv2.resize(mask_image, (original_img.shape[1],
                                                original_img.shape[0]),interpolation=cv2.INTER_LINEAR)
#
#        # binary image(2d) to 3D image
#        mask_image = mask_image.reshape([720,1280,1])
#        mask_image2 = mask_image
#        mask_image = np.append(mask_image, mask_image2, axis = 2)
#        mask_image = np.append(mask_image, mask_image2, axis = 2)
#        binary_image = mask_image.copy()
#        if count % 3 == 0 :
#            cv2.imwrite("/home/choiin/superposition_binary_frame/%d.jpg" %num, mask_image)
        mask_image = cv2.addWeighted(original_img, 0.6, mask_image, 0.4, 0) # orgin + mask(lane)
        cv2.waitKey(1)
        #print(mask_image.shape, type(mask_image))
#        dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
#        cv2.imshow("result_img", mask_image) # mask image ~~ -> 0.01~0.02 s
#        return mask_image

#        #print binary image
#        bin_img = np.array(binary_seg_image[0]*255, np.uint8)
#        bin_img[0:20, 0:512] = 0
#        print("bin_img = {0}".format(bin_img.shape))
#        bin_img = cv2.resize(bin_img, (640, 480))
#        cv2.imshow("bin", bin_img)
# image caputre
#		cv2.imshow("bin_img",bin_img) # capture
#        cv2.imwrite("/aveesSSD/cap001"+".png",bin_img)


#        cv2.waitKey(0)

# print instance image
#        inst_img = np.array(instance_seg_image[0], np.uint8)
#        inst_img = np.where(inst_img != 0, inst_img, 150)
#        cv2.imshow("inst_img",inst_img)
#        cv2.imwrite("/aveesSSD/cap01"+".png",inst_img) #capture
#        cv2.waitKey(1)
#        print('postprocessing(), cost time: {:.5f}s'.format(time.time() - t_start))

#        img_msg = self.bridge.cv2_to_imgmsg(bin_img, "mono8")
#        self.pub_image.publish(img_msg)
#        self.pub_image.publish(binary_image)

        return mask_image


    def image_publish(self, img):
        self.pub_image.publish(img)



if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node')
    print('\ninit_node\n')
    lanenet_detector()
    print('\nlanenet_detector()\n')
    rospy.spin()
    print('\nspin()\n')



