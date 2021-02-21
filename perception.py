#!/usr/bin/env python

""" Perception node"""
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch
from torch import nn
import torchvision.transforms.functional as TF
from sensor_msgs.msg import Image
from PIL import Image as PIL_img


import utils
from detector import Detector
class image_conv:
    def __init__(self):
        image_pub = rospy.Publisher("/myresult", Image, queue_size=2)
        image_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, callback)

    def callback(self,img):

        network("cpu", "/home/chigio/dd2419_detector_baseline/det_2021-02-20_11-44-05-076696.pt",img )
        try:
            self.image_pub.publish(img)
        except CvBridgeError as e:
            print(e)



        


#create network
def network(device="cpu", mod_path="/home/chigio/dd2419_detector_baseline/det_2021-02-20_11-44-05-076696.pt", image):
    #create network
    detector = Detector().to(device)
    #load weights
    utils.load_model(detector,mod_path,device)
    #convert ROS message to compatible tensor
    




