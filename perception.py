#!/usr/bin/env python

""" Perception node """
from __future__ import print_function

import time

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import torch
from torch import nn
import torchvision.transforms.functional as TF
from sensor_msgs.msg import Image
from PIL import Image as PIL_img
import tf2_ros 
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion , quaternion_from_euler

from prob21_ms2.msg import Sign
from prob21_ms2.msg import SignArray
from geometry_msgs.msg import TransformStamped, PoseStamped

import math

import utils
from detector import Detector

import rospkg 

rospack = rospkg.RosPack()
path = rospack.get_path('prob21_ms2')

def get_transform_stamped(m,id):


    t = TransformStamped()
    t.header.frame_id = m.header.frame_id
    t.child_frame_id = 'Detected_sign/' + str(id)
    t.header.stamp = m.header.stamp

    t.transform.translation.x = m.pose.position.x
    t.transform.translation.y = m.pose.position.y
    t.transform.translation.z = m.pose.position.z
    
    t.transform.rotation.x=m.pose.orientation.x
    t.transform.rotation.y=m.pose.orientation.y
    t.transform.rotation.z=m.pose.orientation.z
    t.transform.rotation.w=m.pose.orientation.w
    return t
class image_conv:
    def __init__(self,device="cpu", mod_path=path+'/scripts/perception/det_2021-02-20_11-44-05-076696.pt'):
        self.image_pub = rospy.Publisher("/myresult", Image, queue_size=1)
        self.detect_pub = rospy.Publisher("/Sign", Sign, queue_size=1)
        #create network
        self.detector = Detector() #.to(device)
        #load weights
        self.model = utils.load_model(self.detector,mod_path,device)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, self.callback, queue_size=1, buff_size=2**28)
        self.tf_buf   = tf2_ros.Buffer()
        self.tf_lstn  = tf2_ros.TransformListener(self.tf_buf)
        self.broadcaster = tf2_ros.TransformBroadcaster()

    def callback(self, data, device="cpu"):
    # Convert the image from OpenCV to ROS format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
    
    #convert ROS message to compatible tensor
        PIL_image = PIL_img.fromarray(cv_image) #cvimage?
        Tens_image = TF.to_tensor(PIL_image) #PIL_image
        #Tens_to_dev = Tens_image.to(device) 
        #Tens_to_dev = Tens_to_dev.unsqueeze(0)
        Tens_image = Tens_image.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            t = time.time()
            out = self.model(Tens_image) #.cpu() or Tens_to_dev
            elapsed = time.time() - t
            #print("time to process one image", elapsed)
            bbs = self.model.decode_output(out, 0.5)

            for i in range(len(bbs)):
                #print("lenbbs[]",len(bbs[i]))
                if len(bbs[i]) > 0:
                    for bb in bbs[i]:
                        # start point represents the top left corner of rectangle x,y
                        start_point = (bb["x"], bb["y"])
                        # end point represents the bottom right corner of rectangle
                        end_point = (bb["x"] + bb["width"], bb["y"] + bb["height"])
                        category = bb["category"] #for now returns 0, will give the id of the sign
                        # color
                        color = (0,0,255)
                        # thickness
                        thickness = 2
                        image = cv2.rectangle(cv_image, start_point, end_point, color, thickness)
                        cv2.putText(image,'sign', (bb["x"], bb["y"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                        try:
                            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
                        except CvBridgeError as e:
                            print(e)
                        #distance to object
                        fx = 231.250001 #focal length
                        real_width = 0.19
                        bb_width = bb["width"]
                        x = bb["x"]
                        y = bb["y"]
                        Z = real_width * fx / bb_width #in m
                        X= Z*(x-320.519378)/fx
                        Y= Z*(y-240.631482)/fx

                        rospy.loginfo("Sign at ({},{},{}) m in the image frame".format(X,Y,Z))

                        P=PoseStamped()
                        P.header.frame_id="cf1/camera_link"
                        P.header.stamp = data.header.stamp
                        (P.pose.position.x, P.pose.position.y, P.pose.position.z)=(X,Y,Z)
                        (P.pose.orientation.x, P.pose.orientation.y, P.pose.orientation.z, P.pose.orientation.w) = (quaternion_from_euler(math.radians(-90), math.radians(0), math.radians(-90)))
                                                                            #"map"
                        if not self.tf_buf.can_transform(P.header.frame_id, P.header.frame_id ,  P.header.stamp , rospy.Duration(0.5)):
                            rospy.logwarn_throttle(5.0, 'No transform from %s to map' % P.header.frame_id)
                            print("no transform")
                            return None

                        #transform = self.tf_buf.lookup_transform( "map", P.header.frame_id, P.header.stamp , rospy.Duration(0.5))

                        #transformed_maker=tf2_geometry_msgs.do_transform_pose(P, transform) 

                        transformed_maker = self.tf_buf.transform(P,P.header.frame_id) #map

                        transform_stamped = get_transform_stamped(transformed_maker,category)
                        self.broadcaster.sendTransform(transform_stamped)


                        #########
                        #publish msg in camera frame.... maybe change to map frame..
                        det = Sign()
                        det.header = data.header
                        det.header.frame_id = "cf1/camera_link"
                        det.pose.pose.position.x = X
                        det.pose.pose.position.y = Y
                        det.pose.pose.position.z = Z
                        self.detect_pub.publish(det)

                        ################

                        #if not self.tf_buf.can_transform(det.header.frame_id,"map",det.header.stamp):


                else:
                    try:
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8")) #cv_image
                    except CvBridgeError as e:
                        print(e)


def main(args):
    rospy.init_node('perception', anonymous=True)

    ic = image_conv()

    print("running...")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


