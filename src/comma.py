import numpy as np 
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry

eon_dcam_intrinsics = np.array([
  [860,   0,   1152//2],
  [  0,  860,  864//2],
  [  0,    0,     1]])

INPUTS_NEEDED = 100

class CenterTracker:

  def __init__(self):
    rospy.init("Tracker_Node")
    cam_topic = rospy.get_param("camera_topic")
    odom_topic = rospy.get_param("odom_topic")
    camera_sub = rospy.Subscriber(camera_topic,Image, self.cameraCallback)
    odom_sub = rospy.Subscriber(odom_topic,Odometry,self.odomCallback)
    self.bridge = CvBridge()
    self.vps =[]

  def cameraCallback(self,image):
    self.image = self.bridge.imgmsg_to_cv2(image)


  def odomCallback(self,data):
    twist = data.twist.twist.linear
    vp_new = eon_dcam_intrinsics.dot(twist)
    vp_new = vp[:2]/vp[2]
    self.vps.append(vp)
    self.vp = np.mean(self.vps[-INPUTS_NEEDED:],axis=0)

  def drawVp(self):
    cols,rows,ch = self.image.shape
    cv2.rectangle(self.image,(int(rows/4),int(cols/4)),(int(3/4*rows), (int(3/4*cols))))
    cv2.circle(self.image,tuple(self.vp),5,(170,200,255), -11)
    cv2.imshow("Image", self.image)



if __name__=="__main__":
  ct = CenterTracker()
  rospy.spin()
