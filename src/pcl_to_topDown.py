import rospy
from sensor_msgs.msg import Image, PointCloud, CameraInfo
import rosparam 
import numba


class PclToImage:
    """
        A class to convert PCL to a top down image view. 
        Iterates through the points and zeros out the z coordinate. Output to an image
    """

    def __init__(self):

        pclTopic = rosparam.get_param("pcl_topic")
        imageTopic = rosparam.get_param("image_topic")
        camInfo = rosparam.get_param("camera_info")
        self.publish = ""
        self.img = Image()


        self.listener(pclTopic,imageTopic,camInfo)


    def camCallback(self,data):

        self.img.height = data.height
        self.img.width = data.width
        self.img.encoding = "rgb8"
        self.img.is_bigendian = False
        self.img.step = 3*data.width
        self.img.data = np.zeros(data.height,data.width,3)
        self.CamSub.shutdown()


    @numba.jit(nopython=True)
    def PclCallback(self,pcl):
        for i ,val  in enumerate(pcl.channels):
            self.img.data[pcl.points[i].x,pcl.points[i].y,:] = val
        self.pub.publish(self.img)


    def listener(self,pclTopic,imageTopic,camInfo):
        sub = rospy.Subscriber(pclTopic,PointCloud,self.PclCallback)
        self.CamSub = rospy.Subscriber(camInfo,CameraInfo,self.camCallback)
        self.pub = rospy.Publisher(imageTopic,Image,queue_size = 10)





