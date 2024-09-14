import rospy
import time
import cv2
import rospy
import time
import cv2

class Perception_Slave:
    def __init__(self) -> None:
        rospy.init_node('Perception_Slave', anonymous=True)
        self.rate = rospy.Rate(0.5) # 10hz

    def close_perception(self,):
        print(f"Closing perception")

    def run_perception(self,):
        while not rospy.is_shutdown():
            # print(f"I am perception")
            self.rate.sleep()
        self.close_perception()


if __name__ =='__main__':

    perception_part = Perception_Slave()
    perception_part.run_perception()

