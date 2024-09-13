import rospy
import time
import cv2
import rospy
import time
import cv2

class Planning_Slave:
    def __init__(self) -> None:
        rospy.init_node('Planning_Slave', anonymous=True)
        self.rate = rospy.Rate(1) # 10hz

    def close_planning(self,):
        print(f"Closing planning")

    def run_planning(self,):
        while not rospy.is_shutdown():
            print(f"I am planning")
            self.rate.sleep()
        self.close_planning()


if __name__ =='__main__':

    planning_part = Planning_Slave()
    planning_part.run_planning()

