import rospy
import time
import cv2

class Sensing_Slave:
    def __init__(self) -> None:
        rospy.init_node('Sensing_Slave', anonymous=True)
        self.rate = rospy.Rate(1) # 10hz

    def close_sensing(self,):
        print(f"Closing sensing")

    def run_sensing(self,):
        while not rospy.is_shutdown():
            print(f"I am sensing")
            self.rate.sleep()
        self.close_sensing()


if __name__ =='__main__':

    sensing_part = Sensing_Slave()
    sensing_part.run_sensing()

