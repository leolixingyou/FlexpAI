import rospy
import time
import cv2

class Control_Slave:
    def __init__(self) -> None:
        rospy.init_node('Control_Slave', anonymous=True)
        self.rate = rospy.Rate(0.5) # 10hz

    def close_control(self,):
        print(f"Closing control")

    def run_control(self,):
        while not rospy.is_shutdown():
            # print(f"I am control")
            self.rate.sleep()
        self.close_control()

if __name__ =='__main__':

    # control_part = Control_Slave()
    # control_part.run_control()

    print('control')
