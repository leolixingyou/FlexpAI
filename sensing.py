import rospy
import time
import cv2

def run_sensing():
    print(f"I am sensing")

def close_sensing():
    print(f"Closing sensing")

if __name__ =='__main__':
    # main()
    while not rospy.is_shutdown():
        run_sensing()
    # rospy.spin()


