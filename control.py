import rospy
import time
import cv2

def run_control():
    print(f"I am Control")

def close_control():
    print(f"Closing Control")

if __name__ =='__main__':
    # main()
    while not rospy.is_shutdown():
        run_control()
    # rospy.spin()


