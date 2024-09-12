import rospy
import time
import cv2

def run_planning():
    print(f"I am planning")

def close_planning():
    print(f"Closing planning")

if __name__ =='__main__':
    # main()
    while not rospy.is_shutdown():
        run_planning()
    # rospy.spin()


