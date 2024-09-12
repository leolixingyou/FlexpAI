import rospy
import time
import cv2

def run_perception():
    print(f"I am Perception")

def close_perception():
    print(f"Closing perception")
    
if __name__ =='__main__':
    while not rospy.is_shutdown():
        run_perception()
    # rospy.spin()


