#!/usr/bin/env python3
import rospy
import time
import cv2
from global_path_subscriber import EnhancedGlobalPathSubscriber

class Control_Lancher_Manager:
    def __init__(self) -> None:
        rospy.init_node('Control_Server', anonymous=False)
        self.global_path_cl = EnhancedGlobalPathSubscriber()
        self.control_init = False
        self.global_path = []


    def check_reset_signal(self):
        """
        TODO 
        Implement your reset signal check here
        For example, you could subscribe to a reset topic or check a parameter
        Return True if reset is requested, False otherwise
        
        """
        return False
    
    def control_setting_init(self,):
        if self.global_path_cl.path_changed == 0:
            self.global_path = self.global_path_cl.get_waypoints()
            self.control_init = True
        return self.global_path

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        temp_flag = False
        while not rospy.is_shutdown():
            print(self.global_path_cl.get_waypoints())
            if not self.control_init:
                global_path = self.control_setting_init()
            elif self.control_init and not temp_flag:
                print('start-mpc')
                self.mpc.run(global_path)
                temp_flag = True
            else:
                print('waiting')
            

if __name__ == "__main__":
    test_temp = Control_Lancher_Manager()
    test_temp.run()