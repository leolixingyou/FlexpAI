import rospy
import time
import cv2
import platform  # for getch()

from sensing import run_sensing, close_sensing
from perception import run_perception, close_perception
from planning import run_planning, close_planning
from control import run_control, close_control

os_name = platform.platform().split('-')[0]
if os_name == 'Linux':
    import getch as getch_os  # for getch()
elif os_name == 'Windows':
    import msvcrt as getch_os  # for getch()

class Master_Server:
    def __init__(self) -> None:
        rospy.init_node('Master_Server', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz

    def master_manager(self, keyboard_input):
        print(keyboard_input)
        """
        keyboard_input:
        "q" is exit
        "a" is sensing
        "s" is perception
        "d" is planning
        "f" is control
        """
        if keyboard_input == 'a':
            run_sensing()
        elif keyboard_input == 'z':
            close_sensing()

        if keyboard_input == 's':
            run_perception()
        elif keyboard_input == 'x':
            close_perception()

        if keyboard_input == 'd':
            run_planning()
        elif keyboard_input == 'c':
            close_planning()

        if keyboard_input == 'f':
            run_control()
        elif keyboard_input == 'v':
            close_control()

    def run_master(self):
        while not rospy.is_shutdown() :
            keyboard_input = getch_os.getch()
            if keyboard_input != 'q':
                self.master_manager(keyboard_input)
            else:
                break
            self.rate.sleep()

if __name__ =='__main__':

    master_server = Master_Server()
    master_server.run_master()
