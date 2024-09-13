import rospy
import time
import cv2
import subprocess
import platform  # for get os info

os_name = platform.platform().split('-')[0]
if os_name == 'Linux':
    import getch as getch_os  # for getch()
elif os_name == 'Windows':
    import msvcrt as getch_os  # for getch()

class Master_Server:
    def __init__(self) -> None:
        rospy.init_node('Master_Server', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz
        self.processes = {}

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
            if 'sensing' not in self.processes or not self.processes['sensing'].poll() is None:
                self.processes['sensing'] = subprocess.Popen(['python', 'sensing.py'])
        elif keyboard_input == 'z':
            if 'sensing' in self.processes:
                self.processes['sensing'].terminate()

        if keyboard_input == 's':
            if 'perception' not in self.processes or not self.processes['perception'].poll() is None:
                self.processes['perception'] = subprocess.Popen(['python', 'perception.py'])
        elif keyboard_input == 'x':
            if 'perception' in self.processes:
                self.processes['perception'].terminate()

        if keyboard_input == 'd':
            if 'planning' not in self.processes or not self.processes['planning'].poll() is None:
                self.processes['planning'] = subprocess.Popen(['python', 'planning.py'])
        elif keyboard_input == 'c':
            if 'planning' in self.processes:
                self.processes['planning'].terminate()

        if keyboard_input == 'f':
            if 'control' not in self.processes or not self.processes['control'].poll() is None:
                self.processes['control'] = subprocess.Popen(['python', 'control.py'])
        elif keyboard_input == 'v':
            if 'control' in self.processes:
                self.processes['control'].terminate()


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
