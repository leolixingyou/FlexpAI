#!/usr/bin/env python3
import os
import sys
import cv2
import rospy
import rosnode
import rosgraph
import platform  # for get os info
from std_msgs.msg import String
from informantion_observing import Observing_Server, \
launch_generator, launcher_start

os_name = platform.platform().split('-')[0]
if os_name == 'Linux':
    import getch as getch_os  # for getch()
elif os_name == 'Windows':
    import msvcrt as getch_os  # for getch()



def test_clean():
    """
    This is a semi-hidden routine for cleaning up stale node
    registration information on the ROS Master. The intent is to
    remove this method once Master TTLs are properly implemented.
    """
    ID = '/rosnode'
    pinged, unpinged = rosnode.rosnode_ping_all()
    if unpinged:
        master = rosgraph.Master(ID)
        rosnode.cleanup_master_blacklist(master, unpinged)

        print("done")

class Boss_Lancher_Manager:
    def __init__(self) -> None:
        #
        ### No Needs for same name with launch the using name defined by launcher
        self.boss_package = '/Boss_Server'
        self.control_package = '/Control_Server'
        self.perception_package = '/Perception_Server' 
        self.planning_package = '/Planning_Server' 
        self.sensing_package = '/Sensing_Server'
        
        self.control_launch = [
            "/workspace/src/launch/control_launch.launch"]
        self.perception_launch = [
            "/workspace/src/launch/perception_launch.launch"]
        self.planning_launch = [
            "/workspace/src/launch/planning_launch.launch"]
        self.sensing_launch = [
            "/workspace/src/launch/sensing_launch.launch"]
        
        ### 0 is off, 1 is on
        self.control_state = 0
        self.perception_state = 0
        self.sensing_state = 0
        self.planning_state = 0
        self.boss_state = 0

        self.sub_package = ['/Control_Server']
        self.sub_launch = [
            "/workspace/src/launch/control_launch.launch"]
        self.sub_launcher = launch_generator(self.sub_launch)
        
        self.sub_state = 0
        self.sub_require_state = 0
        self.process_state = 0
        self.key = cv2.waitKey()

        rospy.init_node('Boss_Server', anonymous=True)
        self.rate = rospy.Rate(10) # 10hz
        self.observation = Observing_Server()
        self.nodes = self.observation.update_nodes_info()
        self.keyboard_input = 'l'

    def active_n_mode(self,):
        if self.keyboard_input == 'a':
            self.sub_require_state = 1
        if self.keyboard_input == 'z':
            self.sub_require_state = 0


    def observe_node(self):
        self.nodes = self.observation.update_nodes_info()
        if self.sub_package[0] in self.nodes:
            self.sub_state = 1
        else:
            self.sub_state = 0 

    def switch_process_state(self):
        if self.sub_state != self.sub_require_state:
            self.process_state = 1
        else:
            self.process_state = 0

    def execute_and_release(self,):
        if self.sub_require_state == 1:
            print('starting Launcher')
            launcher_start(self.sub_launcher)
        elif self.sub_require_state == 0:
            ### input should be list [package1, package2, ...] ## a is given and will give z and q
            print('Kill Nodes')
            rosnode.kill_nodes(self.sub_package)

    def run(self):
        self.boss_state = 1
        while not rospy.is_shutdown() and self.boss_state == 1:
            print()
            self.keyboard_input = getch_os.getch()

            if self.keyboard_input == 'q':
                break
            self.active_n_mode() ### keyboard to activate or deactivate packages
            self.observe_node() ### observing node and change state
            self.switch_process_state() ### judge the state goes process or not

            if self.process_state == 1:
                self.execute_and_release()
                self.process_state = 0
            else:
                print('No Process')

            self.rate.sleep()

            print(f'Sub state is {self.sub_state}')
            print(f'Sub require is {self.sub_require_state}')
            print(f'Sub process_state is {self.process_state}')
            print(f'Nodes is {self.nodes}')
            self.keyboard_input = 'l'
        print('Killing Nodes')
        rosnode.kill_nodes(self.nodes)
        test_clean()
        print('Process End Safely')
        
if __name__ == "__main__":
    test_temp = Boss_Lancher_Manager()
    test_temp.run()


        
        