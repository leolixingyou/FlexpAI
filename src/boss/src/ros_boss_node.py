#!/usr/bin/env python3
import os
import sys
import cv2
import signal
import platform  # for get os info

import rospy
import rosnode
import rosgraph
from std_msgs.msg import String

from informantion_observing import Observing_Server, \
launch_generator, launcher_start, my_kill_nodes

os_name = platform.platform().split('-')[0]
if os_name == 'Linux':
    import getch as getch_os  # for getch()
elif os_name == 'Windows':
    import msvcrt as getch_os  # for getch()


def rosnodes_my_cleanup():
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

def process_cleanup():
    print("Finally: Performing cleanup")
    nodes = Observing_Server.update_nodes_info()
    rosnode.kill_nodes(nodes)
    rosnodes_my_cleanup()
    print('Finally: Process End Safely')

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    process_cleanup()
    sys.exit(0)


class Boss_Lancher_Manager:
    def __init__(self) -> None:
        ### TODO: need to use them with dataclass

        # hrkim
        self.node_launchpath_dict = { # OrderedDict 대신 dict 사용 (python >= 3.6)
            '/Control_Server':["/workspace/src/launch/control_launch.launch"],
            '/Perception_Server':["/workspace/src/launch/perception_launch.launch"],
            '/Planning_Server':["/workspace/src/launch/planning_launch.launch"],
            '/Sensing_Server':["/workspace/src/launch/sensing_launch.launch"]}
        self.sub_launchers = [launch_generator(v) for v in self.node_launchpath_dict.values()] # lambda 마려움
        
        ### No Needs for same name with launch the using name defined by launcher
        rospy.init_node('Boss_Server', anonymous=False)
        self.rate = rospy.Rate(10) # 10hz
        self.nodes = []

    def run(self):
        keymap_on = {'a':0,'s':1,'d':2,'f':3}
        keymap_off = {'z':0,'x':1,'c':2,'v':3}
        self.boss_state = 1
        while not rospy.is_shutdown() and self.boss_state == 1:
            print()
            self.nodes = Observing_Server.update_nodes_info()
            key_in = getch_os.getch()

            # hrkim
            if key_in in keymap_on:
                node_name = list(self.node_launchpath_dict)[keymap_on[key_in]]
                if node_name not in self.nodes:
                    launcher_start(self.sub_launchers[keymap_on[key_in]])
            elif key_in in keymap_off:
                node_name = list(self.node_launchpath_dict)[keymap_off[key_in]]
                my_kill_nodes(node_name)

            self.rate.sleep()

            print(f'Nodes is {self.nodes}')

        self.process_cleanup()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    test_temp = Boss_Lancher_Manager()
    test_temp.run()


        
        