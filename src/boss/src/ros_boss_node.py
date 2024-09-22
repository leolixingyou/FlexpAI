#!/usr/bin/env python3
import sys
import signal
import platform  # for get os info

import rospy
import rosnode
import rosgraph
from std_msgs.msg import String

from informantion_observing import Observing_Server, \
launch_id_generator, launcher_start, my_kill_nodes
from launch_generator import Node_Launcher_Generator

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
    for launcher in test_temp.node_launchers:
        launcher.shutdown()
    process_cleanup()
    sys.exit(0)


class Boss_Lancher_Manager:
    def __init__(self) -> None:
        # process_cleanup()
        ### TODO: need to use them with dataclass

        # launching 1
        # hrkim
        # self.node_launchpath_dict = { # OrderedDict 대신 dict 사용 (python >= 3.6)
        #     '/Control_Server':["/workspace/self-driving_framework_kml/src/launch/control_launch.launch"],
        #     '/Perception_Server':["/workspace/self-driving_framework_kml/src/launch/perception_launch.launch"],
        #     '/Planning_Server':["/workspace/self-driving_framework_kml/src/launch/planning_launch.launch"],
        #     '/Sensing_Server':["/workspace/self-driving_framework_kml/src/launch/sensing_launch.launch"]}
        # self.sub_launchers = [launch_id_generator(v) for v in self.node_launchpath_dict.values()] # lambda 마려움

        # # launching 2
        nodes_properties = {
        'Control_Server' : 
        {'package':'control','node_type':'control.py','name':'Control_Server',
        'required':False,'output':'screen'},
        
        'Sensing_Server' : 
        {'package':'sensing','node_type':'sensing.py','name':'Sensing_Server',
        'required':False,'output':'screen'},
        

        'Perception_Server' : 
        {'package':'perception','node_type':'perception.py','name':'Perception_Server',
        'required':False,'output':'screen'},
        

        'Planning_Server' : 
        {'package':'planning','node_type':'planning.py','name':'Planning_Server',
        'required':False,'output':'screen'},
        }

        print('Please Choose mode for Launch \n \
        \'Separate\': choose \'a\' \n \
        \'runall\': choose \'s\' \
        ')
        
        key_mode_map = {'a':0,'s':1}
        key_mode = getch_os.getch()
        mode = ['separate', 'runall'][key_mode_map[key_mode]]
        self.modify_node_launcher_input(nodes_properties, mode)

        ### No Needs for same name with launch the using name defined by launcher
        rospy.init_node('Boss_Server', anonymous=False)
        self.rate = rospy.Rate(10) # 10hz
        self.nodes = []
    
    def modify_node_launcher_input(self, nodes_properties, mode ='separate'):
        if mode == 'separate':
            self.node_launchers =[Node_Launcher_Generator({x:y}) for x,y in nodes_properties.items()]
        if mode == 'runall':
            self.node_launchers =[Node_Launcher_Generator(nodes_properties)]

    def run(self):
        keymap_on = {'a':0,'s':1,'d':2,'f':3}
        keymap_off = {'z':0,'x':1,'c':2,'v':3}

        keys_on = ['a', 's', 'd', 'f', 'g'] #5 launcher on are given
        keys_off = ['z', 'x', 'c', 'v', 'b'] #5 launcher off are given
        keymap_on = {keys_on[x]:x for x in range(len(self.node_launchers))}
        keymap_off = {keys_off[x]:x for x in range(len(self.node_launchers))}

        self.boss_state = 1
        while not rospy.is_shutdown() and self.boss_state == 1:
            print()
            self.nodes = Observing_Server.update_nodes_info()
            key_in = getch_os.getch()

            # # hrkim -> using launch file
            # if key_in in keymap_on:
            #     node_name = list(self.node_launchpath_dict)[keymap_on[key_in]]
            #     if node_name not in self.nodes:
            #         launcher_start(self.sub_launchers[keymap_on[key_in]])
            # elif key_in in keymap_off:
            #     node_name = list(self.node_launchpath_dict)[keymap_off[key_in]]
            #     my_kill_nodes(node_name)

            # launch generator
            if key_in in keymap_on:
                self.node_launchers[keymap_on[key_in]].launch_nodes()
            elif key_in in keymap_off:
                lancher = self.node_launchers[keymap_off[key_in]]
                node_name = list(lancher.node_names)
                lancher.shutdown()
                my_kill_nodes(node_name)

            self.rate.sleep()
            print(f'Nodes is {self.nodes}')

        process_cleanup()


if __name__ == "__main__":
    test_temp = Boss_Lancher_Manager()
    signal.signal(signal.SIGINT, signal_handler)
    test_temp.run()
        