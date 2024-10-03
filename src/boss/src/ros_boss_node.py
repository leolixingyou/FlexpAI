#!/usr/bin/env python3
import signal
import curses
import platform  # for get os info

import rospy
import rosnode
import rosgraph
from std_msgs.msg import String

from informantion_observing import Observing_Server
from launch_generator import Node_Launcher_Generator
from utils import DestroyMan, launch_id_generator, \
    launcher_start, my_kill_nodes, rosnodes_my_cleanup

os_name = platform.platform().split('-')[0]
if os_name == 'Linux':
    import getch as getch_os  # for getch()
elif os_name == 'Windows':
    import msvcrt as getch_os  # for getch()

class Boss_Lancher_Manager(DestroyMan):
    def __init__(self) -> None:
        super().__init__()
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
        self.nodes_properties = {
        '/Control_Server' : 
        {'package':'control','node_type':'control.py','name':'Control_Server',
        'required':False,'output':'screen'},
        
        '/Sensing_Server' : 
        {'package':'sensing','node_type':'sensing.py','name':'Sensing_Server',
        'required':False,'output':'screen'},

        '/Perception_Server' : 
        {'package':'perception','node_type':'perception.py','name':'Perception_Server',
        'required':False,'output':'screen'},

        '/Planning_Server' : 
        {'package':'planning','node_type':'planning.py','name':'Planning_Server',
        'required':False,'output':'screen'},
        }

        print('Please Choose mode for Launch Separate: choose a runall: choose s')
        
        key_mode_map = {'a':0,'s':1}
        key_mode = getch_os.getch()
        if key_mode in key_mode_map:
            self.mode = ['separate', 'runall'][key_mode_map[key_mode]]

        self.modify_node_launcher_input(self.nodes_properties)

        
        ## curses
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.nodelay(True)  # set to unblocking mode: True

        ### No Needs for same name with launch the using name defined by launcher
        rospy.init_node('Boss_Server', anonymous=False)
        self.rate = rospy.Rate(10) # 10hz
        self.nodes = []
        self.stopped_nodes = {}
    
    def __del__(self):
        # destory instance and recover terminal settings
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()

    def modify_node_launcher_input(self, nodes_properties):
        if self.mode == 'separate':
            self.node_launchers = {x:Node_Launcher_Generator({x:y}) for x,y in nodes_properties.items()}
        if self.mode == 'runall':
            self.node_launchers = {'runall':Node_Launcher_Generator(nodes_properties)}

    def modify_stopped_launcher(self, nodes_properties):
        if self.mode == 'separate':
            for key, value in nodes_properties.items():
                self.node_launchers[key] = Node_Launcher_Generator({key:value})
        if self.mode == 'runall':
            self.node_launchers = {'runall':Node_Launcher_Generator(self.nodes_properties)}

    # from DestroyMan
    def process_cleanup(self):
        print("Finally: Performing cleanup")
        nodes = Observing_Server.update_nodes_info()
        rosnode.kill_nodes(nodes)
        rosnodes_my_cleanup()
        self.__del__()
        print('Finally: Process End Safely')

    def run(self):
        keys_on = ['a', 's', 'd', 'f', 'g'] #5 launcher on are given
        keys_off = ['z', 'x', 'c', 'v', 'b'] #5 launcher off are given
        keymap_on = {ord(keys_on[i]):x for i, x in enumerate(self.node_launchers)}
        keymap_off = {ord(keys_off[i]):x for i, x in enumerate(self.node_launchers)}

        while not rospy.is_shutdown():
            self.nodes = Observing_Server.update_nodes_info()
            # key_in = getch_os.getch()
            key_in = self.screen.getch()
            
            if key_in == -1:  # no input
                rospy.sleep(0.1) 
                continue

            # launch generator
            if key_in in keymap_on:
                node_name_idx = keymap_on[key_in]
                if node_name_idx not in self.stopped_nodes.keys():
                    launcher = self.node_launchers[node_name_idx]
                    ### to run process which is not running
                    for node_name in list(launcher.node_names):
                        if node_name not in self.nodes:
                            launcher.launch_nodes()

                else:
                    ### to restart stopped process
                    self.modify_stopped_launcher(self.stopped_nodes)
                    launcher = self.node_launchers[keymap_on[key_in]]
                    if self.mode == 'runall':
                        launcher.launch_nodes()
                        self.stopped_nodes={}
                    else:
                        for node_name in list(launcher.node_names):
                            if node_name not in self.nodes:
                                launcher.launch_nodes()
                                self.stopped_nodes.pop(node_name)
                            
            elif key_in in keymap_off:
                delet_list = []
                node_name_idx = keymap_off[key_in]
                if node_name_idx not in list(self.stopped_nodes.keys()) and node_name_idx != 'runall':
                    launcher = self.node_launchers[node_name_idx]
                    for node_name in list(launcher.node_names):
                        self.stopped_nodes[node_name] = self.nodes_properties[node_name]
                        self.node_launchers.pop(node_name)
                        delet_list.append(node_name)

                elif node_name_idx == 'runall':
                    launcher = self.node_launchers[node_name_idx]
                    self.stopped_nodes[node_name_idx] = self.nodes_properties
                    self.node_launchers={}
                    for node_name in list(launcher.node_names):
                        delet_list.append(node_name)
                my_kill_nodes(delet_list)

            self.rate.sleep()
            print(f'Nodes is {self.nodes}')
        self.process_cleanup()

if __name__ == "__main__":
    test_temp = Boss_Lancher_Manager()
    test_temp.run()
        