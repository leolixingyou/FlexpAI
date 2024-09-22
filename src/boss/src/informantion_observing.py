#!/usr/bin/env python3
import rosnode
import roslaunch

from utils import *

class Observing_Server:
    def __init__(self) -> None:
        pass
    
    def update_nodes_info():
        return [x for x in rosnode.get_node_names() if x != '/rosout']

    def run(self, input_top):
        packages = self.rp.list()
        py = self.rp.get_path('rosnode')
        print(py)


if __name__ == "__main__":
    launch_file_list = ["/workspace/src/launch/control_launch.launch"]
    ## rosnode kill "/topic name" to stopi
    test_launch_f(launch_file_list)
    
    # ros_la = roslaunch.scriptapi.ROSLaunch()
    # ros_la.load(launch_file_list[0])
    # ros_la.launch()

    # test_temp = Observing_Server()
    # test_temp.run()

