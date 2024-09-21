#!/usr/bin/env python3
import os
import signal

import rosnode
import roslaunch

def launch_f(launch_file_list):
    """    
    TODO ISSUE the launch shutdowned by itself with launch.start(False) 
    which should run automaticlly on background
    -----    
    No more development
    """

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, launch_file_list)
    # start_l.append(launch.start(False))
    launch.start(True)
    # launch.spin_once()
    # try:
    #     launch.spin()
    # finally:
    #     launch.shutdown()

def launch_id_generator(launch_file_list):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    return roslaunch.parent.ROSLaunchParent(uuid, launch_file_list)

def launcher_start(launcher):
    """
    @param auto_terminate: stop process monitor once there are no
    more processes to monitor (default True). This defaults to
    True, which is the command-line behavior of roslaunch. Scripts
    may wish to set this to False if they wish to keep the
    roslauch infrastructure up regardless of processes being
    monitored.
    """
    launcher.start(False) 

def my_kill_nodes(nodes):
    if type(nodes) != list:
        nodes = [nodes]
    return rosnode.kill_nodes(nodes)

def test_launch_f(launch_file_list):
    launch_f(launch_file_list)
    # rosnode.kill_nodes(launch_file_list)


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

