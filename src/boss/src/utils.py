import sys
import signal
import rosnode
from abc import ABC, abstractmethod

class DestroyMan(ABC):
    def __init__(self):
        # register signal processer
        # for Ctrl + C
        signal.signal(signal.SIGINT, self.signal_handler)
        # for kill the process
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        self.process_cleanup()
        sys.exit(0)

    @abstractmethod
    def process_cleanup(self):
        pass  # develope later

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

