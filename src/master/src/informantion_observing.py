#!/usr/bin/env python3
from rospkg import RosPack
import roslaunch
import rosnode

start_l= []
def launch_f(launch_file_list):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, launch_file_list)
    start_l.append(launch.start(False))
    launch.spin_once()
    # try:
    #     launch.spin()
    # finally:
    #     launch.shutdown()

def test_launch_f(launch_file_list):
    launch_f(launch_file_list)
    # rosnode.kill_nodes(launch_file_list)

class Observing_Server:
    def __init__(self) -> None:
        self.rp = RosPack()

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

