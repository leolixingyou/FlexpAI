import roslaunch
import rospy
import signal
import sys

class Node_Launcher_Generator:
    def __init__(self, nodes_properties_dict):
        self.processes = []
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()
        self.nodes_stand_by_dict = self.launch_node_setting(nodes_properties_dict)
        self.node_names = nodes_properties_dict.keys()

    def nodes_properties_setting(self,node_properties):
        # Node setting
        node = roslaunch.core.Node(
            package = node_properties['package'],
            node_type = node_properties['node_type'],
            name = node_properties['name'],
            required = node_properties['required'],
            output = node_properties['output']
        )
        return node

    def launch_node(self, node):
        launch_process = self.launcher.launch(node)
        return launch_process

    def launch_nodes(self,):
        for node in self.nodes_stand_by_dict.values():
            self.processes.append(self.launch_node(node))

    def launch_node_setting(self, nodes_properties_dict) -> dict:
        """
        @param package: node package name
        @type  package: str

        @param node_type: node type
        @type  node_type: str
        
        @param name: node name
        @type  name: str
        
        @param namespace: namespace for node
        @type  namespace: str
        
        @param machine_name: name of machine to run node on
        @type  machine_name: str
        
        @param args: argument string to pass to node executable
        @type  args: str
        
        @param respawn: if True, respawn node if it dies
        @type  respawn: bool
        
        @param remap_args: list of [(from, to)] remapping arguments
        @type  remap_args: [(str, str)]: 
        
        @param env_args: list of [(key, value)] of
        additional environment vars to set for node
        @type  env_args: [(str, str)]
        
        @param output: where to log output to, either Node, 'screen' or 'log'
        @type  output: str
        
        @param cwd: current working directory of node, either 'node', 'ROS_HOME' or 'ros-root'. Default: ROS_HOME
        @type  cwd: str
        
        @param launch_prefix: launch command/arguments to prepend to node executable arguments
        @type  launch_prefix: str
        
        @param required: node is required to stay running (launch fails if node dies)
        @type  required: bool
        
        @param filename: name of file Node was parsed from
        @type  filename: str

        @raise ValueError: if parameters do not validate
        """        

        return {x:self.nodes_properties_setting(y) for x,y in nodes_properties_dict.items()}

    def shutdown(self):
        print("Shutting down all nodes...")
        for process in self.processes:
            if process.is_alive():
                process.stop()
        self.launcher.stop()
        print("All nodes have been stopped.")

    def run(self):
        try:
            self.launch_nodes()
            print("All nodes have been launched. Press Ctrl+C to stop.")
            rospy.spin()
        finally:
            self.shutdown()

def signal_handler(signum, frame):
    print("Received shutdown signal. Stopping all nodes...")
    launcher.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('Node_Launcher_Generator', anonymous=False)
    nodes_properties = {

        # 'Boss_Server' : 
        # {'package':'boss','node_type':'ros_boss_node.py','name':'Boss_Server',
        # 'required':False,'output':'screen'},

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
    
    
    launcher = Node_Launcher_Generator(nodes_properties)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    launcher.run()
