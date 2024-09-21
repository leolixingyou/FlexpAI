import roslaunch
import rospy
import signal
import sys

class NodeLauncher:
    def __init__(self):
        rospy.init_node('launch_node', anonymous=False)
        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()
        self.processes = []

    def launch_nodes(self):
        # 配置重要节点
        node1 = roslaunch.core.Node(
            package='boss',
            node_type='ros_boss_node.py',
            name='Boss_Sever_1',
            required=True
        )
        # 配置可选节点
        node2 = roslaunch.core.Node(
            package='your_package',
            node_type='optional_node',
            name='optional_node'
        )
        # 配置自动重启的节点
        node3 = roslaunch.core.Node(
            package='your_package',
            node_type='another_node',
            name='another_node',
            respawn=True,
            respawn_delay=5
        )

        # 启动节点
        self.processes.append(self.launch.launch(node1))
        # self.processes.append(self.launch.launch(node2))
        # self.processes.append(self.launch.launch(node3))

    def shutdown(self):
        print("Shutting down all nodes...")
        for process in self.processes:
            if process.is_alive():
                process.stop()
        self.launch.stop()
        print("All nodes have been stopped.")

def signal_handler(signum, frame):
    print("Received shutdown signal. Stopping all nodes...")
    launcher.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    launcher = NodeLauncher()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        launcher.launch_nodes()
        print("All nodes have been launched. Press Ctrl+C to stop.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        launcher.shutdown()