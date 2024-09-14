import os
import signal
import subprocess
import time
import psutil
import rospy
import rosnode

class LaunchManager:
    def __init__(self, launch_files):
        self.launch_files = launch_files
        self.processes = {}

    def check_launch_file(self, file_path):
        return os.path.exists(file_path)

    def start(self, file_name):
        if file_name not in self.launch_files:
            print(f"No such file in named: {file_name}")
            return

        file_path = self.launch_files[file_name]
        if self.check_launch_file(file_path):
            if file_name not in self.processes:
                self.processes[file_name] = subprocess.Popen(["roslaunch", file_path])
                print(f"Executing {file_name}")
            else:
                print(f"{file_name} has Executed")
        else:
            print(f"No such .launch file named : {file_path}")

    def stop(self, file_name):
        if file_name in self.processes:
            self.processes[file_name].send_signal(signal.SIGINT)
            self.processes[file_name].wait()
            del self.processes[file_name]
            print(f"Stopped {file_name}")
        else:
            print(f"{file_name} No Executing")

    def check_status(self):
        rospy.init_node('launch_manager', anonymous=True)
        for file_name, process in self.processes.items():
            if process.poll() is None:
                print(f"{file_name} In Progress")
                # 检查该launch文件启动的节点
                nodes = rosnode.get_node_names()
                print(f"  Activate Node: {', '.join(nodes)}")
            else:
                print(f"{file_name} has Stopped")

def main():
    launch_files = {
        "file1": "/path/to/your/launch/file1.launch",
        "file2": "/path/to/your/launch/file2.launch"
    }
    manager = LaunchManager(launch_files)

    while True:
        command = input("输入命令 (start/stop/status/quit): ").lower().split()
        if not command:
            continue

        action = command[0]
        if action == "start" and len(command) > 1:
            manager.start(command[1])
        elif action == "stop" and len(command) > 1:
            manager.stop(command[1])
        elif action == "status":
            manager.check_status()
        elif action == "quit":
            for file_name in list(manager.processes.keys()):
                manager.stop(file_name)
            break
        else:
            print("无效命令。用法: start <file_name>, stop <file_name>, status, 或 quit")

        time.sleep(1)

if __name__ == "__main__":
    # main()
    print('master')