#!/usr/bin/env python3
import rospy
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Header
import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Twist

class Carla_Command:
    def __init__(self, isPlot=False) -> None:
        rospy.init_node('Ctrl_CV_IO', anonymous=False)
        self.pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)
        
        # Subscribe to MPC data
        rospy.Subscriber('/Ctrl_CV/control/mpc', Twist, self.mpc_callback)
    
        self.control_data = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'hand_brake': False,
            'reverse': False,
            'gear': 0,
            'manual_gear_shift': False
        }
        
        self.isPlot = isPlot
        if self.isPlot:
            self.setup_plot()

    def setup_plot(self):
        # Setup for plotting (if enabled)
        self.fig, self.ax = plt.subplots()
        self.lines = {key: self.ax.plot([], [], label=key)[0] for key in ['throttle', 'brake', 'steer']}
        self.ax.legend()
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 100)
        self.data = {key: [] for key in ['throttle', 'brake', 'steer']}
        plt.ion()
        plt.show()

    def update_plot(self):
        # Update the plot (if enabled)
        if not self.isPlot:
            return

        for key in ['throttle', 'brake', 'steer']:
            self.data[key].append(self.control_data[key])
            if len(self.data[key]) > 100:
                self.data[key] = self.data[key][-100:]
            self.lines[key].set_data(range(len(self.data[key])), self.data[key])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def pub_command(self):
        # Publish control command to Carla
        msg = CarlaEgoVehicleControl()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        for key, value in self.control_data.items():
            setattr(msg, key, value)
        self.pub.publish(msg)

    def mpc_callback(self, msg):
        # Process received MPC data
        linear_x = msg.linear.x  # Assuming this is acceleration
        angular_z = msg.angular.z  # Assuming this is steering angle

        # Update steering
        self.control_data['steer'] = np.clip(angular_z, -1.0, 1.0)

        # Update throttle and brake
        if linear_x > 0:
            self.control_data['throttle'] = np.clip(linear_x, 0.0, 1.)
            self.control_data['brake'] = 0.0
        else:
            self.control_data['throttle'] = 0.0
            self.control_data['brake'] = np.clip(-linear_x, 0.0, 1.0)

        # Publish command
        self.pub_command()

        # Update plot (if enabled)
        if self.isPlot:
            self.update_plot()

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == "__main__":
    carla_command = Carla_Command(isPlot=False)  # Set to True to enable plotting
    carla_command.run()