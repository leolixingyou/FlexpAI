#!/usr/bin/env python3
import rospy
import time
import curses
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus # pylint: disable=no-name-in-module,import-error
from std_msgs.msg import Header
from mpc import *

class Carla_Command:
    def __init__(self) -> None:
        rospy.init_node('Control_Server', anonymous=False)
        self.pub = rospy.Publisher('/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, queue_size=1)
        # self.pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)
        self.control_data = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'hand_brake': False,
            'reverse': False,
            'gear': 0,
            'manual_gear_shift': False
        }
        ## cursesw
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.nodelay(True)  # set to unblocking mode: True

    def pub_command_control(self, ):
        """
        std_msgs/Header header
            uint32 seq
            time stamp
            string frame_id
        carla_ackermann_msgs/EgoVehicleControlMaxima restrictions
            float32 max_steering_angle
            float32 max_speed
            float32 max_accel
            float32 max_decel
            float32 min_accel
            float32 max_pedal
        carla_ackermann_msgs/EgoVehicleControlTarget target
            float32 steering_angle
            float32 speed
            float32 speed_abs
            float32 accel
            float32 jerk
        carla_ackermann_msgs/EgoVehicleControlCurrent current
            float32 time_sec
            float32 speed
            float32 speed_abs
            float32 accel
        carla_ackermann_msgs/EgoVehicleControlStatus status
            string status
            uint8 speed_control_activation_count
            float32 speed_control_accel_delta
            float32 speed_control_accel_target
            float32 accel_control_pedal_delta
            float32 accel_control_pedal_target
            float32 brake_upper_border
            float32 throttle_lower_border
        carla_msgs/CarlaEgoVehicleControl output
            std_msgs/Header header
                uint32 seq
                time stamp
                string frame_id
            float32 throttle
            float32 steer
            float32 brake
            bool hand_brake
            bool reverse
            int32 gear
            bool manual_gear_shift
        """
        msg = CarlaEgoVehicleControl()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.throttle = self.control_data['throttle']
        msg.steer = self.control_data['steer']
        msg.brake = self.control_data['brake']
        msg.hand_brake = self.control_data['hand_brake']
        msg.reverse = self.control_data['reverse']
        msg.gear = self.control_data['gear']
        msg.manual_gear_shift = self.control_data['manual_gear_shift']
        self.pub.publish(msg)

    def pub_command_status(self, ):
        """
        header: 
            seq: 67865
            stamp: 
                secs: 4701
                nsecs: 950934564
            frame_id: "map"
        velocity: 0.0
        acceleration: 
            linear: 
                x: 0.0
                y: -0.0
                z: 0.0
            angular: 
                x: 0.0
                y: 0.0
                z: 0.0
        orientation: 
            x: -9.734735440929984e-09
            y: -5.325432196310676e-07
            z: 0.9998329679659436
            w: 0.01827665637168592
        control: 
            header: 
                seq: 0
                stamp: 
                secs: 0
                nsecs:        0
                frame_id: ''
            throttle: 0.0
            steer: 0.0
            brake: 0.0
            hand_brake: False
            reverse: False
            gear: 1
            manual_gear_shift: False

        """
        msg = CarlaEgoVehicleStatus()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.velocity = self.control_data['speed_set']
        msg.control.steer = self.control_data['steer']
        msg.control.brake = self.control_data['brake']
        msg.control.hand_brake = self.control_data['hand_brake']
        msg.control.reverse = self.control_data['reverse']
        msg.control.gear = self.control_data['gear']
        msg.control.manual_gear_shift = self.control_data['manual_gear_shift']
        self.pub.publish(msg)

    def keyboard_development(self, key ):

        self.control_data = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'hand_brake': False,
            'reverse': False,
            'gear': 0,
            'manual_gear_shift': False,
            'speed_set': 0.,
        }
        if key == ord('w'):
            self.control_data['throttle'] = 1.
            self.control_data['brake'] = 0.0
        elif key == ord('s'):
            self.control_data['brake'] = 1.
            self.control_data['throttle'] = 0.0
        elif key == ord('a'):
            self.control_data['steer'] = 1.
        elif key == ord('d'):
            self.control_data['steer'] = 1.
        elif key == ord('z'):
            self.control_data['speed_set'] = 10.
        elif key == ord('x'):
            self.control_data['speed_set'] = 0.

    def __del__(self):
        # destory instance and recover terminal settings
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()


    def run(self):
        rate = rospy.Rate(5)  # 10 Hz
        temp_flag = False
        while not rospy.is_shutdown():
            key_in = self.screen.getch()
        
            if key_in == -1:  # no input
                rospy.sleep(0.1) 
                continue
            self.keyboard_development(key_in)
            self.pub_command_status()
            rate.sleep()
            

if __name__ == "__main__":
    test_temp = Carla_Command()
    test_temp.run()