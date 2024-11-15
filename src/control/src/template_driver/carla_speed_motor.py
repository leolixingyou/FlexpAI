#!/usr/bin/env python3
import rospy
import time
import curses
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus # pylint: disable=no-name-in-module,import-error
from std_msgs.msg import Header, Float32

class Carla_Command:
    def __init__(self) -> None:
        rospy.init_node('Control_Server', anonymous=False)
        self.pub = rospy.Publisher('/carla/ego_vehicle/speedometer', Float32, queue_size=1)
        # self.pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)

        ## cursesw
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.nodelay(True)  # set to unblocking mode: True

    def pub_command_status(self, ):
        msg = Float32()
        msg.data = 10

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
            # key_in = self.screen.getch()
        
            # if key_in == -1:  # no input
            #     rospy.sleep(0.1) 
            #     continue
            # self.keyboard_development(key_in)
            self.pub_command_status()
            rate.sleep()
            

if __name__ == "__main__":
    test_temp = Carla_Command()
    test_temp.run()