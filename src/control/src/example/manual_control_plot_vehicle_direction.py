


import matplotlib.pyplot as plt
import math
import numpy as np

from carla_msgs.msg import CarlaEgoVehicleInfo,CarlaEgoVehicleStatus
from std_msgs.msg import Bool
import rospy
from nav_msgs.msg import Odometry

import cv2
import numpy as np
from transforms3d.euler import quat2euler

def plot_car(ax_main, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    # Vehicle parameters
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB = 2.5  # [m]q

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    # # triangle
    # fmr_wheel =  np.array([[0,          0,  WHEEL_LEN,             0,                   0],
    #                         [0, -WHEEL_LEN,          0,    WHEEL_LEN,           -WHEEL_LEN]])

    fmr_wheel =             np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH , -WHEEL_WIDTH , WHEEL_WIDTH , WHEEL_WIDTH , -WHEEL_WIDTH ]])

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fmr_wheel = (fmr_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB
    fmr_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T
    fmr_wheel = (fmr_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    fmr_wheel[0, :] += x
    fmr_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    #plot mid line instead of two wheel
    ax_main.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    # ax_main.plot(np.array(fr_wheel[0, :]).flatten(),
    #          np.array(fr_wheel[1, :]).flatten(), truckcolor)
    # ax_main.plot(np.array(fl_wheel[0, :]).flatten(),
    #          np.array(fl_wheel[1, :]).flatten(), truckcolor)
    ax_main.plot(np.array(fmr_wheel[0, :]).flatten(),
             np.array(fmr_wheel[1, :]).flatten(), truckcolor)
    ax_main.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    ax_main.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    ax_main.plot(x, y, "*")

def leave_some_number(v):
        return float("%.7f" % v)

class Vehicle_Listener:
    def __init__(self) -> None:
        rospy.Subscriber("/carla/ego_vehicle/odometry",Odometry, self.cb_odometry)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_info",CarlaEgoVehicleInfo, self.cb_vehicle_info)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_status",CarlaEgoVehicleStatus, self.cb_vehicle_status)

        self.init_plot = False
        self.wheel_max_angle = None
        self.g_x = None
        self.g_y = None
        self.g_yaw = None
        self.speed = None

    def cb_odometry(self, msg):
        self.g_x = msg.pose.pose.position.x
        self.g_y = msg.pose.pose.position.y
        _, _, self.g_yaw = quat2euler(
            [msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z])
        # yaw = math.degrees(yaw)

    def cb_vehicle_status(self, msg):
        self.status = msg

    def cb_vehicle_info(self, msg):
        # will not update
        self.wheel_max_angle = msg.wheels[0].max_steer_angle
        print(f'wheel_max_angle is {self.wheel_max_angle}')


    def plot_route(self, g_x_update, g_y_update, g_yaw_update, ax_main, time, steer):

        di = self.init_yaw - g_yaw_update
        self.x.append(leave_some_number(g_x_update))
        self.y.append(leave_some_number(g_y_update))
        self.yaw.append(leave_some_number(np.sin((g_yaw_update))))
        self.t.append(leave_some_number(time))
        self.d.append(leave_some_number(di))


        ax_main.clear()
        # Main window drawing code
        ax_main.plot(self.x, self.y, "ob", label="trajectory")
        plot_car(ax_main, g_x_update, g_y_update, g_yaw_update, steer=steer )
        ax_main.axis("equal")
        ax_main.grid(True)
        ax_main.set_title("Time[s]:" + str(round(time, 2))
                + ", yaw[radian]:" + str(round(g_yaw_update, 2))
                + ", d_yaw[radian]:" + str(round(di, 2))
                + ", steer[radian]:" + str(round(steer, 2)))
        ax_main.legend()

        plt.pause(0.0001)


    def run(self,):
        self.x = []
        self.y = []
        self.yaw = []
        self.t = []
        self.d = []

        img = np.zeros([1,1])
        cv2.imshow('img',img)

        fig = plt.figure(figsize=(10, 5))  # Increase the overall graphic size
        # Create a large main window and three small status windows
        ax_main = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

        while not rospy.is_shutdown():
            time = 0.0
            x, y, yaw = self.g_x, self.g_y, self.g_yaw
            
            """
            range -1 ~ 1; 
            Carla Direction: - is Left + is Right;
            mpc or Cartesian coordinate system: + is Right - is Left
            """
            
            steer_norm = -self.status.control.steer 
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if all([self.g_x, self.g_y, self.g_yaw]):
                if not self.init_plot:
                    wheel_max_angle = self.wheel_max_angle
                    self.x.append(leave_some_number(x))
                    self.y.append(leave_some_number(y))
                    self.yaw.append(leave_some_number(np.sin((yaw))))
                    self.t.append(0.0)
                    self.d.append(0.0)
                    self.init_yaw = self.g_yaw
                    self.init_plot = True
                else:
                    steer_temp = np.abs(steer_norm) * wheel_max_angle
                    steer = steer_temp if steer_norm >0 else -steer_temp
                    self.plot_route(x, y, yaw, ax_main, time, steer)
            rospy.Rate(10).sleep##
        plot_update(x, y)


def plot_update(x, y ):
    plt.close("all")
    plt.subplots()
    plt.plot(x, y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    rospy.init_node('asdf')
    vehicle_listener = Vehicle_Listener()
    vehicle_listener.run()