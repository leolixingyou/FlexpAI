


import matplotlib.pyplot as plt
import math
import numpy as np

from carla_msgs.msg import CarlaEgoVehicleInfo, CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from std_msgs.msg import Bool
import rospy
from nav_msgs.msg import Odometry

from rospy.exceptions import ROSInterruptException
from rospy import ServiceException

import cv2
import numpy as np
from transforms3d.euler import quat2euler

from model_predictive_speed_and_steer_control_modified import *
from carla_spawn_example import spawn_ego_vehicle, clean_ego_vehicle

TARGET_SPEED = 10 / 3.6 # m/s

def plot_car(ax_main, x, y ,yaw , steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
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

def mpc_init(g_yaw, velocity):
    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_my_course2(dl)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    initial_state = State(x=0, y=0, yaw=g_yaw, v=velocity)
    target_ind, _ = calc_nearest_index(initial_state, cx, cy, cyaw, 0)

    return [cx, cy, cyaw, sp, initial_state, dl, ck, target_ind]

def update_mpc_state(state, my_info):
    my_x, my_y, my_yaw, my_speed = my_info
    state.x = my_x
    state.y = my_y
    state.v = my_speed
    state.yaw = my_yaw
    return state

def do_mpc(mpc_infos, g_x, g_y, g_yaw, speed, _control, wheel_max_angle, odelta, oa):

    def get_my_pose(initial, v):
        # transform world to my coordinate
        my_info = {}
        # my_info[initial.x - v[0]] = 'my_x'
        # my_info[initial.y - v[1]] = 'my_y'
        my_info[v[0]] = 'my_x'
        my_info[v[1]] = 'my_y'


        my_info[v[2]] = 'my_yaw'
        my_info[v[3]] = 'my_yaw'
        return my_info
    
    cx, cy, cyaw, sp, initial_state, dl, ck, target_ind = mpc_infos

    my_info = get_my_pose(initial_state, [g_x, g_y, g_yaw, speed])
    
    state = initial_state
    xref, target_ind, dref = calc_ref_trajectory(
    state, cx, cy, cyaw, ck, sp, dl, target_ind)

    x0 = [state.x, state.y, state.v, state.yaw]  # current state

    oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
        xref, x0, dref, oa, odelta)

    di, ai = 0.0, 0.0
    if odelta is not None:
        di, ai = odelta[0], oa[0] # radian
        state = update_mpc_state(state, my_info)
        _control.brake = 0

        # unit:radian -> the mpc output direction is opposite Carla's
        # _control.steer = ((-di) / wheel_max_angle) * .2 
        _control.steer = ((-di) / wheel_max_angle) 
        # _control.steer = 0

        if ai > 0:
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = ai * 0.5
        else:
            _control.gear = -1
            _control.reverse = _control.gear < 0
            _control.throttle = ai * 0.5
            # _control.throttle = 0
            # _control.brake = 1

    return _control, ox, oy, xref, target_ind


class Controller_MPC:
    def __init__(self, config_file) -> None:
        self.config_file = config_file
        ## !!! spawn vehicle must be before than subscriber !!!
        spawn_ego_vehicle(config_file)

        rospy.Subscriber("/carla/ego_vehicle/odometry",Odometry, self.cb_odometry)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_info",CarlaEgoVehicleInfo, self.cb_vehicle_info)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_status",CarlaEgoVehicleStatus, self.cb_vehicle_status)
        self.vehicle_control_publisher = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=1)

        self.init_plot = False
        self.wheel_max_angle = None
        self.g_x = None
        self.g_y = None
        self.g_yaw = None
        self.speed = None

    def cleanup(self,):
        clean_ego_vehicle(self.config_file)

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

    def steering_to_wheel(self, steer_norm, wheel_max_angle):
        wheel_anlge_from_steer = np.abs(steer_norm) * wheel_max_angle
        return wheel_anlge_from_steer if steer_norm >0 else -wheel_anlge_from_steer
    
    def plot_car_state(self, g_x_update, g_y_update, g_yaw_update, ax_main, time, wheel_anlge, mpc_infos, ox, oy, xref, target_ind):
        cx, cy, cyaw, sp, initial_state, dl, ck, _ = mpc_infos
        
        self.x.append(leave_some_number(g_x_update))
        self.y.append(leave_some_number(g_y_update))
        self.yaw.append(leave_some_number(np.sin((g_yaw_update))))
        self.t.append(leave_some_number(time))

        ax_main.clear()
        # Main window drawing code
        if ox is not None:
            ax_main.plot(ox, oy, "xr", label="MPC")
        ax_main.plot(cx, cy, "-r", label="course")
        ax_main.plot(self.x, self.y, "ob", label="trajectory")
        ax_main.plot(xref[0, :], xref[1, :], "xk", label="xref")
        ax_main.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        plot_car(ax_main, g_x_update, g_y_update, g_yaw_update, steer=wheel_anlge)
        ax_main.axis("equal")
        ax_main.grid(True)
        ax_main.set_title("Time[s]:" + str(round(time, 2))
                + ", yaw[radian]:" + str(round(g_yaw_update, 2))
                + ", steer[radian]:" + str(round(wheel_anlge, 2)))
        ax_main.legend()

        plt.pause(0.0001)

    def run(self,):
        self.x = []
        self.y = []
        self.yaw = []
        self.t = []

        img = np.zeros([1,1])
        cv2.imshow('img',img)

        fig = plt.figure(figsize=(10, 5))  # Increase the overall graphic size
        # Create a large main window and three small status windows
        ax_main = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

        while not rospy.is_shutdown():
            time = 0.0
            g_x, g_y, g_yaw = self.g_x, self.g_y, self.g_yaw
            
            """
            range -1 ~ 1; 
            Carla Direction: - is Left + is Right;
            mpc or Cartesian coordinate system: + is Right - is Left
            """
            
            steer_norm = -self.status.control.steer
            speed = self.status.velocity
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if all([g_x, g_y, g_yaw,self.wheel_max_angle]):
                if not self.init_plot:
                    mpc_infos = mpc_init(g_yaw, speed)
                    odelta, oa = None, None
                    self.x.append(leave_some_number(g_x))
                    self.y.append(leave_some_number(g_y))
                    self.yaw.append(leave_some_number(np.sin((g_yaw))))
                    self.t.append(0.0)
                    _control = CarlaEgoVehicleControl()
                    wheel_max_angle = self.wheel_max_angle
                    self.init_plot = True
                else:
                    wheel_anlge_from_listener = self.steering_to_wheel(steer_norm, wheel_max_angle)
                    _control, ox, oy, xref, target_ind = do_mpc(mpc_infos, g_x, g_y, g_yaw, speed, _control, wheel_max_angle, odelta, oa)
                    self.vehicle_control_publisher.publish(_control)
                    self.plot_car_state(g_x, g_y, g_yaw, ax_main, time, wheel_anlge_from_listener, mpc_infos, ox, oy, xref, target_ind)


            rospy.Rate(10).sleep
        self.cleanup()

if __name__ == '__main__':
    rospy.init_node('asdf')
    config_file = '/workspace/src/base_io/src/carla_bridge/objects.json'
    vehicle_listener = Controller_MPC(config_file)
    try:
        vehicle_listener.run()
        plot_update(vehicle_listener.x, vehicle_listener.y)
    except (ROSInterruptException, ServiceException, KeyboardInterrupt):
        pass