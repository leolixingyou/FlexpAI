#!/usr/bin/env python3
"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import cvxpy
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Twist

from yaw_converter import GPSYawCalculator
from ros_tools import *
from pyproj import Transformer

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True



class GNSStoUTMConverter:
    def __init__(self):
        # Initialize the converter
        pass

    def determine_utm_zone(self, lon):
        # Determine the UTM zone based on longitude
        return int((lon + 180) / 6) + 1

    def convert(self, lat, lon):
        # Convert GNSS coordinates to UTM
        zone_number = self.determine_utm_zone(lon)
        
        # Create a transformer object
        transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 GNSS coordinate system
            f"+proj=utm +zone={zone_number} +datum=WGS84",  # UTM coordinate system
            always_xy=True
        )
        
        # Perform the transformation
        easting, northing = transformer.transform(lon, lat)
        
        return easting, northing, zone_number

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

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
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state

def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov =  linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs( pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

def pi_2_pi(angle):
    return angle_mod(angle)

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref

def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False

def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def angle_norm(yaw_angle):
    return  yaw_angle / MAX_STEER 

def accelerate_norm(acc_current):
    return  acc_current / MAX_ACCEL

def write_log(input_list, file_name):
    with open(file_name, 'a') as file:
        file.write(','.join(map(str, input_list))+ '\n')

def read_file_to_list(filename):
    with open(filename, 'r') as file:
        content = file.read()
        return [int(item) for item in content.split(',') if item]

class MPC_Controller:
    def __init__(self, isPlot = False) -> None:
        # rospy.init_node('Control_MPC', anonymous=False)
        self.isRuningMPC = False
        self.cmd_pub = rospy.Publisher('/Ctrl_CV/control/mpc', Twist, queue_size=1)

        platform = 'carla'
        sensors = SensorConfig(platform)
        self.gps_listener = GPS_GNSS_Listener(sensors.sensor_config['Gps'])
        self.gps_converter = GNSStoUTMConverter()

        vehicle = VehicleConfig(platform)
        self.ego_vehicle = CarlaEgoVehicle_Listener(vehicle.vehicle_config['Ego'])

        self.yaw_convertor = GPSYawCalculator()

        self.isPlot = isPlot        
        self.isGPSGet = False
        self.isVEHGet = False
        self.isYAWGet = False

    def load_path_data(self, filename='/workspace/src/control/src/path_data.txt'):
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        cx, cy, cyaw, ck = data.T
        return cx, cy, cyaw, ck

    def plot_path_and_direction(self, cx, cy, cyaw):
        plt.figure(figsize=(10, 8))
        
        # Plot the path
        plt.plot(cx, cy, '-b', label='Path')
        
        # Plot direction arrows
        arrow_spacing = max(1, len(cx) // 20)  # Adjust spacing of arrows
        for i in range(0, len(cx), arrow_spacing):
            plt.arrow(cx[i], cy[i], 
                    math.cos(cyaw[i]), math.sin(cyaw[i]), 
                    head_width=0.3, head_length=0.6, fc='r', ec='r')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path and Direction')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def do_mpc(self, cx, cy, cyaw, ck, sp, dl, initial_state, easting, northing, speed, steering):
        """
        Simulation

        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]

        """

        goal = [cx[-1], cy[-1]]

        state = initial_state

        # initial yaw compensation
        if state.yaw - cyaw[0] >= math.pi:
            state.yaw -= math.pi * 2.0
        elif state.yaw - cyaw[0] <= -math.pi:
            state.yaw += math.pi * 2.0

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]
        target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

        odelta, oa = None, None

        cyaw = smooth_yaw(cyaw)

        while MAX_TIME >= time:
            xref, target_ind, dref = calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl, target_ind)

            x0 = [easting, northing, speed, steering]  # current state

            oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
                xref, x0, dref, oa, odelta)

            di, ai = 0.0, 0.0
            if odelta is not None:
                di, ai = angle_norm(odelta[0]), accelerate_norm(oa[0]) # delta is steering, a is accelerate
            else:
                di, ai = None, None

            if di != None and ai != None:
                # write_log([di, ai, easting, northing, speed, steering],'/workspace/src/control/src/do_carla_data.txt')
                twist_msg = Twist()
                twist_msg.linear.x = -ai if ai is not None else 0.0  # Set acceleration as linear velocity
                twist_msg.angular.z = di if di is not None else 0.0  # Set steering as angular velocity
                self.cmd_pub.publish(twist_msg)

            # Update state (you may need to implement this based on your simulation)
            # state = update_state(state, ai, di)

            time += DT

            # Break the loop if goal is reached
            if check_goal(state, goal, target_ind, len(cx)):
                print("Goal reached")
                break

            # Add a small delay to control the loop rate
        self.isRuningMPC = False
        
    def run(self, cx, cy, cyaw, ck, dl):
        print(__file__ + " start!!")
        spline = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

        initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
        t, x, y, yaw, v, d, a = do_simulation(
                    cx, cy, cyaw, ck, spline, dl, initial_state)
        
        # while not rospy.is_shutdown():
        #     self.gps_listener.gathering_msg()
        #     datas, times = self.ego_vehicle.gathering_msg()

        #     if self.gps_listener.data_received:
        #         location = self.gps_listener.datas[-1]
        #         easting, northing, zone = self.gps_converter.convert(location[0], location[1])
        #         self.isGPSGet = True

        #     if self.isGPSGet:
        #         yaw = self.yaw_convertor.update(easting, northing)
        #         if yaw != None:
        #             self.isYAWGet = True

        #     if self.ego_vehicle.data_received:
        #         data = self.ego_vehicle.datas[-1]
        #         speed = data.velocity
        #         steering = data.control.steer
        #         self.isVEHGet = True

        #     if not self.isRuningMPC and self.isGPSGet and self.isVEHGet and self.isYAWGet:
        #         self.isRuningMPC = True
        #         if self.isPlot:
        #             self.plot_path_and_direction(cx, cy, cyaw)
        #         print(__file__ + " start!!")
        #         spline = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

        #         initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

        #         # t, x, y, yaw, v, d, a = self.do_mpc(
        #         #     cx, cy, cyaw, ck, spline, dl, initial_state, easting, northing, speed, yaw)

        #     else:
        #         print(f'MPC is Running')

def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)

        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ", acc[m/s^2]:" + str(round(ai, 2))
                      + ", steering[rad]:" + str(round(di, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a
if __name__ == '__main__':
    try:
        mpc = MPC_Controller()
        cx, cy, cyaw, ck = mpc.load_path_data()
        mpc.run(cx, cy, np.deg2rad(cyaw), ck, 1)
    except rospy.ROSInterruptException:
        pass