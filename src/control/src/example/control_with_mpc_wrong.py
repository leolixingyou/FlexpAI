
"""This file is mpc example for matplot and carla with straight"""


import matplotlib.pyplot as plt
import math
import numpy as np
from collections import deque

import cvxpy

from carla_msgs.msg import CarlaEgoVehicleInfo, CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from std_msgs.msg import Bool
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header # pylint: disable=wrong-import-order
from ackermann_msgs.msg import AckermannDrive

from rospy.exceptions import ROSInterruptException
from rospy import ServiceException

import cv2
import numpy as np
from transforms3d.euler import quat2euler

# from class_mpc importa *
from carla_spawn_example import spawn_ego_vehicle, clean_ego_vehicle
from PathPlanning.CubicSpline import cubic_spline_planner
from utils.angle import angle_mod

print = str

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 10  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
# Q = np.diag([1.0, 1.0, .5, .5])  # state cost matrix
Q = np.diag([1.0, 1.0, 50, 40.])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
# MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iterationss
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 30.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10 # 10  # Search index number

DT = .1  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]q

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(15.0)  # maximum steering speed [rad/s]
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = 0.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.  # maximum accel [m/ss]

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.observed_acc = 0
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
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    # TODO: speed adeption with index """This file is mpc example for matplot and carla with straight"""

    speed_profile[-1] = 0.0

    return speed_profile

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

def pi_2_pi(angle):
    return angle_mod(angle)

def friction_difine(state):
    mu = 0.015  # Friction coeffecient
    g = 9.81    # Gravity acceleration

    # Calculating deceleration due to friction
    friction_acc = -mu * g * math.cos(state.yaw)  # Consider the influence of slope
    return friction_acc

def update_state_main(state, ai, delta):
    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT

    friction_acc = friction_difine(state)
    
    # Actual observed acceleration (command acceleration + friction)
    state.observed_acc = ai - friction_acc

    # Update velocity with observed acceleration
    state.v = state.v + state.observed_acc * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED
    return state

def update_state_matplot(state, ai, delta):
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER
    return update_state_main(state, ai, delta)

def update_state_carla(state, delta, ai, wheel_max_angle):
    if delta >= wheel_max_angle:
        delta = wheel_max_angle
    elif delta <= -wheel_max_angle:
        delta = -wheel_max_angle
    return update_state_main(state, ai, delta)

def predict_motion_carla(x0, oa, od, xref, wheel_max_angle):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state_carla(state, ai, di, wheel_max_angle)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

def iterative_linear_mpc_control_carla(xref, x0, dref, oa, od, wheel_max_angle):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw, ov = None, None, None, None
    his_ox, his_oy, his_oyaw, his_ov = None, None, None, None
    
    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion_carla(x0, oa, od, xref, wheel_max_angle)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        # du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        try:
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            his_ox, his_oy, his_oyaw, his_ov = ox, oy, oyaw, ov
        except:
            ox, oy, oyaw, ov = his_ox, his_oy, his_oyaw, his_ov
            du = 1
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state_matplot(state, ai, di)
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
    his_ox, his_oy, his_oyaw, his_ov = None, None, None, None
    
    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        try:
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            his_ox, his_oy, his_oyaw, his_ov = ox, oy, oyaw, ov
        except:
            ox, oy, oyaw, ov = his_ox, his_oy, his_oyaw, his_ov
            du = 1
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

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
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB
    fmr_wheel = (fmr_wheel.T.dot(Rot2)).T
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
        return float("%.3f" % v)

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

def corrinate_transformation(v):
    v_array = np.array(v)
    v_array -= v_array[0]
    return v_array

def reduce_course(n):
    with open('/workspace/src/control/src/example/odm_x_y_full_course_town05.txt', 'r') as file:
        data = file.readlines()
    data_temp = [x for i,x in enumerate(data) if i % n ==0 ]
    info_temp = [x.split(',')[:2] for x in data_temp]
    data_float = []
    for info in info_temp:
        print(info)
        float_info = [round(float(x),2) for x in info]
        data_float.append(float_info)
    data_int = np.array(data_float)
    ax, ay = data_int[:,0], data_int[:,1]
    return ax, ay

def init_course(dl): #curve
    # make start point to (0,0)
    ## straght path
    ax = [383.8, 334.9]
    ay = [-326.9, -329.2]
    # ## curves
    # ax = [383.8, 342.6, 346.5, 338.1]
    # ay = [-326.9, -327.7, -326.9, -203.7]
    # ## curves
    # ax = [383.8, 346.5, 338.1, 310.3]
    # ay = [-326.9, -326.9, -203.7, -195.5]
    ## round
    # ax, ay = read_txt_round()

    ax, ay = reduce_course(2)    

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        corrinate_transformation(ax), corrinate_transformation(ay), ds=dl)
    return cx, cy, cyaw, ck, [ax[0], ay[0]]

def mpc_init(mode, g_yaw=0, velocity=0):
    dl = 1.0  # course tick
    cx, cy, cyaw, ck, initial_position = init_course(dl)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    if mode == 'carla':
        state = State(x=cx[0], y=cy[0], yaw=g_yaw, v=velocity)
    if mode == 'matplot':
        state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=velocity)
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    goal = [cx[-1], cy[-1]]

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    return state, target_ind, goal, [cx, cy, cyaw, sp, dl, ck, initial_position]

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

        # isstop = (abs(state.v) <= STOP_SPEED)

        # if isgoal and isstop:
        #     return True

        return False

def pacakage_msg(target_steer, target_speed, target_acc, target_jerk):
    msg = AckermannDrive()
    msg.steering_angle = target_steer
    msg.steering_angle_velocity = 0
    msg.speed = target_speed
    msg.acceleration = target_acc
    msg.jerk =target_jerk 
    return msg

def read_txt_round():
    with open('/workspace/src/control/src/example/odm_x_y_full_course_town05.txt') as f:
        data = f.readlines()
    x_y = np.array([x.split(',') for x in data])[:, :2]
    ax, ay = x_y[:,0], x_y[:,1]
    def numpy_string_to_list(v):
        return [float(x) for x in v]
    return numpy_string_to_list(ax), numpy_string_to_list(ay)

class Controller_MPC:
    def __init__(self) -> None:
        self.init_plot = False
        self.wheel_max_angle = None
        self.g_x = None
        self.g_y = None
        self.g_yaw = None
        self.speed = None
        self.get_info = False

    def mpc_for_carla(self,config_file):
        rospy.init_node('asdf')
        self.config_file = config_file
        ## !!! spawn vehicle must be before than subscriber !!!
        spawn_ego_vehicle(config_file)
        
        self.vehicle_control_publisher_arcker = rospy.Publisher('/carla/ego_vehicle/ackermann_cmd', AckermannDrive , queue_size=1)

        rospy.Subscriber("/carla/ego_vehicle/odometry",Odometry, self.cb_odometry)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_info",CarlaEgoVehicleInfo, self.cb_vehicle_info)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_status",CarlaEgoVehicleStatus, self.cb_vehicle_status)
        self.vehicle_control_publisher = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=1)

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
        self.get_info = True

    def cb_vehicle_info(self, msg):
        # will not update
        self.wheel_max_angle = msg.wheels[0].max_steer_angle
        print(f'wheel_max_angle is {self.wheel_max_angle}')

    def steering_to_wheel(self, steer_norm, wheel_max_angle):
        wheel_anlge_from_steer = np.abs(steer_norm) * wheel_max_angle
        return wheel_anlge_from_steer if steer_norm >0 else -wheel_anlge_from_steer
    
    def plot_car_state(self, g_infos, wheel_anlge, mpc_infos, mpc_updates, axs):
        g_x_update, g_y_update, g_yaw_update = g_infos
        cx, cy, cyaw, sp, dl, ck, _ = mpc_infos
        ax_main, ax1, ax2, ax3 = axs
        ox, oy, xref, target_ind, di, ai = mpc_updates
        x_update, y_update, yaw_update, v_update \
            = self.state.x, self.state.y, self.state.yaw, self.state.v
        self.x.append(leave_some_number(g_x_update))
        self.y.append(leave_some_number(g_y_update))
        self.yaw.append(leave_some_number(np.sin((yaw_update))))
        self.v.append(leave_some_number(v_update))
        self.t.append(leave_some_number(self.time))
        self.d.append(leave_some_number(di))
        self.a.append(leave_some_number(ai))

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
        ax_main.set_title("Time[s]:" + str(round(self.time, 2))
                + ", speed[km/h]:" + str(round(v_update*3.6, 2))
                + ", yaw[radian]:" + str(round(yaw_update, 2))
                + ", steer[radian]:" + str(round(wheel_anlge, 2)))
        ax_main.legend()
        ax_main.set_xlim(g_x_update-10, g_x_update+10)

        ax1.plot(self.t, self.yaw, "-r", label="yaw")
        ax1.grid(True)
        ax1.set_title('Fig 1.G_Yaw; Fig 2. V_Yaw; Fig 3. V_Accelerate')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Yaw [sin]")
        ax1.set_ylim(-1,1)

        ax2.plot(self.t, self.d, "-r", label="d_yaw")
        ax2.grid(True)
        ax2.set_title('')
        ax2.set_xlabel("qTime [s]")
        ax2.set_ylabel("yaw [rad]")

        ax3.plot(self.t, self.a, "-r", label="d_acc")
        ax3.grid(True)
        ax3.set_title('')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("acc [m/ss]")
        
        plt.pause(0.0001)

    def do_matplot(self, mpc_infos, odelta, oa):

        cx, cy, cyaw, sp, dl, ck,_= mpc_infos

        xref, self.target_index, dref = calc_ref_trajectory(
        self.state, cx, cy, cyaw, ck, sp, dl, self.target_index)

        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0] # radian
            self.state = update_state_matplot(self.state, ai, di)

        return _, [ox, oy, xref, self.target_index, di, ai]
    
    def matplot_loop(self, mode, axs):

        self.state, self.target_index, goal, mpc_infos = mpc_init(mode)
        odelta, oa = None, None

        self.x.append(leave_some_number(self.state.x))
        self.y.append(leave_some_number(self.state.y))
        self.yaw.append(leave_some_number(np.sin((self.state.yaw))))
        self.t.append(0.0)
        self.v.append(self.state.v)
        self.d.append(0.0)
        self.a.append(0.0)

        self.state.x, self.state.y = -290,-0.2

        while 1 :
            g_x, g_y = self.state.x, self.state.y 
            _, mpc_updates = self.do_matplot(mpc_infos, odelta, oa)
            self.plot_car_state([g_x, g_y, self.state.yaw], self.state.yaw, mpc_infos, mpc_updates, axs)
            self.time += DT

    def acc_brake_with_velocity(self,_control, ai, di, wheel_max_angle, cx):

        # unit:radian -> the mpc output direction is opposite Carla's
        _control.steer = -(di / wheel_max_angle)

        # if ai < 0: 
        #     _control.gear = 1
        #     _control.reverse = _control.gear < 0
        #     _control.throttle = 0 
        #     _control.brake = ai

        # elif ai >=  0: 
        #     _control.gear = 1
        #     _control.reverse = _control.gear < 0
        #     _control.throttle =  ai
        #     _control.brake = 0
        rate_throttle = abs(ai)/MAX_ACCEL

        if self.state.v <= MIN_SPEED :
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = rate_throttle 
            _control.brake = 0
            
        elif  MIN_SPEED < self.state.v <= TARGET_SPEED:
            if rate_throttle > 0:
                _control.gear = 1
                _control.reverse = _control.gear < 0
                _control.throttle = rate_throttle
                _control.brake = 0
            else:
                _control.gear = 0
                _control.reverse = _control.gear < 0
                _control.throttle = 0
                _control.brake = 0

        if self.state.v > TARGET_SPEED:
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = 0
            _control.brake = 0

        if rate_throttle < -.8 * MAX_ACCEL:
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = 0
            _control.brake = rate_throttle

        if check_goal(self.state, self.goal, self.target_index, len(cx)):
            print("Goal")
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = 0
            _control.brake = 1

        return _control
        
    def do_mpc(self, mpc_infos, g_x, g_y, g_yaw, speed, _control, wheel_max_angle, odelta, oa, controller):
        cx, cy, cyaw, sp, dl, ck,_= mpc_infos

        self.state.x = g_x
        self.state.y = g_y
        self.state.yaw = g_yaw
        self.state.v = speed

        # initial yaw compensation
        if self.state.yaw - cyaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - cyaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0

        xref, self.target_index, dref = calc_ref_trajectory(
        self.state, cx, cy, cyaw, ck, sp, dl, self.target_index)

        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control_carla(
            xref, x0, dref, oa, odelta, wheel_max_angle)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0] # radian
            self.state = update_state_carla(self.state, di, ai, wheel_max_angle)

            if controller == 'carla':
                _control = self.acc_brake_with_velocity(_control, ai, di, wheel_max_angle, cx)
            
            if controller == 'arckermann':
                target_speed = sp[self.target_index] 
                if check_goal(self.state, self.goal, self.target_index, len(cx)):
                    print('Goal')
                    _control = pacakage_msg(round(0,2), round(0,2), round(0,2), 0) ## Arckerman and mpc has same direction for yaw
                _control = pacakage_msg(round(di,2), round(target_speed,2), round(ai,2), 0) ## Arckerman and mpc has same direction for yaw
            self.hist_control = _control
    
        _control = self.hist_control

        return _control, [ox, oy, xref, self.target_index, di, ai]

    def carla_loop(self, mode, axs, controller):
        count = 0

        while not rospy.is_shutdown():
            g_x, g_y, g_yaw = self.g_x, self.g_y, self.g_yaw
            
            """
            range -1 ~ 1; 
            Carla Direction: - is Left + is Right;
            mpc or Cartesian coordinate system: + is Right - is Left
            """

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if all([g_x, g_y, g_yaw,self.wheel_max_angle]) and self.get_info:
                steer_norm = -self.status.control.steer
                speed = self.status.velocity
                if not self.init_plot:
                    self.state, self.target_index, self.goal, mpc_infos = mpc_init(mode, g_yaw, speed)
                    odelta, oa = None, None
                    self.x.append(leave_some_number(self.state.x))
                    self.y.append(leave_some_number(self.state.y))
                    self.yaw.append(leave_some_number(np.sin((self.state.yaw))))
                    self.t.append(0.0)
                    self.v.append(0.0)
                    self.d.append(0.0)
                    self.a.append(0.0)
                    _control = CarlaEgoVehicleControl()
                    wheel_max_angle = self.wheel_max_angle
                    self.init_plot = True
                else:
                    print(f'state is {self.state.x, self.state.y, self.state.yaw, self.state.v}')
                    g_x, g_y = g_x- mpc_infos[-1][0], g_y - mpc_infos[-1][1]
                    wheel_anlge_from_listener = self.steering_to_wheel(steer_norm, wheel_max_angle)

                    _control, mpc_updates = self.do_mpc(mpc_infos, g_x, g_y, g_yaw, speed, _control, wheel_max_angle, odelta, oa, controller)
                    count += 1 

                    if controller == 'carla' and count > 50:
                        self.vehicle_control_publisher.publish(_control)
                    if controller == 'arckermann':
                        self.vehicle_control_publisher_arcker.publish(_control)
                    self.time += DT
                    self.plot_car_state([g_x, g_y, g_yaw,], wheel_anlge_from_listener, mpc_infos, mpc_updates, axs)

    def run(self,config_file):
        mode = ['matplot', 'carla'][0]
        controller = ['arckermann', 'carla'][1]
        # self.x = []
        # self.y = []
        # self.yaw = []
        # self.t = []
        # self.v = []
        # self.d = []
        # self.a = []

        self.x = deque(maxlen=50)
        self.y = deque(maxlen=50)
        self.yaw = deque(maxlen=50)
        self.t = deque(maxlen=50)
        self.v = deque(maxlen=50)
        self.d = deque(maxlen=50)
        self.a = deque(maxlen=50)

        img = np.zeros([1,1])
        cv2.imshow('img',img)

        fig = plt.figure(figsize=(10, 5))  # Increase the overall graphic size
        # Create a large main window and three small status windows
        ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
        ax1 = plt.subplot2grid((3, 3), (0, 2))
        ax2 = plt.subplot2grid((3, 3), (1, 2))
        ax3 = plt.subplot2grid((3, 3), (2, 2))
        axs = [ax_main, ax1, ax2, ax3]
        self.time = 0.0
        if mode == 'matplot':
            self.matplot_loop(mode, axs)
        if mode == 'carla':
            self.mpc_for_carla(config_file)
            self.carla_loop(mode, axs, controller)
            rospy.Rate(60).sleep
        self.cleanup()

if __name__ == '__main__':
    config_file = '/workspace/src/base_io/src/carla_bridge/objects.json'
    vehicle_listener = Controller_MPC()
    try:
        vehicle_listener.run(config_file)
        plot_update(vehicle_listener.x, vehicle_listener.y)
    except (ROSInterruptException, ServiceException, KeyboardInterrupt):
        pass