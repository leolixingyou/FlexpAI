"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from utils.angle import angle_mod

from PathPlanning.CubicSpline import cubic_spline_planner

from carla_msgs.msg import CarlaEgoVehicleControl,CarlaEgoVehicleStatus
from std_msgs.msg import Bool
import rospy
from nav_msgs.msg import Odometry

import cv2
import numpy as np
from transforms3d.euler import quat2euler

from  manual_control_plot_vehicle_direction import plot_car
g_x = 0
g_y = 0
g_z = 0
g_yaw = 0
def cb_odometry(data):
    global g_x,g_y,g_z,g_yaw
    g_x = data.pose.pose.position.x
    g_y = data.pose.pose.position.y
    g_z = data.pose.pose.position.z
    _, _, g_yaw = quat2euler(
        [data.pose.pose.orientation.w,
        data.pose.pose.orientation.x,
        data.pose.pose.orientation.y,
        data.pose.pose.orientation.z])
    # yaw = math.degrees(yaw)

g_status = CarlaEgoVehicleStatus()
def cb_status(data):
    global g_status
    g_status = data


_control = CarlaEgoVehicleControl()
vehicle_control_publisher = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=1)

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
WB = 2.5  # [m]q

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True


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

def pi_2_pi(angle):
    return angle_mod(angle)

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

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

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
    def a_b(v):
            return float("%.7f" % v)
    
    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [a_b(state.x)]
    y = [a_b(state.y)]
    yaw = [a_b(np.sin(state.yaw))]
    v = [a_b(state.v)]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)
    
    fig = plt.figure(figsize=(10, 5))  # Increase the overall graphic size
    # Create a large main window and three small status windows
    ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    ax1 = plt.subplot2grid((3, 3), (0, 2))
    ax2 = plt.subplot2grid((3, 3), (1, 2))
    ax3 = plt.subplot2grid((3, 3), (2, 2))

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            # di, ai = 0, oa[0]
            state = update_state(state, ai, di)

        time = time + DT
        
        x.append(a_b(state.x))
        y.append(a_b(state.y))
        yaw.append(a_b(np.sin((state.yaw))))
        v.append(a_b(state.v))
        t.append(a_b(time))
        d.append(a_b(di))
        a.append(a_b(ai))
        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            ax_main.clear()
            # Main window drawing code
            if ox is not None:
                ax_main.plot(ox, oy, "xr", label="MPC")
            ax_main.plot(cx, cy, "-r", label="course")
            ax_main.plot(x, y, "ob", label="trajectory")
            ax_main.plot(xref[0, :], xref[1, :], "xk", label="xref")
            ax_main.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(ax_main, state.x, state.y, state.yaw, steer=di)
            ax_main.axis("equal")
            ax_main.grid(True)
            ax_main.set_title("Time[s]:" + str(round(time, 2))
                    + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            ax_main.legend()


            ax1.plot(t, yaw, "-r", label="yaw")
            ax1.grid(True)
            ax1.set_title('Fig 1.G_Yaw; Fig 2. V_Yaw; Fig 3. V_Accelerate')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Yaw [sin]")
            ax1.set_ylim(-1,1)

            ax2.plot(t, d, "-r", label="d_yaw")
            ax2.grid(True)
            ax2.set_title('')
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("yaw [rad]")

            ax3.plot(t, a, "-r", label="d_acc")
            ax3.grid(True)
            ax3.set_title('')
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("acc [m/ss]")

            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a

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

    # TODO: speed adeption with index 
    speed_profile[-1] = 0.0

    return speed_profile

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

def get_my_course1(dl): #straight
    ax = [383.8, 334.9]
    ay = [-326.9, -326.9]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def do_sim_carla(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """
    def a_b(v):
            return float("%.7f" % v)
    goal = [cx[-1], cy[-1]]

    state = initial_state

    # # initial yaw compensation
    # if state.yaw - cyaw[0] >= math.pi:
    #     state.yaw -= math.pi * 2.0
    # elif state.yaw - cyaw[0] <= -math.pi:
    #     state.yaw += math.pi * 2.0

    time = 0.0
    x = [a_b(state.x)]
    y = [a_b(state.y)]
    yaw = [a_b(np.sin(state.yaw))]
    v = [a_b(state.v)]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    # cyaw = smooth_yaw(cyaw)

    fig = plt.figure(figsize=(20, 10))  # Increase the overall graphic size
    # Create a large main window and three small status windows
    ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    ax1 = plt.subplot2grid((3, 3), (0, 2))
    ax2 = plt.subplot2grid((3, 3), (1, 2))
    ax3 = plt.subplot2grid((3, 3), (2, 2))
    
    count = 0
    img = np.zeros([1,1])
    cv2.imshow('img',img)

    # while MAX_TIME >= time:
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)
            # while not rospy.is_shutdown():
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = ai * 0.5
            _control.brake = 0

            # unit:radian -> the mpc output direction is opposite Carla's
            _control.steer = (- di) / MAX_STEER 

        vehicle_control_publisher.publish(_control)
        state.x = g_x
        state.y = g_y
        state.v = g_status.velocity
        state.yaw = -g_yaw
        print(state.x,state.y,state.v,state.yaw)

        time = time + DT

        x.append(a_b(state.x))
        y.append(a_b(state.y))
        yaw.append(a_b(np.sin((state.yaw))))
        v.append(a_b(state.v))
        t.append(a_b(time))
        d.append(a_b(di))
        a.append(a_b(ai))

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            ax_main.clear()
            # Main window drawing code
            if ox is not None:
                ax_main.plot(ox, oy, "xr", label="MPC")
            ax_main.plot(cx, cy, "-r", label="course")
            ax_main.plot(x, y, "ob", label="trajectory")
            ax_main.plot(xref[0, :], xref[1, :], "xk", label="xref")
            ax_main.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(ax_main, state.x, state.y, state.yaw, steer=di)
            ax_main.axis("equal")
            ax_main.grid(True)
            ax_main.set_title("Time[s]:" + str(round(time, 2))
                    + ", speed[km/h]:" + str(round(state.v * 3.6, 2))
                    + ", yaw[radian]:" + str(round(state.yaw, 2))
                    + ", accelerate[m/ss]:" + str(round(ai, 2))
                    + ", d_yaw[m/ss]:" + str(round(di, 2)))
            ax_main.legend()


            ax1.plot(t, yaw, "-r", label="yaw")
            ax1.grid(True)
            ax1.set_title('Fig 1.Yaw; Fig 2. d_Yaw; Fig 3. d_Accelerate')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Yaw [sin]")
            ax1.set_ylim(-1,1)

            ax2.plot(t, d, "-r", label="d_yaw")
            ax2.grid(True)
            ax2.set_title('')
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("yaw [rad]")

            ax3.plot(t, a, "-r", label="d_acc")
            ax3.grid(True)
            ax3.set_title('')
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("acc [m/ss]")

            plt.pause(0.0001)
            
            count +=1
        # if count > 150:
        #     break

    return t, x, y, yaw, v, d, a

def get_my_course2(dl): #curve
    ax = [383.8, 334.9]
    ay = [-326.9, -329.2]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def main():
    print(__file__ + " start!!")
    start = time.time()

    dl = 1.0  # course tick
    # cx, cy, cyaw, ck = get_my_course2(dl)
    cx, cy, cyaw, ck = get_straight_course2(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    #TODO: : check yaw angle_mod direction and inputs

    t, x, y, yaw, v, d, a = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)
    # t, x, y, yaw, v, d, a = do_sim_carla(cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))
        ax1.plot(t, yaw, "-r", label="yaw")
        ax1.grid(True)
        ax1.set_title('Yaw')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Yaw [rad]")

        ax2.plot(t, d, "-r", label="d_yaw")
        ax2.grid(True)
        ax2.set_title('yaw')
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("yaw [rad]")

        ax3.plot(t, yaw, "-r", label="d_acc")
        ax3.grid(True)
        ax3.set_title('acc')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("d_acc [m/ss]")

        plt.show()



