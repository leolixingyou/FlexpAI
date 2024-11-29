"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Li Xingyou (@leolixingyou)

Ref: Atsushi Sakai (@Atsushi_twi)

"""
import cvxpy
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))


# from class_mpc importa *
from utils.angle_process import pi_2_pi
from utils.config_reader import load_mpc_parameters



show_animation = True


# vehicle state
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

def calculate_friction(state):
    mu_static = 0.02   # 静摩擦系数
    mu_dynamic = 0.015 # 动摩擦系数
    g = 9.81           # 重力加速度
    mass = 1500        # 假设车辆质量（kg）

    if abs(state.v) < 0.01:  # 认为速度接近零
        return 0  # 静止状态下，不产生加速度
    else:
        friction_force = mu_dynamic * mass * g
        friction_acc = friction_force / mass
        return np.sign(state.v) * friction_acc

def friction_difine(state):
    mu = 0.015  # Friction coeffecient
    g = 9.81    # Gravity acceleration

    # Calculating deceleration due to friction
    friction_acc = -mu * g * math.cos(state.yaw)  # Consider the influence of slope
    return friction_acc

def get_nparray_from_matrix(x):
    return np.array(x).flatten()


class MPC_Controller:
    def __init__(self, file_path) -> None:
        config_variable = load_mpc_parameters(file_path)

        self.n_x = config_variable['NX']  # x = x, y, v, yaw: system model
        self.n_u = config_variable['NU']  # a = [accel, steer]: output from system
        self.t = config_variable['T']  # horizon length
        self.dt = config_variable['DT']  # [s] time tick

        # mpc parameters
        self.R = config_variable['R']  # input cost matrix
        self.R_d = config_variable['Rd']  # input difference cost matrix
        self.Q = config_variable['Q']  # state cost matrix
        self.Q_f = config_variable['Qf']  # state final matrix

        # iterative parameter
        self.max_iter = config_variable['MAX_ITER']  # Max iteration
        self.du_th = config_variable['DU_TH']  # iteration finish param

        self.n_ind_search = config_variable['N_IND_SEARCH']  # Search index number

        self.max_steer = config_variable['MAX_STEER']  # maximum steering angle [rad]
        self.max_dsteer = config_variable['MAX_DSTEER']  # maximum steering speed [rad/s]
        self.max_speed = config_variable['MAX_SPEED']  # maximum speed [m/s]
        self.min_speed = config_variable['MIN_SPEED']  # minimum speed [m/s]
        self.max_accel = config_variable['MAX_ACCEL']  # maximum accel [m/ss]

        # Vehicle parameters
        self.length = config_variable['LENGTH']  # [m]
        self.width = config_variable['WIDTH']  # [m]
        self.backtowheel = config_variable['BACKTOWHEEL']  # [m]
        self.wheel_len = config_variable['WHEEL_LEN']  # [m]
        self.wheel_width = config_variable['WHEEL_WIDTH']  # [m]
        self.tread = config_variable['TREAD']  # [m]
        self.w_b = config_variable['WB']  # [m]

        self.goal_dis = config_variable['GOAL_DIS']  # goal distance
        self.stop_speed = config_variable['STOP_SPEED']  # stop speed

    def initial_state(self, cx, cy, cyaw, velocity = 0):
        state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=velocity)
        
        # initial yaw compensation
        if state.yaw - cyaw[0] >= math.pi:
            state.yaw -= math.pi * 2.0
        elif state.yaw - cyaw[0] <= -math.pi:
            state.yaw += math.pi * 2.0
        
        target_ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, 0)
        return state, target_ind

    def control_output(self, state, odelta, oa):
        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = self.update_state(state, ai, di)
        return di, ai


    def get_linear_model_matrix(self, v, phi, delta):

        A = np.zeros((self.n_x, self.n_x))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.dt * math.cos(phi)
        A[0, 3] = - self.dt * v * math.sin(phi)
        A[1, 2] = self.dt * math.sin(phi)
        A[1, 3] = self.dt * v * math.cos(phi)
        A[3, 2] = self.dt * math.tan(delta) / self.w_b

        B = np.zeros((self.n_x, self.n_u))
        B[2, 0] = self.dt
        B[3, 1] = self.dt * v / (self.w_b * math.cos(delta) ** 2)

        C = np.zeros(self.n_x)
        C[0] = self.dt * v * math.sin(phi) * phi
        C[1] = - self.dt * v * math.cos(phi) * phi
        C[3] = - self.dt * v * delta / (self.w_b * math.cos(delta) ** 2)

        return A, B, C

    def update_state(self, state, ai, delta):

        # input check
        if delta >= self.max_steer:
            delta = self.max_steer
        elif delta <= -self.max_steer:
            delta = -self.max_steer

        state.x = state.x + state.v * math.cos(state.yaw) * self.dt
        state.y = state.y + state.v * math.sin(state.yaw) * self.dt
        state.yaw = state.yaw + state.v / self.w_b * math.tan(delta) * self.dt

        ## TODO: Make correct model, These models are sample
        # friction_acc = friction_difine(state)
        friction_acc = calculate_friction(state)
        
        # Actual observed acceleration (command acceleration + friction)
        state.observed_acc = ai - friction_acc

        # Update velocity with observed acceleration
        state.v = state.v + state.observed_acc * self.dt

        if state.v > self.max_speed:
            state.v = self.max_speed
        elif state.v < self.min_speed:
            state.v = self.min_speed
        return state


    def calc_nearest_index(self, state, cx, cy, cyaw, pind):

        dx = [state.x - icx for icx in cx[pind:(pind + self.n_ind_search)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.n_ind_search)]]

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


    def predict_motion(self, x0, oa, od, xref):
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.t + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar


    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """
        MPC control with updating operational point iteratively
        """
        ox, oy, oyaw, ov = None, None, None, None

        if oa is None or od is None:
            oa = [0.0] * self.t
            od = [0.0] * self.t

        for i in range(self.max_iter):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.du_th:
                break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov


    def linear_mpc_control(self, xref, xbar, x0, dref):
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        x = cvxpy.Variable((self.n_x, self.t + 1))
        u = cvxpy.Variable((self.n_u, self.t))

        cost = 0.0
        constraints = []

        for t in range(self.t):
            cost += cvxpy.quad_form(u[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.t - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.R_d)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                self.max_dsteer * self.dt]

        cost += cvxpy.quad_form(xref[:, self.t] - x[:, self.t], self.Q_f)

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.max_speed]
        constraints += [x[2, :] >= self.min_speed]
        constraints += [cvxpy.abs(u[0, :]) <= self.max_accel]
        constraints += [cvxpy.abs(u[1, :]) <= self.max_steer]

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

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, dl, pind):
        xref = np.zeros((self.n_x, self.t + 1))
        dref = np.zeros((1, self.t + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self.t + 1):
            travel += abs(state.v) * self.dt
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


    ## for visualization
    def plot_car(self, ax_main, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

        outline = np.array([[-self.backtowheel, (self.length - self.backtowheel), (self.length - self.backtowheel), -self.backtowheel, -self.backtowheel],
                            [self.width / 2, self.width / 2, - self.width / 2, -self.width / 2, self.width / 2]])

        fr_wheel = np.array([[self.wheel_len, -self.wheel_len, -self.wheel_len, self.wheel_len, self.wheel_len],
                            [-self.wheel_width - self.tread, -self.wheel_width - self.tread, self.wheel_width - self.tread, self.wheel_width - self.tread, -self.wheel_width - self.tread]])

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                        [-math.sin(steer), math.cos(steer)]])

        fmr_wheel =             np.array([[self.wheel_len, -self.wheel_len, -self.wheel_len, self.wheel_len, self.wheel_len],
                            [-self.wheel_width , -self.wheel_width , self.wheel_width , self.wheel_width , -self.wheel_width ]])
        fmr_wheel = (fmr_wheel.T.dot(Rot2)).T
        fmr_wheel[0, :] += self.w_b
        fmr_wheel = (fmr_wheel.T.dot(Rot1)).T
        fmr_wheel[0, :] += x
        fmr_wheel[1, :] += y

        ax_main.plot(np.array(fmr_wheel[0, :]).flatten(),
                np.array(fmr_wheel[1, :]).flatten(), truckcolor)


        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += self.w_b
        fl_wheel[0, :] += self.w_b

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

        ax_main.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), truckcolor)
        ax_main.plot(np.array(rr_wheel[0, :]).flatten(),
                np.array(rr_wheel[1, :]).flatten(), truckcolor)
        ax_main.plot(np.array(rl_wheel[0, :]).flatten(),
                np.array(rl_wheel[1, :]).flatten(), truckcolor)
        ax_main.plot(x, y, "*")


if __name__ == '__main__':
    file_path = '/workspace/src/control/src/controller/config/carla_mpc_parameters.yaml'  # Assume the YAML file is named mpc_parameters.yaml
    config_variable = load_mpc_parameters(file_path)