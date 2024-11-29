"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Li Xingyou (@leolixingyou)

Ref: Atsushi Sakai (@Atsushi_twi)

"""

from collections import deque
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.planning.src.PathPlanning.CubicSpline import cubic_spline_planner
from speed_profile import calc_speed_profile
from mpc_controller import MPC_Controller
from utils.angle_process import smooth_yaw

show_animation = True

class Coordinates_Transformation:
    def __init__(self,) -> None:
        pass
    
    def position_initial(self, ax, ay):
        self.init_position = [ax[0], ay[0]]
        self.init_ax, self.init_ay = self.world2my_trans(ax, ay)

    def world2my_trans(self, ax, ay):
        ax_array = np.zeros(len(ax))
        ay_array = np.zeros(len(ay))
        for i, position in enumerate(zip(ax, ay)):
            x, y = self.coor_trans_position(position)
            ax_array[i] = x
            ay_array[i] = y
        return ax_array, ay_array

    def coor_trans_position(self, cur_position):
        """
        position: [x, y]
        """
        return [cur_position[0] - self.init_position[0], cur_position[1] - self.init_position[1]]

## global path input
def target_waypoints_input(n, mode):
    with open('/workspace/src/control/src/example/odm_x_y_full_course_town05_round.txt', 'r') as file:
        data = file.readlines()
    data_temp = [x for i,x in enumerate(data) if i % n ==0 ]
    info_temp = [x.split(',')[:2] for x in data_temp]
    data_float = []
    for info in info_temp:
        float_info = [round(float(x),2) for x in info]
        data_float.append(float_info)
    data_int = np.array(data_float)
    ax, ay = data_int[:,0], data_int[:,1]
    if mode == 'infinity':
        ax, ay = list(ax) * 100, list(ay) * 100, 
    return ax, ay

## path generation
def path_generation(coor_trans, dl): #curve
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        coor_trans.init_ax, coor_trans.init_ay, ds=dl)
    return cx, cy, cyaw, ck

## path with speed profile
def path2trajectroy(coor_trans, target_speed, dl):
    cx, cy, cyaw, ck = path_generation(coor_trans, dl)
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)
    return cx, cy, cyaw, ck, sp

## Check_goal
def check_goal(state, goal, tind, nind, c_mpc):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= c_mpc.goal_dis)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= c_mpc.stop_speed)

    if isgoal and isstop:
        return True

    return False

if __name__ == '__main__':
    ## main 

    # I/O
    ## set driving mode
    mode = 'finite'

    ## set goal point
    goal = [11, 22]

    ## initial map information
    ax, ay = target_waypoints_input(2, mode)
    
    ## set initial param
    time = 0.0 # initial time
    dl = 1.0  # course tick
    maximum_speed = 10 # km/h


    # Perception
    ## map information transformation
    coor_trans = Coordinates_Transformation()
    coor_trans.position_initial(ax, ay)


    # Planning
    ## path generation
    target_speed = maximum_speed
    cx, cy, cyaw, ck, sp = path2trajectroy(coor_trans, target_speed, dl) 

    ## update goal point
    updated_goal = [cx[-1], cy[-1]]

    ## state initial
    file_path = '/workspace/src/control/src/controller/config/carla_mpc_parameters.yaml'  # Assume the YAML file is named mpc_parameters.yaml
    c_mpc = MPC_Controller(file_path)
    state, target_ind = c_mpc.initial_state(cx, cy, cyaw, velocity=0)

    odelta, oa = None, None

    ## Solution for problem of sudden changes in yaw angle during continuous measurement
    cyaw = smooth_yaw(cyaw)


    # visualization
    fig = plt.figure(figsize=(10, 5))  # Increase the overall graphic size
    # Create a large main window and three small status windows
    ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    ax1 = plt.subplot2grid((3, 3), (0, 2))
    ax2 = plt.subplot2grid((3, 3), (1, 2))
    ax3 = plt.subplot2grid((3, 3), (2, 2))

    x = deque([state.x], maxlen= 50)
    y = deque([state.y], maxlen= 50)
    yaw = deque([state.yaw], maxlen= 50)
    v = deque([state.v], maxlen= 50)
    t = deque([.0], maxlen= 50)
    d = deque([.0], maxlen= 50)
    a = deque([.0], maxlen= 50)

    while 1:

        xref, target_ind, dref = c_mpc.calc_ref_trajectory(
            state, cx, cy, cyaw, sp, dl, target_ind)
        
        x0 = [state.x, state.y, state.v, state.yaw]  # current state
        
        oa, odelta, ox, oy, oyaw, ov = c_mpc.iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        di, ai = c_mpc.control_output(state, odelta, oa)

        if updated_goal != None:
            if check_goal(state, updated_goal, target_ind, len(cx), c_mpc):
                print("Goal")
                break

        time = time + c_mpc.dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if show_animation:  # pragma: no cover
            # for stopping simulation with the esc key.
            # ax_main.gcf().canvas.mpl_connect('key_release_event',
            #         lambda event: [exit(0) if event.key == 'escape' else None])
            ax_main.clear()
            if ox is not None:
                ax_main.plot(ox, oy, "xr", label="MPC")
            ax_main.plot(cx, cy, "-r", label="course")
            ax_main.plot(x, y, "ob", label="trajectory")
            ax_main.plot(xref[0, :], xref[1, :], "xk", label="xref")
            ax_main.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            c_mpc.plot_car(ax_main, state.x, state.y, state.yaw, steer=di)
            ax_main.axis("equal")
            ax_main.grid(True)
            ax_main.set_title("Time[s]:" + str(round(time, 2))
                    + ", speed[km/h]:" + str(round(state.v*3.6, 2))
                    + ", yaw[radian]:" + str(round(state.yaw, 2))
                    + ", steer[radian]:" + str(round(di, 2)))
            ax_main.set_xlim(state.x-10, state.x+10)

            ax1.plot(t, yaw, "-r", label="yaw")
            ax1.grid(True)
            ax1.set_title('Fig 1.G_Yaw; Fig 2. V_Yaw; Fig 3. V_Accelerate')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Yaw [sin]")
            ax1.set_ylim(-1,1)

            ax2.plot(t, d, "-r", label="d_yaw")
            ax2.grid(True)
            ax2.set_title('')
            ax2.set_xlabel("qTime [s]")
            ax2.set_ylabel("yaw [rad]")

            ax3.plot(t, a, "-r", label="d_acc")
            ax3.grid(True)
            ax3.set_title('')
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("acc [m/ss]")
        


            plt.pause(0.0001)

