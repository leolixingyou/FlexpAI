"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Li Xingyou (@leolixingyou)

Ref: Atsushi Sakai (@Atsushi_twi)

"""
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.planning.src.PathPlanning.CubicSpline import cubic_spline_planner
from speed_profile import calc_speed_profile

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

def corrinate_transformation(v):
    v_array = np.array(v)
    v_array -= v_array[0]
    return v_array

def global_path_generation(dl, mode): #curve
    ax, ay = target_waypoints_input(2, mode)
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        corrinate_transformation(ax), corrinate_transformation(ay), ds=dl)
    return cx, cy, cyaw, ck, [ax[0], ay[0]]

def global_path_with_speed_profile(mode, target_speed, x_state=0, y_state=0, yaw_state=0, velocity=0):
    dl = 1.0  # course tick
    cx, cy, cyaw, ck, initial_pose = global_path_generation(dl, mode)
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)
    return cx, cy, cyaw, sp