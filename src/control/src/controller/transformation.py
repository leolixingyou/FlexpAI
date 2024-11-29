"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Li Xingyou (@leolixingyou)

Ref: Atsushi Sakai (@Atsushi_twi)

"""
import numpy as np
from waypoints_io import target_waypoints_input

def corrinate_transformation(v):
    v_array = np.array(v)
    v_array -= v_array[0]
    return v_array

def global_path_generation(dl, mode): #curve
    ax, ay = target_waypoints_input(2, mode)
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        corrinate_transformation(ax), corrinate_transformation(ay), ds=dl)
    return cx, cy, cyaw, ck, [ax[0], ay[0]]

def global_path_with_speed_profile(mode, target_speed):
    dl = 1.0  # course tick
    cx, cy, cyaw, ck, initial_pose = global_path_generation(dl, mode)
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)
    return cx, cy, cyaw, 