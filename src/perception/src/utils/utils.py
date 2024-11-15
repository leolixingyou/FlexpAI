"""
The utils file for perception
Author: Li Xingyou
"""

def reduce_course(n, mode):
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

def init_course(dl, mode): #curve
    ax, ay = reduce_course(2, mode)   
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        corrinate_transformation(ax), corrinate_transformation(ay), ds=dl)
    return cx, cy, cyaw, ck, [ax[0], ay[0]]

