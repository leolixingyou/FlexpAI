# TODO:  Clear the params 
from carla_controller_setup import Carla_Controller_MPC
from mpc_controller import calc_speed_profile, TARGET_SPEED,


def main(mode):
    print(__file__ + " start!!")

    cx, cy, cyaw, ck, sp, dl,  initial_position = mpc_init(mode)[-1]

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_position = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    # initial_position = State(x=-290, y=0.2, yaw=cyaw[0], v=0.0)

    do_simulation(cx, cy, cyaw, ck, sp, dl, initial_position)

def main_carla(mode):
    config_file = '/workspace/src/base_io/src/carla_bridge/objects.json'
    vehicle_listener = Carla_Controller_MPC()
    vehicle_listener.mpc_for_carla(config_file)
    print(__file__ + " start!!")

    fig = plt.figure(figsize=(10, 5))  # Increase the overall graphic size
    # Create a large main window and three small status windows
    ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    ax1 = plt.subplot2grid((3, 3), (0, 2))
    ax2 = plt.subplot2grid((3, 3), (1, 2))
    ax3 = plt.subplot2grid((3, 3), (2, 2))
    # ax_main = plt.subplot()
    count = 0

    init_flag = False

    while not rospy.is_shutdown():
        g_x, g_y, g_yaw = vehicle_listener.g_x, vehicle_listener.g_y, vehicle_listener.g_yaw
        if all([g_x, g_y, g_yaw,vehicle_listener.wheel_max_angle]) and vehicle_listener.get_info:
            # print('yeah')
            wheel_max_angle = vehicle_listener.wheel_max_angle
            speed = vehicle_listener.status.velocity



            if not init_flag:
                state, target_ind, goal, [cx, cy, cyaw, ck, sp, dl,  initial_pose] = mpc_init(
                    mode, x_state=g_x, y_state=g_y,yaw_state=g_yaw,velocity=speed
                )

                time = 0.0
                x = deque([state.x], maxlen= 50)
                y = deque([state.y], maxlen= 50)
                yaw = deque([state.yaw], maxlen= 50)
                v = deque([state.v], maxlen= 50)
                t = deque([0.0], maxlen= 50)
                d = deque([0.0], maxlen= 50)
                a = deque([0.0], maxlen= 50)

                odelta, oa = None, None

                cyaw = smooth_yaw(cyaw)
                _control = CarlaEgoVehicleControl()
                init_flag = True

            else:
                state.x = g_x - initial_pose[0]
                state.y = g_y - initial_pose[1]
                state.yaw = g_yaw
                state.v = speed

                xref, target_ind, dref = calc_ref_trajectory(
                    state, cx, cy, cyaw, ck, sp, dl, target_ind)

                x0 = [state.x, state.y, state.v, state.yaw]  # current state

                oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
                    xref, x0, dref, oa, odelta)
                
                
                di, ai = 0.0, 0.0
                if odelta is not None:
                    di, ai = odelta[0], oa[0]
                    state = update_state(state, ai, di)
                    _control = acc_brake_with_velocity(_control, ai, di, wheel_max_angle, cx, state, goal, target_ind)
                    vehicle_listener.vehicle_control_publisher.publish(_control)
                _control = acc_brake_with_velocity()

                time = time + DT

                x.append(state.x)
                y.append(state.y)
                yaw.append(state.yaw)
                v.append(state.v)
                t.append(time)
                d.append(di)
                a.append(ai)

                if mode == 'finite':
                    if check_goal(state, goal, target_ind, len(cx)):
                        print("Goal")
                        break


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
                    plot_car(ax_main, state.x, state.y, state.yaw, steer=di)
                    ax_main.axis("equal")
                    ax_main.grid(True)
                    ax_main.set_title("Time[s]:" + str(round(time, 2))
                            + ", speed[km/h]:" + str(round(state.v*3.6, 2))
                            + ", yaw[radian]:" + str(round(state.yaw, 2))
                            + ", steer[radian]:" + str(round(di, 2)))
                    ax_main.set_xlim(state.x-10, state.x+10)
                    ax_main.set_ylim(state.y-10, state.y+10)

                    ax1.clear()
                    if ox is not None:
                        ax1.plot(ox, oy, "xr", label="MPC")
                    ax1.plot(cx, cy, "-r", label="course")
                    ax1.plot(x, y, "ob", label="trajectory")
                    ax1.plot(xref[0, :], xref[1, :], "xk", label="xref")
                    ax1.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                    ax1.grid(True)
                    ax1.set_title('Map')

                    ax2.plot(t, d, "-r", label="yaw")
                    ax2.grid(True)
                    ax2.set_title('')
                    ax2.set_xlabel("qTime [s]")
                    ax2.set_ylabel("yaw [rad]")

                    ax3.plot(t, a, "-r", label="acc")
                    ax3.grid(True)
                    ax3.set_title('')
                    ax3.set_xlabel("Time [s]")
                    ax3.set_ylabel("pedal [m/ss]")
                
                    plt.pause(0.0001)
        rospy.Rate(60).sleep
    