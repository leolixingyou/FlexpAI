
import math
import rospy
import numpy as np

from simple_pid import PID  # pylint: disable=import-error,wrong-import-order

from ackermann_msgs.msg import AckermannDrive  # pylint: disable=import-error,wrong-import-order
from std_msgs.msg import Header # pylint: disable=wrong-import-order
from carla_msgs.msg import CarlaEgoVehicleStatus  # pylint: disable=no-name-in-module,import-error
from carla_msgs.msg import CarlaEgoVehicleControl  # pylint: disable=no-name-in-module,import-error
from carla_msgs.msg import CarlaEgoVehicleInfo  # pylint: disable=no-name-in-module,import-error
from carla_ackermann_msgs.msg import EgoVehicleControlInfo  # pylint: disable=no-name-in-module,import-error

from transforms3d.euler import quat2euler

def set_target_steering_angle(target_steering_angle):
    """
    set target sterring angle
    """
    info.target.steering_angle = -target_steering_angle
    if abs(info.target.steering_angle) > info.restrictions.max_steering_angle:
        rospy.logerr("Max steering angle reached, clipping value")
        info.target.steering_angle = np.clip(
            info.target.steering_angle,
            -info.restrictions.max_steering_angle,
            info.restrictions.max_steering_angle)

def set_target_speed(target_speed):
    """
    set target speed
    """
    if abs(target_speed) > info.restrictions.max_speed:
        rospy.logerr("Max speed reached, clipping value")
        info.target.speed = np.clip(
            target_speed, -info.restrictions.max_speed, info.restrictions.max_speed)
    else:
        info.target.speed = target_speed
    info.target.speed_abs = abs(info.target.speed)

def set_target_accel(target_accel):
    """
    set target accel
    """
    epsilon = 0.00001
    # if speed is set to zero, then use max decel value
    if info.target.speed_abs < epsilon:
        info.target.accel = -info.restrictions.max_decel
    else:
        info.target.accel = np.clip(
            target_accel, -info.restrictions.max_decel, info.restrictions.max_accel)

def set_target_jerk(target_jerk):
    """
    set target accel
    """
    info.target.jerk = target_jerk

def get_time():
   return rospy.get_time()

def get_param(name, alternative_value=None):
    if name.startswith('/'):
        raise RuntimeError("Only private parameters are supported.")
    return rospy.get_param("~" + name, alternative_value)

def cb_ackermann_command_updated():
    """
    Stores the ackermann drive message for the next controller calculation

    :param ros_ackermann_drive: the current ackermann control input
    :type ros_ackermann_drive: ackermann_msgs.AckermannDrive
    :return:
    """
    global  last_ackermann_msg_received_sec
    last_ackermann_msg_received_sec = get_time()
    # set target values
    set_target_steering_angle(ros_ackermann_drive.steering_angle)
    set_target_speed(ros_ackermann_drive.speed)
    set_target_accel(ros_ackermann_drive.acceleration)
    set_target_jerk(ros_ackermann_drive.jerk)

def vehicle_status_updated(msg):
    global  vehicle_status
    vehicle_status = msg

def vehicle_info_updated(vehicle_info, info):
    """
    Stores the ackermann drive message for the next controller calculation

    :param ros_ackermann_drive: the current ackermann control input
    :type ros_ackermann_drive: ackermann_msgs.AckermannDrive
    :return:
    """

    # calculate restrictions
    info.restrictions.max_steering_angle = get_vehicle_max_steering_angle(
        vehicle_info)
    info.restrictions.max_speed = get_vehicle_max_speed(
        vehicle_info)
    info.restrictions.max_accel = get_vehicle_max_acceleration(
        vehicle_info)
    info.restrictions.max_decel = get_vehicle_max_deceleration(
        vehicle_info)
    info.restrictions.min_accel = get_param('min_accel', 1.)
    # clipping the pedal in both directions to the same range using the usual lower
    # border: the max_accel to ensure the the pedal target is in symmetry to zero
    info.restrictions.max_pedal = min(
        info.restrictions.max_accel, info.restrictions.max_decel)
    return info, vehicle_info

## sub Function for Main Loop
def get_msg_header():
    """
    Get a filled ROS message header
    :return: ROS message header
    :rtype: std_msgs.msg.Header
    """
    header = Header()
    header.frame_id = "map"
    header.stamp = ros_timestamp(sec=get_time(), from_sec=True)
    return header

def ros_timestamp(sec=0, nsec=0, from_sec=False):
    if from_sec:
        return rospy.Time.from_sec(sec)
    return rospy.Time(int(sec), int(nsec))

def control_steering(info):
    """
    Basic steering control
    """
    info.output.steer = info.target.steering_angle / \
        info.restrictions.max_steering_angle
    return info

def control_stop_and_reverse(info):
    """
    Handle stop and switching to reverse gear
    """
    # from this velocity on it is allowed to switch to reverse gear
    standing_still_epsilon = 0.1
    # from this velocity on hand brake is turned on
    full_stop_epsilon = 0.00001

    # auto-control of hand-brake and reverse gear
    info.output.hand_brake = False
    if info.current.speed_abs < standing_still_epsilon:
        # standing still, change of driving direction allowed
        info.status.status = "standing"
        if info.target.speed < 0:
            if not info.output.reverse:
                rospy.loginfo(
                    "VehicleControl: Change of driving direction to reverse")
                info.output.reverse = True
        elif info.target.speed > 0:
            if info.output.reverse:
                rospy.loginfo(
                    "VehicleControl: Change of driving direction to forward")
                info.output.reverse = False
        if info.target.speed_abs < full_stop_epsilon:
            info.status.status = "full stop"
            info.status.speed_control_accel_target = 0.
            info.status.accel_control_pedal_target = 0.
            set_target_speed(0.)
            info.current.speed = 0.
            info.current.speed_abs = 0.
            info.current.accel = 0.
            info.output.hand_brake = True
            info.output.brake = 1.0
            info.output.throttle = 0.0

    elif np.sign(info.current.speed) * np.sign(info.target.speed) == -1:
        # requrest for change of driving direction
        # first we have to come to full stop before changing driving
        # direction
        rospy.loginfo("VehicleControl: Request change of driving direction."
                    " v_current={} v_desired={}"
                    " Set desired speed to 0".format(info.current.speed,
                                                    info.target.speed))
        set_target_speed(0.)
    return info

def run_speed_control_loop(info):
    """
    Run the PID control loop for the speed

    The speed control is only activated if desired acceleration is moderate
    otherwhise we try to follow the desired acceleration values

    Reasoning behind:

    An autonomous vehicle calculates a trajectory including position and velocities.
    The ackermann drive is derived directly from that trajectory.
    The acceleration and jerk values provided by the ackermann drive command
    reflect already the speed profile of the trajectory.
    It makes no sense to try to mimick this a-priori knowledge by the speed PID
    controller.
    =>
    The speed controller is mainly responsible to keep the speed.
    On expected speed changes, the speed control loop is disabled
    """
    epsilon = 0.00001
    target_accel_abs = abs(info.target.accel)
    ### xingyou: info.restrictions -> vehicle info from vehicle_info_updated
    ### xingyou: Speed control activation for active speed keeping mode-> pid
    if target_accel_abs < info.restrictions.min_accel:
        if info.status.speed_control_activation_count < 5:
            info.status.speed_control_activation_count += 1
    else:
        if info.status.speed_control_activation_count > 0:
            info.status.speed_control_activation_count -= 1
    # set the auto_mode of the controller accordingly
    speed_controller.auto_mode = info.status.speed_control_activation_count >= 5

    if speed_controller.auto_mode:
        speed_controller.setpoint = info.target.speed_abs
        info.status.speed_control_accel_delta = float(speed_controller(
            info.current.speed_abs))

        # clipping borders
        clipping_lower_border = -target_accel_abs
        clipping_upper_border = target_accel_abs
        # per definition of ackermann drive: if zero, then use max value
        if target_accel_abs < epsilon:
            clipping_lower_border = -info.restrictions.max_decel
            clipping_upper_border = info.restrictions.max_accel
        info.status.speed_control_accel_target = np.clip(
            info.status.speed_control_accel_target +
            info.status.speed_control_accel_delta,
            clipping_lower_border, clipping_upper_border)
    else:
        info.status.speed_control_accel_delta = 0.
        info.status.speed_control_accel_target = info.target.accel
    return info

def run_accel_control_loop(info):
    """
    Run the PID control loop for the acceleration
    """
    # setpoint of the acceleration controller is the output of the speed controller
    accel_controller.setpoint = info.status.speed_control_accel_target
    info.status.accel_control_pedal_delta = float(accel_controller(
        info.current.accel))
    # @todo: we might want to scale by making use of the the abs-jerk value
    # If the jerk input is big, then the trajectory input expects already quick changes
    # in the acceleration; to respect this we put an additional proportional factor on top
    info.status.accel_control_pedal_target = np.clip(
        info.status.accel_control_pedal_target +
        info.status.accel_control_pedal_delta,
        -info.restrictions.max_pedal, info.restrictions.max_pedal)
    return info

def update_drive_vehicle_control_command(info, vehicle_info, vehicle_status):
    """
    Apply the current speed_control_target value to throttle/brake commands
    """

    # the driving impedance moves the 'zero' acceleration border
    # Interpretation: To reach a zero acceleration the throttle has to pushed
    # down for a certain amount
    info.status.throttle_lower_border = get_vehicle_driving_impedance_acceleration(
        vehicle_info, vehicle_status, info.output.reverse)

    # the engine lay off acceleration defines the size of the coasting area
    # Interpretation: The engine already prforms braking on its own;
    #  therefore pushing the brake is not required for small decelerations
    info.status.brake_upper_border = info.status.throttle_lower_border + \
        get_vehicle_lay_off_engine_acceleration(vehicle_info)

    if info.status.accel_control_pedal_target > info.status.throttle_lower_border:
        info.status.status = "accelerating"
        info.output.brake = 0.0
        # the value has to be normed to max_pedal
        # be aware: is not required to take throttle_lower_border into the scaling factor,
        # because that border is in reality a shift of the coordinate system
        # the global maximum acceleration can practically not be reached anymore because of
        # driving impedance
        info.output.throttle = (
            (info.status.accel_control_pedal_target -
                info.status.throttle_lower_border) /
            abs(info.restrictions.max_pedal))
    elif info.status.accel_control_pedal_target > info.status.brake_upper_border:
        info.status.status = "coasting"
        # no control required
        info.output.brake = 0.0
        info.output.throttle = 0.0
    else:
        info.status.status = "braking"
        # braking required
        info.output.brake = (
            (info.status.brake_upper_border -
                info.status.accel_control_pedal_target) /
            abs(info.restrictions.max_pedal))
        info.output.throttle = 0.0

    # finally clip the final control output (should actually never happen)
    info.output.brake = np.clip(
        info.output.brake, 0., 1.)
    info.output.throttle = np.clip(
        info.output.throttle, 0., 1.)
    return info

def get_vehicle_driving_impedance_acceleration(vehicle_info, vehicle_status, reverse):
    rolling_resistance_force = get_rolling_resistance_force(vehicle_info)
    aerodynamic_drag_force = get_aerodynamic_drag_force(vehicle_status)
    slope_force = get_slope_force(vehicle_info, vehicle_status)
    if reverse:
        slope_force = -slope_force
    deceleration = -(rolling_resistance_force +
                     aerodynamic_drag_force +
                     slope_force) /  \
        get_vehicle_mass(vehicle_info)

    return deceleration

def get_weight_force(vehicle_info):
    weight = get_vehicle_mass(vehicle_info) * \
        get_acceleration_of_gravity(vehicle_info)

    return weight

def get_rolling_resistance_force(vehicle_info):

    rolling_resistance_coefficient = 0.01
    normal_force = get_weight_force(vehicle_info)

    rolling_resistance_force = rolling_resistance_coefficient * normal_force

    return rolling_resistance_force

def get_aerodynamic_drag_force(vehicle_status):
    """
    Calculate the aerodynamic drag force of a carla vehicle

    :param vehicle_status: the ego vehicle status
    :type vehicle_status: carla_ros_bridge.CarlaEgoVehicleStatus
    :return: aerodynamic drag force [N]
    :rtype: float64
    """
    # see also https://en.wikipedia.org/wiki/Automobile_drag_coefficient
    default_aerodynamic_drag_coefficient = 0.3
    default_drag_reference_area = 2.37
    # @todo currently not provided in vehicle_info
    drag_area = default_aerodynamic_drag_coefficient * default_drag_reference_area
    rho_air_25 = 1.184
    speed_squared = vehicle_status.velocity * vehicle_status.velocity

    aerodynamic_drag_force = 0.5 * drag_area * rho_air_25 * speed_squared
    return aerodynamic_drag_force

def get_slope_force(vehicle_info, vehicle_status):
    """
    Calculate the force of a carla vehicle faces when driving on a slope.

    :param vehicle_info: the vehicle info
    :type vehicle_info: carla_ros_bridge.CarlaEgoVehicleInfo
    :param vehicle_status: the ego vehicle status
    :type vehicle_status: carla_ros_bridge.CarlaEgoVehicleStatus
    :return: slope force [N, >0 uphill, <0 downhill]
    :rtype: float64
    """
    dummy_roll, pitch, dummy_yaw = quat2euler(
        [vehicle_status.orientation.w, vehicle_status.orientation.x,
         vehicle_status.orientation.y, vehicle_status.orientation.z])
    slope_force = get_acceleration_of_gravity(
        vehicle_info) * get_vehicle_mass(vehicle_info) * math.sin(-pitch)
    return slope_force

def get_acceleration_of_gravity(_):
    acceleration = 9.81
    return acceleration

def get_vehicle_mass(vehicle_info):
    mass = 1500.0
    if vehicle_info.mass:
        mass = vehicle_info.mass
    return mass

def get_vehicle_lay_off_engine_acceleration(vehicle_info):
    """
    Calculate the acceleration a carla vehicle faces by the engine on lay off

    This respects the following forces:
    - engine brake force

    :param vehicle_info: the vehicle info
    :type vehicle_info: carla_ros_bridge.CarlaEgoVehicleInfo
    :return: acceleration the vehicle [m/s^2 < 0]
    :rtype: float64
    """
    return -get_engine_brake_force(vehicle_info) / get_vehicle_mass(vehicle_info)

def get_engine_brake_force(_):
    return 500.0

def get_vehicle_mass(vehicle_info):
    mass = 1500.0
    if vehicle_info.mass:
        mass = vehicle_info.mass

    return mass

def get_vehicle_max_steering_angle(vehicle_info):
    """
    Get the maximum steering angle of a carla vehicle

    :param vehicle_info: the vehicle info
    :type vehicle_info: carla_ros_bridge.CarlaEgoVehicleInfo
    :return: maximum steering angle [radians]
    :rtype: float64
    """
    # 70 degrees is the default max steering angle of a car
    max_steering_angle = math.radians(70)
    # get max steering angle (use smallest non-zero value of all wheels)
    for wheel in vehicle_info.wheels:
        if wheel.max_steer_angle:
            if wheel.max_steer_angle and wheel.max_steer_angle < max_steering_angle:
                max_steering_angle = wheel.max_steer_angle
    return max_steering_angle

def get_vehicle_max_speed(_):
    # 180 km/h is the default max speed of a car
    max_speed = 180.0 / 3.6

    return max_speed

def get_vehicle_max_acceleration(_):
    max_acceleration = 3.0

    return max_acceleration

def get_vehicle_max_deceleration(_):
    max_deceleration = 8.0
    return max_deceleration

## Main loop Function
def update_current_values():
    current_time_sec = get_time()
    delta_time = current_time_sec - info.current.time_sec
    current_speed = vehicle_status.velocity

    if delta_time > 0:
    ### xingyou: updated speed info -> info.current.speed default 0
        delta_speed = current_speed - info.current.speed
        current_accel = delta_speed / delta_time
        # average filter
        info.current.accel = (info.current.accel * 4 + current_accel) / 5
    info.current.time_sec = current_time_sec
    info.current.speed = current_speed
    info.current.speed_abs = abs(current_speed)
    return info

def vehicle_control_cycle(info, vehicle_info, vehicle_status):
    """
    Perform a vehicle control cycle and sends out CarlaEgoVehicleControl message
    """
    # perform actual control
    ### xingyou: control_steering-> normalize the steering angel(current / max)
    info = control_steering(info)
    ### xingyou: set stop and reverse speed 즉 우선순위 높음
    info = control_stop_and_reverse(info)
    ### xingyou: 
    info = run_speed_control_loop(info)
    ### xingyou: 
    info = run_accel_control_loop(info)
    if not info.output.hand_brake:
        info = update_drive_vehicle_control_command(info, vehicle_info, vehicle_status)

        # only send out the Carla Control Command if AckermannDrive messages are
        # received in the last second (e.g. to allows manually controlling the vehicle)
        if (last_ackermann_msg_received_sec + 1.0) > \
                get_time():
            info.output.header = get_msg_header()
            carla_control_publisher.publish(info.output)
    return info

# from ego vehicle
def send_ego_vehicle_control_info_msg(info):
    """
    Function to send carla_ackermann_control.msg.EgoVehicleControlInfo message.

    :return:
    """
    info.header = get_msg_header()
    control_info_publisher.publish(info)

## main Loop
def run():
    while not rospy.is_shutdown():
        def loop(timer_event=None):
            info = update_current_values()
            info = vehicle_control_cycle(info, vehicle_info, vehicle_status)
            info = send_ego_vehicle_control_info_msg(info)

        ## control rate
        control_loop_rate = 0.5
        rospy.Timer(rospy.Duration(control_loop_rate), loop)


if __name__ == '__main__':
    rospy.init_node('carla_ackermann_control')
    ## Subscriber
    # **** ackermann drive commands to Modify this msg to control ****
    rospy.Subscriber('/carla/ego_vehicle/ackermann_cmd', AckermannDrive, cb_ackermann_command_updated)
    # current status of the vehicle
    rospy.Subscriber('/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, cb_ackermann_command_updated)
    # vehicle info
    rospy.Subscriber('/carla/ego_vehicle/vehicle_info', CarlaEgoVehicleInfo, cb_ackermann_command_updated)

    ## Publisher
    # to send command to carla
    carla_control_publisher = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)
    # report controller info
    control_info_publisher = rospy.Publisher('/carla/ego_vehicle/ackermann_control/control_info', EgoVehicleControlInfo, queue_size=1)


    ## Initialization
    # PID 
    speed_controller = PID(Kp=get_param("speed_Kp", alternative_value=0.05),
                                Ki=get_param("speed_Ki", alternative_value=0.),
                                Kd=get_param("speed_Kd", alternative_value=0.5),
                                sample_time=0.05,
                                output_limits=(-1., 1.))
    accel_controller = PID(Kp=get_param("accel_Kp", alternative_value=0.05),
                                Ki=get_param("accel_Ki", alternative_value=0.),
                                Kd=get_param("accel_Kd", alternative_value=0.05),
                                sample_time=0.05,
                                output_limits=(-1, 1))

    # use the correct time for further calculations
    PID._current_time = (       # pylint: disable=protected-access
        lambda: get_time())

    from carla_ackermann_control.cfg import EgoVehicleControlParameterConfig # pylint: disable=no-name-in-module,import-error,ungrouped-imports
    from dynamic_reconfigure.server import Server # pylint: disable=no-name-in-module,import-error
    def reconfigure_pid_parameters(ego_vehicle_control_parameter, _level):
        """
        Callback for dynamic reconfigure call to set the PID parameters

        :param ego_vehicle_control_parameter:
        :type ego_vehicle_control_parameter: \
            carla_ackermann_control.cfg.EgoVehicleControlParameterConfig

        """
        rospy.loginfo("Reconfigure Request:  "
                        "speed ({speed_Kp}, {speed_Ki}, {speed_Kd}), "
                        "accel ({accel_Kp}, {accel_Ki}, {accel_Kd})"
                        "".format(**ego_vehicle_control_parameter))
        speed_controller.tunings = (
            ego_vehicle_control_parameter['speed_Kp'],
            ego_vehicle_control_parameter['speed_Ki'],
            ego_vehicle_control_parameter['speed_Kd'],
        )
        accel_controller.tunings = (
            ego_vehicle_control_parameter['accel_Kp'],
            ego_vehicle_control_parameter['accel_Ki'],
            ego_vehicle_control_parameter['accel_Kd'],
        )
        return ego_vehicle_control_parameter


    reconfigure_server = Server(
        EgoVehicleControlParameterConfig,
        namespace="",
        callback=reconfigure_pid_parameters,
    )

    ## udpate ros infos
    vehicle_status = CarlaEgoVehicleStatus()
    vehicle_info = CarlaEgoVehicleInfo()
    info = EgoVehicleControlInfo()

    role_name = get_param('role_name', 'ego_vehicle')
    control_loop_rate = get_param("control_loop_rate", 0.05)
    last_ackermann_msg_received_sec =  get_time()

    # set initial maximum values
    info, vehicle_info= vehicle_info_updated(vehicle_info, info)

    ## Info Definition
    # target values
    info.target.steering_angle = 0.
    info.target.speed = 0.
    info.target.speed_abs = 0.
    info.target.accel = 0.
    info.target.jerk = 0.

    # current values
    info.current.time_sec = get_time()
    info.current.speed = 0.
    info.current.speed_abs = 0.
    info.current.accel = 0.

    # control values
    info.status.status = 'n/a'
    info.status.speed_control_activation_count = 0
    info.status.speed_control_accel_delta = 0.
    info.status.speed_control_accel_target = 0.
    info.status.accel_control_pedal_delta = 0.
    info.status.accel_control_pedal_target = 0.
    info.status.brake_upper_border = 0.
    info.status.throttle_lower_border = 0.

    # control output
    info.output.throttle = 0.
    info.output.brake = 1.0
    info.output.steer = 0.
    info.output.reverse = False
    info.output.hand_brake = True

    run()















