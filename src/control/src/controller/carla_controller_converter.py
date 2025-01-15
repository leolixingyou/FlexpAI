"""

Accelerate to Pedal, Brake, and Steer

author: Li Xingyou (@leolixingyou)

"""

# Input is CarlaEgoVehicleControl() saved in _control
def acc_brake_with_velocity(_control, ai, di, wheel_max_angle, max_accel):

        # unit:radian -> the mpc output direction is opposite Carla's
        _control.steer = -(di / wheel_max_angle)

        rate_throttle = abs(ai)/max_accel

        if ai > 0:
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = rate_throttle 
            _control.brake = 0

        elif ai < 0:
            _control.gear = 1
            _control.reverse = _control.gear < 0
            _control.throttle = 0 
            _control.brake = rate_throttle

        return _control