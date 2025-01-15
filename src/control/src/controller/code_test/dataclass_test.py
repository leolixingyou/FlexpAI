"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Li Xingyou (@leolixingyou)

Ref: Atsushi Sakai (@Atsushi_twi)

"""
from dataclasses import dataclass

@dataclass
class Mine_Map:
    x_init: float = 0.0 # initial x coordinate on General coordinates [m]
    y_init: float = 0.0 # initial y coordinate on General coordinates [m]

    x_sensor: float = 0.0 # sensor x coordinate on General coordinates [m]
    y_sensor: float = 0.0 # sensor y coordinate on General coordinates [m]

    @property
    def x_cur(self) -> float: # current x coordinate on General coordinates [m]
        return self.x_sensor - self.x_init

    @property
    def y_cur(self) -> float: # current y coordinate on General coordinates [m]
        return self.y_sensor - self.y_init
    

if __name__ == '__main__':
    map = Mine_Map(x_init=1.0, y_init=2.0, x_sensor=5.0, y_sensor=7.0)
    print(f"x_cur: {map.x_cur}, y_cur: {map.y_cur}")

    map.y_sensor = 10.0
    print(f"Updated y_cur: {map.y_cur}")