import math
import numpy as np
from collections import deque

class GPSYawCalculator:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.gps_buffer = deque(maxlen=window_size)
        self.current_yaw = None

    def update(self, gps_x, gps_y):
        """
        Update GPS buffer and calculate current yaw
        
        :param gps_x: Current GPS x coordinate
        :param gps_y: Current GPS y coordinate
        :return: Currently calculated yaw (in radians)
        """
        self.gps_buffer.append((gps_x, gps_y))

        if len(self.gps_buffer) >= 2:
            return self._calculate_yaw()
        return None

    def _calculate_yaw(self):
        """
        Calculate yaw using GPS points in the buffer
        """
        if len(self.gps_buffer) < 2:
            return None

        # Calculate yaw using the latest two points
        x1, y1 = self.gps_buffer[-2]
        x2, y2 = self.gps_buffer[-1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        yaw = math.atan2(dy, dx)
        
        # If enough points, use more points to smooth yaw
        if len(self.gps_buffer) >= self.window_size:
            yaws = []
            for i in range(1, len(self.gps_buffer)):
                x1, y1 = self.gps_buffer[i-1]
                x2, y2 = self.gps_buffer[i]
                dx = x2 - x1
                dy = y2 - y1
                yaws.append(math.atan2(dy, dx))
            
            # Use circular mean to handle angle periodicity
            sin_sum = sum(math.sin(y) for y in yaws)
            cos_sum = sum(math.cos(y) for y in yaws)
            yaw = math.atan2(sin_sum, cos_sum)

        self.current_yaw = yaw
        return yaw

    def get_current_yaw(self):
        """
        Get the currently calculated yaw
        """
        return self.current_yaw

# # 使用示例
# yaw_calculator = GPSYawCalculator()

# # 模拟 GPS 输入
# gps_inputs = [
#     (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
#     (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)
# ]

# for gps_x, gps_y in gps_inputs:
#     yaw = yaw_calculator.update(gps_x, gps_y)
#     if yaw is not None:
#         print(f"GPS: ({gps_x}, {gps_y}), Yaw: {math.degrees(yaw):.2f} 度")