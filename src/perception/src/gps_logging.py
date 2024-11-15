#!/usr/bin/env python3
import numpy as np
import math
import rospy
from ros_tools import *
import os
from math import radians, sin, cos, sqrt, atan2
from gnss_to_utm_converter import GNSStoUTMConverter
from transforms3d.euler import quat2euler

gnssconverter = GNSStoUTMConverter()

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def get_unique_filename(base_path, base_name):
    """Generate a unique filename by adding a number if the file already exists."""
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 0
    while os.path.exists(os.path.join(directory, f"{name}{ext}")):
        name = f"{base_name}_{counter}"
        counter += 1
    
    return os.path.join(directory, f"{name}{ext}")

def create_log_directory(base_path):
    """Create a gps_log directory if it doesn't exist."""
    log_dir = os.path.join(os.path.dirname(base_path), "gps_log")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir



class GPS_Logging_Manager:
    def __init__(self, platform) -> None:
        rospy.init_node('GPS_Logging', anonymous=False)
        self.rate = rospy.Rate(100) # 100hz

        sensors = SensorConfig(platform)
        self.gps_listener = GPS_GNSS_Listener(sensors.sensor_config['Gps'])
        self.odem_listener = Odem_Listener(sensors.sensor_config['Odem'])
        self.current_lat, self.current_lon = 0, 0

    def log_gps_data(self, locations, save_path):
        # Get the last location data
        
        """"""
        # check here to select datas
        # ## gnss to utm
        # easting, northing, _ = gnssconverter.convert(locations[-1][0], locations[-1][1])
        # last_location = [easting, northing]
        
        # ## gnss and 10000 times
        # last_location = [x * 10000 for x in locations[-1]]

        ## odem with location and orientation
        last_location_pose = locations[-1].pose.pose
        last_location_position = last_location_pose.position
        last_location_orientation = last_location_pose.orientation
        _, _, yaw = quat2euler(
            [last_location_orientation.w,
            last_location_orientation.x,
            last_location_orientation.y,
            last_location_orientation.z])
        yaw_degree = math.degrees(yaw)
        
        last_location = [last_location_position.x, last_location_position.y, last_location_position.z, yaw_degree]

        """"""

        # Check if file exists, create it if not
        if not os.path.exists(save_path):
            open(save_path, 'a').close()
        
        # Read the last saved data
        with open(save_path, 'r') as f:
            lines = f.readlines()
            last_saved = lines[-1].strip().split(',') if lines else None
        
        # Convert current location to string
        current_location_str = ','.join(map(str, (last_location)))
        
        # Check distance if we have a previous location
        if last_saved and len(last_saved) >= 2:

            ## odem distance -> euclidian
            distance = np.sqrt((np.square(last_location[0] - self.current_lat) + np.square(last_location[1] - self.current_lon))) 
            
            ## gnss distance -> haversin 
            # last_lat, last_lon = map(float, last_saved[:2])
            # current_lat, current_lon = last_location[:2]
            # distance = haversine_distance(last_lat, last_lon, current_lat, current_lon)
            
            # Save if distance is greater than 5 meters
            if distance > 2: # odem
            # if distance > 5: # gnss
                with open(save_path, 'a') as f:
                    self.current_lat, self.current_lon = last_location[0], last_location[1]
                    f.write(current_location_str + '\n')
                print(f"New GPS data saved: {current_location_str}, Distance: {distance:.2f}m")
            else:
                print(f"GPS data unchanged (distance: {distance:.2f}m), not saving")
        else:
            # If it's the first data point, save it
            with open(save_path, 'a') as f:
                f.write(current_location_str + '\n')
            print(f"First GPS data saved: {current_location_str}")

    def run(self, save_path):
        while not rospy.is_shutdown():
            # self.gps_listener.gathering_msg()
            # if self.gps_listener.data_received:
            #     locations = self.gps_listener.datas
            #     self.log_gps_data(locations, save_path)

            self.odem_listener.gathering_msg()
            if self.odem_listener.data_received:
                locations = self.odem_listener.datas
                self.log_gps_data(locations, save_path)


            self.rate.sleep()


if __name__ == "__main__":
    platform = 'carla'
    # base_path = '/workspace/src/perception/src/gps_log.txt'
    base_path = '/workspace/src/perception/src/odm_x_y_yaw_abs_log.txt'
    
    # Create gps_log directory
    log_dir = create_log_directory(base_path)
    
    # Get unique filename
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    unique_path = get_unique_filename(os.path.join(log_dir, filename), name)
    
    print(f"Logging GPS data to: {unique_path}")
    
    test_temp = GPS_Logging_Manager(platform)
    test_temp.run(unique_path)