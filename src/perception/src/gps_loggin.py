#!/usr/bin/env python3
import numpy as np
import rospy
from ros_tools import *
import os
from math import radians, sin, cos, sqrt, atan2

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

def log_gps_data(locations, save_path):
    # Get the last location data
    last_location = locations[-1]
    
    # Check if file exists, create it if not
    if not os.path.exists(save_path):
        open(save_path, 'a').close()
    
    # Read the last saved data
    with open(save_path, 'r') as f:
        lines = f.readlines()
        last_saved = lines[-1].strip().split(',') if lines else None
    
    # Convert current location to string
    current_location_str = ','.join(map(str, last_location))
    
    # Check distance if we have a previous location
    if last_saved and len(last_saved) >= 2:
        last_lat, last_lon = map(float, last_saved[:2])
        current_lat, current_lon = last_location[:2]
        distance = haversine_distance(last_lat, last_lon, current_lat, current_lon)
        
        # Save if distance is greater than 5 meters
        if distance > 5:
            with open(save_path, 'a') as f:
                f.write(current_location_str + '\n')
            print(f"New GPS data saved: {current_location_str}, Distance: {distance:.2f}m")
        else:
            print(f"GPS data unchanged (distance: {distance:.2f}m), not saving")
    else:
        # If it's the first data point, save it
        with open(save_path, 'a') as f:
            f.write(current_location_str + '\n')
        print(f"First GPS data saved: {current_location_str}")

class GPS_Logging_Manager:
    def __init__(self, platform) -> None:
        rospy.init_node('GPS_Logging', anonymous=False)
        self.rate = rospy.Rate(100) # 100hz

        sensors = SensorConfig(platform)
        self.gps_listener = GPS_GNSS_Listener(sensors.sensor_config['Gps'])

    def run(self, save_path):
        while not rospy.is_shutdown():
            self.gps_listener.gathering_msg()
            if self.gps_listener.data_received:
                locations = self.gps_listener.datas
                log_gps_data(locations, save_path)
            self.rate.sleep()

if __name__ == "__main__":
    platform = 'carla'
    base_path = '/workspace/src/perception/src/gps_log.txt'
    
    # Create gps_log directory
    log_dir = create_log_directory(base_path)
    
    # Get unique filename
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    unique_path = get_unique_filename(os.path.join(log_dir, filename), name)
    
    print(f"Logging GPS data to: {unique_path}")
    
    test_temp = GPS_Logging_Manager(platform)
    test_temp.run(unique_path)