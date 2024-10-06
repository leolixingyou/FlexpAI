#!/usr/bin/env python3

import rospy
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class GlobalPathPublisher:
    def __init__(self):
        rospy.init_node('global_path_planner', anonymous=False)
        self.path_pub = rospy.Publisher('/Ctrl_CV/planning/global_path', Path, queue_size=10)
        self.rate = rospy.Rate(1)  # 1 Hz
        self.waypoints_file = '/workspace/src/perception/src/local_final_global_waypoints/final_glob_waypoints_gnss.txt'

    def read_waypoints(self):
        waypoints = []
        if not os.path.exists(self.waypoints_file):
            rospy.logerr(f"Waypoints file not found: {self.waypoints_file}")
            return waypoints

        with open(self.waypoints_file, 'r') as f:
            for line in f:
                lat, lon = map(float, line.strip().split(','))
                waypoints.append((lat, lon))
        return waypoints

    def create_path_msg(self, waypoints):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"  # Assuming the coordinates are in the map frame

        for lat, lon in waypoints:
            pose = PoseStamped()
            pose.pose.position.x = lon  # Using longitude as x
            pose.pose.position.y = lat  # Using latitude as y
            pose.pose.position.z = 0.0  # Assuming 2D path
            pose.pose.orientation.w = 1.0  # Default orientation
            path_msg.poses.append(pose)

        return path_msg

    def run(self):
        waypoints = self.read_waypoints()
        if not waypoints:
            rospy.logerr("No waypoints found. Exiting.")
            return

        path_msg = self.create_path_msg(waypoints)

        while not rospy.is_shutdown():
            self.path_pub.publish(path_msg)
            rospy.loginfo("Published global path")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        publisher = GlobalPathPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass