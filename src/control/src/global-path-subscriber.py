#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class EnhancedGlobalPathSubscriber:
    def __init__(self):
        """
        Initialize the EnhancedGlobalPathSubscriber.
        """
        rospy.init_node('enhanced_global_path_subscriber', anonymous=True)
        self.path_sub = rospy.Subscriber('/Ctrl_CV/planning/global_path', Path, self.path_callback)
        self.current_path = None
        self.previous_path = None
        self.path_received = False

    def path_callback(self, msg):
        """
        Callback function for the global path topic.

        Args:
        msg (Path): The received Path message.
        """
        self.previous_path = self.current_path
        self.current_path = msg
        self.path_received = True
        rospy.loginfo("Received new global path with %d waypoints", len(msg.poses))
        self.check_path_change()

    def get_current_path(self):
        """
        Get the most recently received path.

        Returns:
        Path: The current path, or None if no path has been received.
        """
        return self.current_path

    def get_waypoints(self):
        """
        Extract waypoints from the current path.

        Returns:
        list: A list of (x, y) tuples representing waypoints, or an empty list if no path is available.
        """
        if self.current_path is None:
            return []
        return [(pose.pose.position.x, pose.pose.position.y) for pose in self.current_path.poses]

    def check_path_change(self):
        """
        Check if the new path is different from the previous path.
        """
        if self.previous_path is None:
            print("1")  # Initial path received
            return

        current_waypoints = self.get_waypoints()
        previous_waypoints = [(pose.pose.position.x, pose.pose.position.y) for pose in self.previous_path.poses]

        if len(current_waypoints) != len(previous_waypoints):
            print("1")  # Path changed: Different number of waypoints
            return

        # Convert to numpy arrays for efficient comparison
        current_array = np.array(current_waypoints)
        previous_array = np.array(previous_waypoints)

        if not np.allclose(current_array, previous_array, atol=1e-6):
            print("1")  # Path changed: Waypoints are different
        else:
            print("0")  # Path unchanged

    def run(self):
        """
        Main loop for the EnhancedGlobalPathSubscriber.
        """
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.path_received:
                waypoints = self.get_waypoints()
                rospy.loginfo("Current path has %d waypoints", len(waypoints))
                self.path_received = False  # Reset the flag
            rate.sleep()

if __name__ == '__main__':
    try:
        subscriber = EnhancedGlobalPathSubscriber()
        subscriber.run()
    except rospy.ROSInterruptException:
        pass