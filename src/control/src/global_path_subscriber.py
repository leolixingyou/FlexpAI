#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path

"""
State transition table
           +---   Changed Path   ----
           |                        |
           v                        ^
2-> path-> 1 -> no changing path -> 0
^          v                        v
|          |                        |
+---Reset--+------------------------+

"""

class EnhancedGlobalPathSubscriber:
    def __init__(self):
        """
        Initialize the EnhancedGlobalPathSubscriber.
        """
        rospy.Subscriber('/Ctrl_CV/planning/global_path', Path, self.path_callback)
        self.current_path = None
        self.previous_path = None
        self.path_received = False
        self.path_changed = False
        self.path_changed = 2 # means no information

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
            self.path_changed = 1
            return

        current_waypoints = self.get_waypoints()
        previous_waypoints = [(pose.pose.position.x, pose.pose.position.y) for pose in self.previous_path.poses]

        if len(current_waypoints) != len(previous_waypoints):
            print("1")  # Path changed: Different number of waypoints
            self.path_changed = 1
            return

        # Convert to numpy arrays for efficient comparison
        current_array = np.array(current_waypoints)
        previous_array = np.array(previous_waypoints)

        if not np.allclose(current_array, previous_array, atol=1e-6):
            print("1")  # Path changed: Waypoints are different
            self.path_changed = 1
        else:
            print("0")  # Path unchanged
            self.path_changed = 0

    def reset(self):
        self.path_changed = 2
        self.current_path = None
        self.previous_path = None
        self.path_received = False
        
    def run(self):
        """
        Main loop for the EnhancedGlobalPathSubscriber.
        """
        rospy.init_node('enhanced_global_path_subscriber', anonymous=False)
        rate = rospy.Rate(1)  # 10 Hz
        while not rospy.is_shutdown():
            if self.path_received:
                waypoints = self.get_waypoints()
                rospy.loginfo("Current path has %d waypoints", len(waypoints))
                self.path_received = False  # Reset the flag
                self.check_path_change()
            rate.sleep()

if __name__ == '__main__':
    try:
        subscriber = EnhancedGlobalPathSubscriber()
        subscriber.run()
    except rospy.ROSInterruptException:
        pass