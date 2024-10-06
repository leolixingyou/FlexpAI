#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class GlobalPathSubscriber:
    def __init__(self):
        """
        Initialize the GlobalPathSubscriber.
        """
        rospy.init_node('global_path_subscriber', anonymous=True)
        self.path_sub = rospy.Subscriber('/Ctrl_CV/planning/global_path', Path, self.path_callback)
        self.current_path = None
        self.path_received = False

    def path_callback(self, msg):
        """
        Callback function for the global path topic.

        Args:
        msg (Path): The received Path message.
        """
        self.current_path = msg
        self.path_received = True
        rospy.loginfo("Received new global path with %d waypoints", len(msg.poses))

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

    def run(self):
        """
        Main loop for the GlobalPathSubscriber.
        """
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.path_received:
                # You can add any periodic processing here
                waypoints = self.get_waypoints()
                rospy.loginfo("Current path has %d waypoints", len(waypoints))
                self.path_received = False  # Reset the flag
            rate.sleep()

if __name__ == '__main__':
    try:
        subscriber = GlobalPathSubscriber()
        subscriber.run()
    except rospy.ROSInterruptException:
        pass
