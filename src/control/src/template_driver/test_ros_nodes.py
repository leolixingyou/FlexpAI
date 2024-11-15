#!/usr/bin/env python3
import rospy

""" 
rospy.init_node should define once in one python script
"""
class Carla_Command:
    def __init__(self) -> None:
        rospy.init_node('Carla_Command_1', anonymous=False)
        self.d_1 = Carla_Test_1()
        self.d_2 = Carla_Test_2()
        self.d_3 = Carla_Test_3()


class Carla_Test_1:
    def __init__(self) -> None:
        rospy.init_node('Carla_Test_1', anonymous=False)

class Carla_Test_2:
    def __init__(self) -> None:
        rospy.init_node('Carla_Test_2', anonymous=False)

class Carla_Test_3:
    def __init__(self) -> None:
        rospy.init_node('Carla_Test_3', anonymous=False)


if __name__ == "__main__":
    test_temp = Carla_Command()
    test_temp.run()