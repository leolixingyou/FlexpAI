
import rospy
from rospy.exceptions import ROSInterruptException
from rospy import ServiceException

from src.base_io.src.carla_bridge.carla_spawn_objects_modified import My_CarlaSpawnObjects, cleanup_ego_vehicle
import ros_compatibility  as roscomp


def main(args=None):
    cleanup_ego_vehicle()
    rospy.sleep(2)
    """
    main function
    """
    # roscomp.init("spawn_objects", args=args)
    
    spawn_objects_node = None
    try:
        spawn_objects_node = My_CarlaSpawnObjects()
        roscomp.on_shutdown(spawn_objects_node.destroy)
    except KeyboardInterrupt:
        roscomp.logerr("Could not initialize My_CarlaSpawnObjects. Shutting down.")

    if spawn_objects_node:
        try:
            spawn_objects_node.spawn_objects()
            try:
                spawn_objects_node.spin()
            except (ROSInterruptException, ServiceException, KeyboardInterrupt):
                pass
        except (ROSInterruptException, ServiceException, KeyboardInterrupt):
            spawn_objects_node.logwarn(
                "Spawning process has been interrupted. There might be actors that have not been destroyed properly")
        except RuntimeError as e:
            roscomp.logfatal("Exception caught: {}".format(e))
        finally:
            roscomp.shutdown()
    print('destory')
    # roscomp.on_shutdown(spawn_objects_node.destroy)
    cleanup_ego_vehicle()


if __name__ == '__main__':
    rospy.init_node('get_actor')
    main()
