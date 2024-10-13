
import rospy
from rospy.exceptions import ROSInterruptException
from rospy import ServiceException

from src.base_io.src.carla_bridge.carla_spawn_objects_modified import My_CarlaSpawnObjects, cleanup_ego_vehicle
import ros_compatibility  as roscomp

def main(config_file, args=None):
    cleanup_ego_vehicle(config_file)
    rospy.sleep(2)
    """
    main function
    """
    # roscomp.init("spawn_objects", args=args)
    
    spawn_objects_node = None
    spawn_objects_node = My_CarlaSpawnObjects(config_file)

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
    cleanup_ego_vehicle(config_file)

def spawn_ego_vehicle(config_file):
    cleanup_ego_vehicle(config_file)
    rospy.sleep(2)
    
    spawn_objects_node = None
    spawn_objects_node = My_CarlaSpawnObjects(config_file)

    if spawn_objects_node:
        spawn_objects_node.spawn_objects()

def clean_ego_vehicle(config_file):
    cleanup_ego_vehicle(config_file)

if __name__ == '__main__':
    config_file = '/workspace/src/base_io/src/carla_bridge/objects.json'
    rospy.init_node('get_actor')
    spawn_ego_vehicle(config_file)
