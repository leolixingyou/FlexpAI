import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gnss_to_utm_converter import GNSStoUTMConverter

def read_txt(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    return content

def write_results_to_file(xs, ys, output_file):
    with open(output_file, 'w') as file:
        file.write("X coordinates:\n")
        file.write(", ".join(map(str, xs)) + "\n\n")
        file.write("Y coordinates:\n")
        file.write(", ".join(map(str, ys)))

def extract_coordinates(content):
    xs = []
    ys = []
    xys = []
    
    # Use regular expressions to match X and Y coordinates
    pattern = r"Location: X=([-\d.]+), Y=([-\d.]+)" # learn
    matches = re.findall(pattern, content)
    
    for match in matches:
        x, y = match
        xs.append(float(x))
        ys.append(float(y))
        xys.append([float(x),float(y)])
    
    return xs, ys, xys

def read_waypoints(waypoints_file):
    waypoints = []
    if not os.path.exists(waypoints_file):
        return waypoints

    with open(waypoints_file, 'r') as f:
        for line in f:
            lat, lon = map(float, line.strip().split(','))
            waypoints.append((lat, lon))
    return waypoints

def get_max_min_combination(xs, ys):
    x_max = max(xs)
    x_min = min(xs)
    y_max = max(ys)
    y_min = min(ys)
    x_center = (x_max + x_min) /2
    y_center = (y_max + y_min) /2
    
    width, height = x_max - x_min, y_max - y_min
    margin = 1/10
    margin_width, margin_height = np.dot([width, height], margin)
    
    x_min = x_min - margin_width 
    x_max = x_max + margin_width
    y_min = y_min - margin_height
    y_max = y_max + margin_height
    extended_width = x_max - x_min
    extended_height = y_max - y_min
    
    bounding_box = np.float32([[x_min, y_min], [x_min, y_max], [x_max, y_max]])
    return bounding_box, [extended_width, extended_height], [x_center, y_center]

def reformating(waypoints):
    waypoints = np.array(waypoints)
    xs = waypoints[:,0]
    ys = waypoints[:,1]
    return xs, ys

def create_grid_map(extended_size, scale=10000):
    x_scale = scale
    y_scale = (extended_size[1] / extended_size[0]) * x_scale
    glob_map = np.float32([[-x_scale, y_scale], [-x_scale, -y_scale],  [x_scale, -y_scale]])
    return glob_map 

def gps2globalMap(gps_location, Matrix_gps_globalMap):
    # Convert GPS coordinates to homogeneous coordinates
    gps_homogeneous = np.array([gps_location[0], gps_location[1], 1])
    
    # Apply affine transformation
    global_map_location = np.dot(Matrix_gps_globalMap, gps_homogeneous)
    
    return global_map_location[:2]  # Return x, y coordinates

def get_path_globalmap(xys, Matrix_gps_globalMap):
    global_map = [np.dot(Matrix_gps_globalMap, [x[0], x[1], 1]) for x in xys]
    return global_map

def get_utm_globalmap(xys, converter):
    global_map = [converter.convert(x[0], x[1]) for x in xys]
    return global_map

def check_points_plot(coordinates):
    x_coords, y_coords = zip(*coordinates)

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'bo-')

    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')

    plt.title('Coordinate Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.grid(True)
    plt.legend()
    plt.show()



# file_path = "/workspace/src/perception/src/utils/selected_spawn_points.txt"
# txt_contents = read_txt(file_path)
# xs, ys, xys = extract_coordinates(txt_contents)


file_path = "/workspace/src/perception/src/local_final_global_waypoints/final_glob_waypoints_gnss.txt"
waypoints = read_waypoints(file_path)
xs, ys = reformating(waypoints)

bounding_box, extended_size, center_point = get_max_min_combination(xs, ys)
glob_map = create_grid_map(extended_size)
# the number of points for affine are three 
Matrix_gps_globalMap = cv2.getAffineTransform(bounding_box, glob_map)

# gps_location = center_point
# location_globalMap = gps2globalMap(gps_location, Matrix_gps_globalMap)
# global_map = get_path_globalmap(waypoints, Matrix_gps_globalMap)

converter_gps2utm = GNSStoUTMConverter()
global_map = get_utm_globalmap(waypoints, converter_gps2utm)
def save_arrays_to_file(arrays, filename):
    with open(filename, 'w') as f:
        for arr in arrays:
            f.write(f"{arr[0]},{arr[1]}\n")




filename = "/workspace/src/perception/src/array_data_globalmap_utm.txt"
save_arrays_to_file(global_map, filename)
print(f"Data saved to {filename}")

print()

# output_file_path = "/workspace/src/perception/src/result.txt"
# write_results_to_file(xs, ys, output_file_path)
# print(f"Results have been written to {output_file_path}")