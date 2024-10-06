import os
import numpy as np
import matplotlib.pyplot as plt

def read_gps_data(file_path):
    latitudes = []
    longitudes = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                lat, lon = map(float, parts[:2])
                latitudes.append(lat)
                longitudes.append(lon)
    
    return np.array(latitudes), np.array(longitudes)

def plot_gps_data(latitudes, longitudes):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.plot(longitudes, latitudes, 'r-', linewidth=2, markersize=6)
    ax.plot(longitudes[0], latitudes[0], 'go', markersize=10, label='Start')
    ax.plot(longitudes[-1], latitudes[-1], 'bo', markersize=10, label='End')
    
    ax.set_title('GPS Data Visualization')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    
    selected_points = []
    point_markers = []
    
    def on_click(event):
        if event.inaxes == ax:
            # 找到最近的原始GPS点
            distances = np.sqrt((longitudes - event.xdata)**2 + (latitudes - event.ydata)**2)
            nearest_index = np.argmin(distances)
            nearest_lon, nearest_lat = longitudes[nearest_index], latitudes[nearest_index]
            
            if event.button == 1:  # 左键点击
                selected_points.append((nearest_lon, nearest_lat))
                marker, = ax.plot(nearest_lon, nearest_lat, 'yo', markersize=10)
                point_markers.append(marker)
                print(f"Point selected: {nearest_lon}, {nearest_lat}")
            elif event.button == 3:  # 右键点击
                for i, (lon, lat) in enumerate(selected_points):
                    if (lon, lat) == (nearest_lon, nearest_lat):
                        selected_points.pop(i)
                        point_markers[i].remove()
                        point_markers.pop(i)
                        print(f"Point removed: {lon}, {lat}")
                        break
            fig.canvas.draw()

    def on_scroll(event):
        # 实现缩放功能
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if event.button == 'up':
            scale_factor = 0.9
        elif event.button == 'down':
            scale_factor = 1.1
        else:
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.tight_layout()
    plt.show()
    
    return selected_points

def save_selected_points(points, file_path):
    if len(points) > 0:
        with open(file_path, 'w') as f:
            for lon, lat in points:
                f.write(f"{lat},{lon}\n")
        print(f"Selected points saved to {file_path}")
    else:
        print("No points selected. Output file not created.")

if __name__ == "__main__":
    input_file_path = '/workspace/src/perception/src/gps_log/gps_log_0.txt'
    output_file_path = '/workspace/src/perception/src/final_glob_waypoints_gnss.txt'
    
    if not os.path.exists(input_file_path):
        print(f"Error: File not found at {input_file_path}")
    else:
        latitudes, longitudes = read_gps_data(input_file_path)
        
        if len(latitudes) == 0 or len(longitudes) == 0:
            print("Error: No valid GPS data found in the file")
        else:
            selected_points = plot_gps_data(latitudes, longitudes)
            print(f"Total points plotted: {len(latitudes)}")
            print(f"Number of points selected: {len(selected_points)}")
            
            save_selected_points(selected_points, output_file_path)