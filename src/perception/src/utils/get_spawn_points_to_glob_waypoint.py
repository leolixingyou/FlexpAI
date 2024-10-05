import cv2
import carla
import matplotlib.pyplot as plt
import numpy as np

class SpawnPointSelector:
    def __init__(self, spawn_transforms, map_image):
        self.spawn_transforms = spawn_transforms
        self.map_image = map_image
        self.selected_points = []

        # Extract coordinates and rotations
        self.coords = np.array([[t.location.x, -t.location.y, t.location.z] for t in spawn_transforms])
        self.rotations = np.array([[t.rotation.pitch, t.rotation.roll, t.rotation.yaw] for t in spawn_transforms])

        # Normalize coordinates
        height, width = map_image.shape[:2]
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
        
        self.x_normalized = ((self.coords[:, 0] - x_min) / (x_max - x_min) * width).astype(int)
        self.y_normalized = ((self.coords[:, 1] - y_min) / (y_max - y_min) * height).astype(int)
        self.y_normalized = height - self.y_normalized  # Flip y-axis

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.map_image)
        self.plot_points()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)  # Add scroll event

    def plot_points(self):
        self.ax.clear()
        self.ax.imshow(self.map_image)
        self.ax.scatter(self.x_normalized, self.y_normalized, c='r', s=20)
        for idx, (x, y) in enumerate(zip(self.x_normalized, self.y_normalized)):
            self.ax.text(x + 2, y - 2, str(idx), fontsize=6, color='white', 
                         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0.2))

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        distances = np.sqrt((self.x_normalized - event.xdata)**2 + (self.y_normalized - event.ydata)**2)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] < 10:  # Threshold for clicking near a point
            self.selected_points.append(nearest_idx)
            print(f"Selected point {nearest_idx}")
            self.save_to_file(nearest_idx)

    def on_scroll(self, event):
        # Zoom in/out with mouse scroll
        base_scale = 1.1
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.ax.figure.canvas.draw()

    def save_to_file(self, idx):
        with open('/workspace/src/perception/src/utils/selected_spawn_points.txt', 'a') as f:
            transform = self.spawn_transforms[idx]
            f.write(f"Index: {idx}\n")
            f.write(f"Location: X={transform.location.x:.2f}, Y={transform.location.y:.2f}, Z={transform.location.z:.2f}\n")
            f.write(f"Rotation: Pitch={transform.rotation.pitch:.2f}, Yaw={transform.rotation.yaw:.2f}, Roll={transform.rotation.roll:.2f}\n")
            f.write("\n")

    def run(self):
        plt.show()

# Connect to CARLA and get spawn points
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
current_map = world.get_map()
spawn_transforms = current_map.get_spawn_points()

# Load the map image
map_image = cv2.imread('/workspace/src/perception/src/utils/Carla_Maps_Town01.png')
map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

# Create and run the selector
selector = SpawnPointSelector(spawn_transforms, map_image)
selector.run()