#!/usr/bin/env python3
import cv2
import carla
import matplotlib.pyplot as plt
import numpy as np

def visualize_spawn_points_on_map(spawn_transforms, map_image):
    # Extract coordinates and rotations
    coords = np.array([[t.location.x, -t.location.y, t.location.z] for t in spawn_transforms])
    rotations = np.array([[t.rotation.pitch, t.rotation.roll, t.rotation.yaw] for t in spawn_transforms])

    # Normalize coordinates to image size
    height, width = map_image.shape[:2]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    x_normalized = ((coords[:, 0] - x_min) / (x_max - x_min) * width).astype(int)
    y_normalized = ((coords[:, 1] - y_min) / (y_max - y_min) * height).astype(int)
    y_normalized = height - y_normalized  # Flip y-axis

    # Normalize z coordinates for color mapping
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    z_normalized = (coords[:, 2] - z_min) / (z_max - z_min)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(map_image)

    # Plot points with color based on z-coordinate
    scatter = ax.scatter(x_normalized, y_normalized, c=z_normalized, cmap='viridis', s=30)

    # Add a colorbar
    plt.colorbar(scatter, label='Height (normalized)')

    # Plot arrows for direction
    for idx, (x, y, yaw) in enumerate(zip(x_normalized, y_normalized, rotations[:, 2])):
        dx = np.cos(np.radians(yaw))
        dy = -np.sin(np.radians(yaw))  # Negative because y-axis is flipped
        ax.arrow(x, y, dx*10, dy*10, color='r', width=1, head_width=5)
        ax.text(x + 3, y - 3, str(idx), fontsize=6, color='white', 
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
        
    ax.set_title('CARLA Spawn Points with Direction on Town Map')
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.show()

# Connect to CARLA and get spawn points
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world() # Town 01 
current_map = world.get_map()
spawn_transforms = current_map.get_spawn_points()

# Load the map image
map_image = cv2.imread('/workspace/src/perception/src/utils/Carla_Maps_Town01.png')
map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Visualize the spawn points on the map
visualize_spawn_points_on_map(spawn_transforms, map_image)