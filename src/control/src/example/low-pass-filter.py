import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from collections import deque

def low_pass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def constrained_low_pass_filter(data, target_value, max_deviation, cutoff, fs, order=5):
    centered_data = np.array(data) - np.mean(data) + target_value
    filtered_data = low_pass_filter(centered_data, cutoff, fs, order)
    constrained_data = np.clip(filtered_data, target_value - max_deviation, target_value + max_deviation)
    return constrained_data

# Your target array
target_array = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999987, 1.0, -0.4981259, 0.7735032, 0.9954741, -0.8593205, -0.242363, 0.7447899, 1.0, -0.1999593, -0.212246, 0.5686728, 1.0, -0.9999998, 0.1752875, 0.6193737, 0.6915901, -1.0, -0.5505096, 0.3989013, 0.6505796, 0.663408, -1.0, -0.6036994, 0.366427, 0.6346226, 0.6599511, -1.0, -0.6038968, 0.3765103, 0.6494528, 0.6767127, -1.0, -0.5967772, 0.3896974, 0.6627934, 0.6779546, -1.0, -0.5939355, 0.3936645, 0.664193, -1.0, -0.4979052, 0.469823, 0.7224448, 0.7265315, -1.0, -0.5735889, 0.3887409, 0.6374607, 0.6153727, -1.0, -0.6437245, 0.3205762, 0.5807664, -1.0, -0.5609054, -0.3274038, 0.522843, 1.0, -0.9999999, 0.2025846, 0.6512738, 0.7268476, -1.0, -0.504903, 0.4270431, 0.6630781, 0.6588576, -1.0, -0.6130187, 0.3490641, 0.6216284, 0.6563735, -1.0, -0.5874394, 0.3974725, 0.6763304, -0.9268859, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4029458, 0.633067, 0.6961131, 0.6373173, 0.5696234, 0.5739659, 0.6398991, 0.6025818, 0.5211708, 0.4343787, 0.3557268, 0.2887911, 0.2334224, 0.1882472, 0.1516409]

# Convert list to NumPy array
data = np.array(target_array)

# Set parameters
target_value = 0.0  # Target value
max_deviation = 1  # Maximum deviation
cutoff = 0.1  # Cutoff frequency
fs = 1  # Sampling frequency
window_size = 20  # Number of recent points to display

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# 1. Plot original data and its LPF result
full_lpf = constrained_low_pass_filter(data, target_value, max_deviation, cutoff, fs)
ax1.plot(data, 'b-', label='Original Data')
ax1.plot(full_lpf, 'r-', label='LPF Result')
ax1.set_title('Original Data vs Full LPF')
ax1.legend()
ax1.grid(True)

# 2. Plot LPF result for accumulated data
updated_list = []
accumulated_lpf = []
for i, value in enumerate(data):
    updated_list.append(value)
    if len(updated_list) > 20:
        lpf_result = constrained_low_pass_filter(updated_list, target_value, max_deviation, cutoff, fs)
        accumulated_lpf.append(lpf_result[-1])
    else:
        accumulated_lpf.append(value)

ax2.plot(data, 'b-', label='Original Data')
ax2.plot(accumulated_lpf, 'g-', label='Accumulated LPF')
ax2.set_title('LPF on Accumulated Data (>20 elements)')
ax2.legend()
ax2.grid(True)

# 3. Plot LPF result for sliding window
window_data = deque(maxlen=window_size)
window_lpf = []
for value in data:
    window_data.append(value)
    if len(window_data) == window_size:
        lpf_result = constrained_low_pass_filter(list(window_data), target_value, max_deviation, cutoff, fs)
        window_lpf.append(lpf_result[-1])
    else:
        window_lpf.append(value)

ax3.plot(data, 'b-', label='Original Data')
ax3.plot(window_lpf, 'm-', label='Window LPF')
ax3.set_title('LPF on Sliding Window (20 elements)')
ax3.legend()
ax3.grid(True)

# Set common labels
fig.text(0.5, 0.04, 'Sample Index', ha='center', va='center')
fig.text(0.06, 0.5, 'Value', ha='center', va='center', rotation='vertical')

# Adjust layout
plt.tight_layout()
plt.show()