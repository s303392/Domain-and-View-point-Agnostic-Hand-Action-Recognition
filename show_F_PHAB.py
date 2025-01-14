import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

# Parse the animation and scaling flags
parser = argparse.ArgumentParser(description='Visualize hand skeleton data from TXT file')
parser.add_argument('--animate', action='store_true', help='Enable animation of hand skeleton')
args = parser.parse_args()

# Define joint names based on the order specified in the file
joint_names = [
    'Wrist', 
    'TMCP', 'IMCP', 'MMCP', 'RMCP', 'PMCP', 'TPIP', 'TDIP', 'TTIP',
    'IPIP', 'IDIP', 'ITIP', 'MPIP', 'MDIP', 'MTIP', 'RPIP', 'RDIP', 'RTIP',
    'PPIP', 'PDIP', 'PTIP'
]

# Load the data from the TXT file
data = []
with open('datasets/F-PHAB/Hand_pose_annotation_v1/Subject_1/charge_cell_phone/1/skeleton.txt', 'r') as f:
    for line in f:
        values = list(map(float, line.strip().split()))
        data.append(values)
data = np.array(data)

# Extract frame numbers and joint positions
frame_numbers = data[:, 0].astype(int)
joint_positions = data[:, 1:].reshape(-1, 21, 3)  # Reshape into (frames, joints, xyz)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Colors for different fingers
colors = {
    'T': 'b',  # Thumb
    'I': 'g',  # Index
    'M': 'r',  # Middle
    'R': 'c',  # Ring
    'P': 'm',  # Pinky
    'W': 'k'   # Wrist (black)
}

# Define the connections for a hand with 21 joints
connections = [
    (0, 1), (1, 6), (6, 7), (7, 8),  # Thumb
    (0, 2), (2, 9), (9, 10), (10, 11),  # Index
    (0, 3), (3, 12), (12, 13), (13, 14),  # Middle
    (0, 4), (4, 15), (15, 16), (16, 17),  # Ring
    (0, 5), (5, 18), (18, 19), (19, 20)  # Pinky
]

# Function to update the frame
def update_frame(num):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Hand Skeleton Frame {frame_numbers[num]}')

    # Extract joint coordinates for the current frame
    joints = joint_positions[num]
    xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]

    # Plot joints with different colors for each finger
    for i, joint_name in enumerate(joint_names):
        finger = joint_name[0]  # Get the first letter to identify the finger
        color = colors.get(finger, 'k')  # Default to black if finger not found
        size = 40 if joint_name.endswith('TIP') else 20  # Larger marker for fingertips
        ax.scatter(xs[i], ys[i], zs[i], c=color, marker='o', s=size)

    # Plot connections
    for start, end in connections:
        finger = joint_names[start][0]  # Get the first letter to identify the finger
        color = colors.get(finger, 'k')  # Default to black if finger not found
        ax.plot([xs[start], xs[end]],
                [ys[start], ys[end]],
                [zs[start], zs[end]], color=color)

if args.animate:
    # Create animation
    import matplotlib.animation as animation
    num_frames = len(joint_positions)
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=100, repeat=True)
    plt.show()
else:
    # Take the first frame for static visualization
    update_frame(0)
    plt.show()
