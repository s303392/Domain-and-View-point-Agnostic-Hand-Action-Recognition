import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

# Define the input file path directly in the code
input_file = 'datasets/common_pose/SHREC2017/gesture1_finger2_subject10_essai4_train_jn20.txt' #grab
#input_file = 'datasets/common_pose/mySHREC-17/grab03_LucaTracker_L_combined_skeleton.txt'
#input_file = 'datasets/common_pose/SHREC2017/gesture11_finger2_subject14_essai1_train_jn20.txt' #11 label

# Parse the animation flag
parser = argparse.ArgumentParser(description='Visualize common pose data from TXT file')
parser.add_argument('--animate', action='store_true', help='Enable animation of common pose')
args = parser.parse_args()

# Define joint names based on the order specified in the file
joint_names = [
    'Wrist', 
    'TMCP', 'TDIP', 'TTIP', 
    'IMCP', 'IPIP', 'IDIP', 'ITIP', 
    'MMCP', 'MPIP', 'MDIP', 'MTIP', 
    'RMCP', 'RPIP', 'RDIP', 'RTIP', 
    'PMCP', 'PPIP', 'PDIP', 'PTIP'
]

# Load the data from the TXT file
data = []
with open(input_file, 'r') as f:
    for line in f:
        values = list(map(float, line.strip().split()))
        data.append(values)
data = np.array(data)

# Extract frame numbers and joint positions
frame_numbers = np.arange(len(data))
joint_positions = data.reshape(-1, 20, 3)  # Reshape into (frames, joints, xyz)

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

# Define the connections for a hand with 20 joints
connections = [
    (0, 1), (1, 2), (2, 3),  # Thumb
    (0, 4), (4, 5), (5, 6), (6, 7),  # Index
    (0, 8), (8, 9), (9, 10), (10, 11),  # Middle
    (0, 12), (12, 13), (13, 14), (14, 15),  # Ring
    (0, 16), (16, 17), (17, 18), (18, 19)  # Pinky
]

# Function to update the frame
def update_frame(num):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Common Pose Frame {frame_numbers[num]}')

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