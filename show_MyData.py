# Import necessary libraries
import pandas as pd
import numpy as np  # Required for numerical operations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import matplotlib.animation as animation
import sys  # Import sys to access command-line arguments

try:
    df = pd.read_csv(
        #'datasets/mySHREC-17/raw_data/9_rotation_ccw/rot_ccw02_LucaTracker_L.csv',
        #'datasets/mySHREC-17/raw_data/15_swipe_left/SL_01_tracker_blenderCorSys_2025-01-23_17-30-43_LucaTracker_L.csv',
        'datasets/myDataset/MANUS_data/pickplace/dx/pickplace_dx_Luca040_30-01_LucaTracker_R.csv',
        #'hand_global_positions.csv',
        sep=';',
        decimal=','
    )
    print("File letto con successo")
except pd.errors.ParserError as e:
    print(f"Errore durante la lettura del file CSV: {e}")

# Remove the first 3 columns
df = df.iloc[:, 3:]

# Separate position and rotation columns
position_columns = [col for col in df.columns if '.1' not in col and '_W' not in col]
rotation_columns = [col for col in df.columns if '.1' in col or '_W' in col]

# Ask the user if they want to use rotations
use_rotations = input("Vuoi utilizzare anche le rotazioni dei giunti? (s/n): ").lower() == 's'

expected_columns = len(df.columns)
for i, row in df.iterrows():
    if len(row) != expected_columns:
        print(f"Riga {i} ha un numero di colonne diverso: {len(row)}")

# List of joints to extract
joints = [
    'Hand',
    # Thumb joints
    'Thumb_CMC', 'Thumb_MCP', 'Thumb_DIP', 'Thumb_TIP',
    # Index finger joints
    'Index_CMC', 'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
    # Middle finger joints
    'Middle_CMC', 'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
    # Ring finger joints
    'Ring_CMC', 'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
    # Pinky finger joints
    'Pinky_CMC', 'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP'
]

# Get the list of columns
columns = df.columns.tolist()

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to extract joint positions for a given frame
def get_joint_positions(frame_data):
    joint_positions = {}
    
    # Extract hand position directly
    hand_position = np.array([
        frame_data['Hand_X'],
        frame_data['Hand_Y'],
        frame_data['Hand_Z']
    ])
    joint_positions['Hand'] = hand_position

    for joint in joints:
        if joint == 'Hand':
            continue  # Already handled
        x_col = f'{joint}_X'
        y_col = f'{joint}_Y'
        z_col = f'{joint}_Z'
        if x_col in frame_data and y_col in frame_data and z_col in frame_data:
            # Get joint position
            joint_pos = np.array([
                frame_data[x_col],
                frame_data[y_col],
                frame_data[z_col]
            ])
            joint_positions[joint] = joint_pos
        else:
            print(f"Warning: Missing data for joint {joint}")

    return joint_positions

# Function to apply rotations to joint positions
def apply_rotations(joint_positions, frame_data, frame_number):
    for joint in joints:
        if joint == 'Hand':
            continue  # Skip the hand itself
        x_col = f'{joint}_X.1'
        y_col = f'{joint}_Y.1'
        z_col = f'{joint}_Z.1'
        w_col = f'{joint}_W'
        if x_col in frame_data and y_col in frame_data and z_col in frame_data and w_col in frame_data:
            # Get joint rotation
            rotation = np.array([
                frame_data[x_col],
                frame_data[y_col],
                frame_data[z_col],
                frame_data[w_col]
            ])
            # Apply rotation to the joint position
            original_position = joint_positions[joint]
            rotated_position = rotate_position(original_position, rotation)
            joint_positions[joint] = rotated_position

            # Debug: Print original and rotated positions for every 10th frame
            if frame_number == 10:
                print(f"Frame: {frame_number}, Joint: {joint}")
                print(f"Original Position: {original_position}")
                print(f"Rotation: {rotation}")
                print(f"Rotated Position: {rotated_position}")
        else:
            print(f"Warning: Missing rotation data for joint {joint}")

    return joint_positions

# Function to rotate a position using a quaternion ->
#utilizzo la formula del quaternione
def rotate_position(position, quaternion):
    # Normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    # Quaternion rotation formula
    rotated_position = np.array([
        (1 - 2*y**2 - 2*z**2) * position[0] + (2*x*y - 2*z*w) * position[1] + (2*x*z + 2*y*w) * position[2],
        (2*x*y + 2*z*w) * position[0] + (1 - 2*x**2 - 2*z**2) * position[1] + (2*y*z - 2*x*w) * position[2],
        (2*x*z - 2*y*w) * position[0] + (2*y*z + 2*x*w) * position[1] + (1 - 2*x**2 - 2*y**2) * position[2]
    ])
    return rotated_position

# Define the connections between joints to form the skeleton
connections = [
    # Thumb
    ('Hand', 'Thumb_CMC'), ('Thumb_CMC', 'Thumb_MCP'),
    ('Thumb_MCP', 'Thumb_DIP'), ('Thumb_DIP', 'Thumb_TIP'),
    # Index Finger
    ('Hand', 'Index_CMC'), ('Index_CMC', 'Index_MCP'),
    ('Index_MCP', 'Index_PIP'), ('Index_PIP', 'Index_DIP'), ('Index_DIP', 'Index_TIP'),
    # Middle Finger
    ('Hand', 'Middle_CMC'), ('Middle_CMC', 'Middle_MCP'),
    ('Middle_MCP', 'Middle_PIP'), ('Middle_PIP', 'Middle_DIP'), ('Middle_DIP', 'Middle_TIP'),
    # Ring Finger
    ('Hand', 'Ring_CMC'), ('Ring_CMC', 'Ring_MCP'),
    ('Ring_MCP', 'Ring_PIP'), ('Ring_PIP', 'Ring_DIP'), ('Ring_DIP', 'Ring_TIP'),
    # Pinky Finger
    ('Hand', 'Pinky_CMC'), ('Pinky_CMC', 'Pinky_MCP'),
    ('Pinky_MCP', 'Pinky_PIP'), ('Pinky_PIP', 'Pinky_DIP'), ('Pinky_DIP', 'Pinky_TIP'),
]

# Define colors for each finger
finger_colors = {
    'Thumb': 'red',
    'Index': 'green',
    'Middle': 'blue',
    'Ring': 'orange',
    'Pinky': 'purple'
}

# Function to update the plot for each frame (used in animation)
def update(frame_number):
    ax.clear()
    frame_data = df.iloc[frame_number]
    joint_positions = get_joint_positions(frame_data)
    
    if use_rotations:
        joint_positions = apply_rotations(joint_positions, frame_data, frame_number)

    # Plot each joint
    for joint, pos in joint_positions.items():
        ax.scatter(pos[0], pos[1], pos[2], color='black', s=20)
        # Optionally label each joint
        # ax.text(pos[0], pos[1], pos[2], joint, size=8)

    # Connect the joints to form the skeleton
    for connection in connections:
        joint1, joint2 = connection
        if joint1 in joint_positions and joint2 in joint_positions:
            x_values = [joint_positions[joint1][0], joint_positions[joint2][0]]
            y_values = [joint_positions[joint1][1], joint_positions[joint2][1]]
            z_values = [joint_positions[joint1][2], joint_positions[joint2][2]]
            
            # Determine the color based on the finger
            if 'Thumb' in joint1:
                color = finger_colors['Thumb']
            elif 'Index' in joint1:
                color = finger_colors['Index']
            elif 'Middle' in joint1:
                color = finger_colors['Middle']
            elif 'Ring' in joint1:
                color = finger_colors['Ring']
            elif 'Pinky' in joint1:
                color = finger_colors['Pinky']
            else:
                color = 'black'  # Default color if not matched
            
            ax.plot(x_values, y_values, z_values, color=color)
        else:
            print(f"Warning: Missing data for connection {joint1} to {joint2}")

    # Set plot limits (adjust as needed)
    x_vals = [pos[0] for pos in joint_positions.values()]
    y_vals = [pos[1] for pos in joint_positions.values()]
    z_vals = [pos[2] for pos in joint_positions.values()]

    ax.set_xlim([min(x_vals) - 0.1, max(x_vals) + 0.1])  # Adjust margins as needed
    ax.set_ylim([min(y_vals) - 0.1, max(y_vals) + 0.1])
    ax.set_zlim([min(z_vals) - 0.1, max(z_vals) + 0.1])

    # Invert the X axis
    #ax.invert_xaxis()

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Hand Skeleton Visualization - Frame {frame_number}')
    
# Check for command-line arguments
if len(sys.argv) > 1:
    # If an argument is passed, show the specified frame
    frame_number = int(sys.argv[1])
    update(frame_number)
    plt.show()
else:
    # Create an animation of the hand movement over time
    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50)
    plt.show()