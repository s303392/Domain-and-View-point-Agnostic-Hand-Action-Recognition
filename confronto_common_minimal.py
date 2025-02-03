import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

def load_pose_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    poses = [list(map(float, line.strip().split())) for line in data]
    poses = np.array(poses)
    print(f"Loaded poses from {file_path}, shape: {poses.shape}")
    return poses

def reduce_to_common_minimal(poses):
    min_common_joints = [0, 8, 3, 7, 11, 15, 19]  # Indici dei giunti comuni minimi
    if len(poses.shape) == 2:
        poses = poses.reshape(-1, 20, 3)  # Assuming 20 joints and 3 coordinates (x, y, z)
    print(f"Reducing to common minimal, original shape: {poses.shape}")
    return poses[:, min_common_joints, :]

def scale_skel(skel):
    wrist_kp, middle_base_kp = 0, 1
    torso_dists = np.linalg.norm(skel[:, middle_base_kp] - skel[:, wrist_kp], axis=1)
    for i in range(skel.shape[0]):
        rel = 1.0 / torso_dists[i] if torso_dists[i] != 0 else 1
        skel[i] = skel[i] * rel
    return skel

def get_relative_coordinates(skel):
    wrist_kp, middle_base_kp = 0, 1
    skels_centers = (skel[:, middle_base_kp, :] + skel[:, wrist_kp, :]) / 2
    return skel - np.expand_dims(skels_centers, axis=1)

def rotate_sequence(skel, angle_noise=10):
    rotation_matrix = R.from_euler('xyz', np.random.uniform(-angle_noise, angle_noise, [3]), degrees=True).as_matrix()
    return np.dot(skel, rotation_matrix)

def plot_hand_connections_7(pose, ax, title, color='b', joint_color='darkblue'):
    joints = pose.reshape(-1, 3)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=joint_color, s=10, edgecolors='k')  
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], str(i), color=joint_color)
    connections = [
        (0, 1),  
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)  
    ]
    for connection in connections:
        joint1, joint2 = connection
        ax.plot([joints[joint1, 0], joints[joint2, 0]], 
                [joints[joint1, 1], joints[joint2, 1]], 
                [joints[joint1, 2], joints[joint2, 2]], c=color, linewidth=1)  
    ax.set_title(title)

def plot_hand_connections_20(pose, ax, title, color='b', joint_color='darkblue'):
    joints = pose.reshape(-1, 3)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=joint_color, s=10, edgecolors='k')  
    
    # Define the connections for a hand with 20 joints
    connections = [
        (0, 1), (1, 2), (2, 3),  # Thumb
        (0, 4), (4, 5), (5, 6), (6, 7),  # Index
        (0, 8), (8, 9), (9, 10), (10, 11),  # Middle
        (0, 12), (12, 13), (13, 14), (14, 15),  # Ring
        (0, 16), (16, 17), (17, 18), (18, 19)  # Pinky
    ]
    
    for connection in connections:
        joint1, joint2 = connection
        color = 'g' if (joint1, joint2) in [(0, 1), (1, 2), (2, 3)] else 'b'  # Color the thumb connections green
        ax.plot([joints[joint1, 0], joints[joint2, 0]], 
                [joints[joint1, 1], joints[joint2, 1]], 
                [joints[joint1, 2], joints[joint2, 2]], c=color, linewidth=1)  
    ax.set_title(title)

def save_poses_to_file(poses, file_path):
    with open(file_path, 'w') as file:
        for pose in poses:
            for joint in pose:
                file.write(' '.join(map(str, joint)) + '\n')
            file.write('\n')  # Separate poses by a newline

def compare_poses(poses_1, poses_2):
    min_frames = min(len(poses_1), len(poses_2))
    poses_1 = poses_1[:min_frames]
    poses_2 = poses_2[:min_frames]
    differences = np.abs(poses_1 - poses_2)
    mean_differences = np.mean(differences, axis=0)
    print("Mean differences between poses:")
    print(mean_differences)

def visualize_poses(file_path_1, file_path_2, animate=False):
    poses_1 = load_pose_data(file_path_1)
    poses_2 = load_pose_data(file_path_2)
    
    poses_1_minimal = reduce_to_common_minimal(poses_1)
    poses_2_minimal = reduce_to_common_minimal(poses_2)
    
    poses_1_scaled = scale_skel(poses_1_minimal.copy())
    poses_2_scaled = scale_skel(poses_2_minimal.copy())
    
    poses_1_relative = get_relative_coordinates(poses_1_scaled.copy())
    poses_2_relative = get_relative_coordinates(poses_2_scaled.copy())
    
    # Apply rotation
    poses_1_rotated = rotate_sequence(poses_1_relative.copy())
    poses_2_rotated = rotate_sequence(poses_2_relative.copy())
    
    # # Save the poses to files
    # save_poses_to_file(poses_1_rotated, 'poses_1_rotated.txt')
    # save_poses_to_file(poses_2_rotated, 'poses_2_rotated.txt')
    
    # Compare poses
    compare_poses(poses_1_rotated, poses_2_rotated)
    
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    
    max_frames = min(len(poses_1), len(poses_2))
    
    if animate:
        def update(frame):
            ax1.cla()
            plot_hand_connections_20(poses_1[min(frame, len(poses_1)-1)], ax1, f'Original {file_path_1.split("/")[-2]} ({frame+1}/{len(poses_1)})', color='b', joint_color='darkblue')
            
            ax2.cla()
            plot_hand_connections_7(poses_1_rotated[min(frame, len(poses_1_rotated)-1)], ax2, f'Final {file_path_1.split("/")[-2]} ({frame+1}/{len(poses_1_rotated)})', color='b', joint_color='darkblue')
            
            ax3.cla()
            plot_hand_connections_20(poses_2[min(frame, len(poses_2)-1)], ax3, f'Original {file_path_2.split("/")[-2]} ({frame+1}/{len(poses_2)})', color='r', joint_color='darkred')
            
            ax4.cla()
            plot_hand_connections_7(poses_2_rotated[min(frame, len(poses_2_rotated)-1)], ax4, f'Final {file_path_2.split("/")[-2]} ({frame+1}/{len(poses_2_rotated)})', color='r', joint_color='darkred')
        
        ani = FuncAnimation(fig, update, frames=max_frames, interval=200)
        plt.show()
    else:
        plot_hand_connections_20(poses_1[0], ax1, f'Original {file_path_1.split("/")[-2]} (1/{len(poses_1)})', color='b', joint_color='darkblue')
        plot_hand_connections_7(poses_1_rotated[0], ax2, f'Final {file_path_1.split("/")[-2]} (1/{len(poses_1_rotated)})', color='b', joint_color='darkblue')
        plot_hand_connections_20(poses_2[0], ax3, f'Original {file_path_2.split("/")[-2]} (1/{len(poses_2)})', color='r', joint_color='darkred')
        plot_hand_connections_7(poses_2_rotated[0], ax4, f'Final {file_path_2.split("/")[-2]} (1/{len(poses_2_rotated)})', color='r', joint_color='darkred')
        plt.show()

# Example usage
f_phab = './datasets/common_pose/F-PHAB/Subject3_putsalt_1_val_jn20.txt'
shrec = './datasets/common_pose/SHREC2017/gesture1_finger2_subject10_essai4_train_jn20.txt'
myShrec = './datasets/common_pose/mySHREC-17/grab04_LucaTracker_R_combined_skeleton.txt'
myDataset = './datasets/common_pose/myDataset/screw_dx_Luca030_31-01_LucaTracker_R_combined_skeleton.txt'

# Chiedi all'utente se vuole visualizzare l'animazione
animate = input("Vuoi visualizzare l'animazione? (s/n): ").strip().lower() == 's'

#visualize_poses(f_phab, shrec, animate=animate)
#visualize_poses(f_phab, myShrec, animate=animate)
#visualize_poses(myShrec, shrec, animate=animate)
visualize_poses(myDataset, shrec, animate=animate)