import os
import random
from collections import defaultdict

def split_dataset(annotation_file, train_percentage, output_dir):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    # Organize lines by action label
    action_dict = defaultdict(list)
    for line in lines:
        action = line.strip().split()[-1]
        action_dict[action].append(line)

    train_lines = []
    test_lines = []

    # Split each action's data
    for action, action_lines in action_dict.items():
        random.shuffle(action_lines)
        split_index = int(len(action_lines) * train_percentage)
        train_lines.extend(action_lines[:split_index])
        test_lines.extend(action_lines[split_index:])

    # Write to output files
    train_file = os.path.join(output_dir, 'myDataset80_train577.txt')
    test_file = os.path.join(output_dir, 'myDataset80_test577.txt')

    os.makedirs(output_dir, exist_ok=True)

    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    with open(test_file, 'w') as f:
        f.writelines(test_lines)

    print(f"Train file created: {train_file}")
    print(f"Test file created: {test_file}")

if __name__ == "__main__":
    ANNOTATIONS_FILE = 'c:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/dataset_scripts/myDataset/ref_ComPose_annotations.txt'
    TRAIN_PERCENTAGE = 0.8
    OUTPUT_DIR = 'c:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/dataset_scripts/common_pose/annotations/myDataset'

    split_dataset(ANNOTATIONS_FILE, TRAIN_PERCENTAGE, OUTPUT_DIR)