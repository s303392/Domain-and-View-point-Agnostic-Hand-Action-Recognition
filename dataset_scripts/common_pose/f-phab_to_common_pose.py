# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle

# Questo file è progettato per convertire le annotazioni del dataset F-PHAB in un formato comune

joints_num = 20

fp_joints = { s:joint_num for joint_num,s in enumerate('Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP'.split(', ')) }

if joints_num == 20:
    common_pose_joints = [ 'Wrist' ] +\
                        ['TPIP', 'TDIP', 'TTIP'] +\
                        [ '{}{}'.format(finger, part) for finger in  ['I', 'M', 'R', 'P' ] \
                         for part in ['MCP', 'PIP', 'DIP', 'TIP'] ]

common_pose_joint_inds = [ fp_joints[s] for s in common_pose_joints ]

# Percorsi dei dataset
common_pose_dataset_path = 'C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/datasets/common_pose/'
base_dataset = 'C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/datasets/F-PHAB/'
output_dataset_path = 'C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/dataset_scripts/F_PHAB/paper_tables_annotations'

annotations_filename = os.path.join(base_dataset, 'data_split_action_recognition.txt')
with open(annotations_filename, 'r') as f:
    anns = f.read().splitlines()

anns_train = anns[1:601]
anns_val = anns[602::]

annotations_dir = os.path.abspath('C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/dataset_scripts/common_pose/annotations/F_PHAB')
os.makedirs(annotations_dir, exist_ok=True)

for split_mode, anns in [('train', anns_train), ('val', anns_val)]:
    # Crea il file di annotazioni
    annotations_file = annotations_dir + '/annotations_' + split_mode + '_jn' + str(joints_num) + '.txt'
    with open(annotations_file, 'w') as store:
        for ann in anns:
            # Leggi e analizza lo scheletro
            filename, label = ann.split()
            skeleton_file = base_dataset + 'Hand_pose_annotation_v1/' + filename + '/skeleton.txt'
            with open(skeleton_file, 'r') as f:
                joints = f.read().splitlines()
            joints = np.array([l.split() for l in joints])[:, 1:].reshape((len(joints), 21, 3))

            # Percorso per il nuovo file di scheletro
            new_skel_path = common_pose_dataset_path + 'F-PHAB/' + filename.replace('_', '').replace('/', '_') + '_' + split_mode + '_jn' + str(joints_num) + '.txt'
            joints_new = joints[:, common_pose_joint_inds]

            os.makedirs(os.path.dirname(new_skel_path), exist_ok=True)
            with open(new_skel_path, 'w') as f:
                for frame_joints in joints_new:
                    frame_joints = ' '.join(frame_joints.flatten().tolist())
                    f.write(frame_joints + '\n')

            store.write(new_skel_path + ' ' + str(label) + '\n')

# Combina i file di annotazioni train e val in un unico file total
total_annotations_file = output_dataset_path + '/total_annotations_jn' + str(joints_num) + '.txt'
with open(total_annotations_file, 'w') as outfile:
    for split_mode in ['train', 'val']:
        annotations_file = annotations_dir + '/annotations_' + split_mode + '_jn' + str(joints_num) + '.txt'
        with open(annotations_file, 'r') as infile:
            for line in infile:
                outfile.write(line)

# Funzione per creare i fold
def create_folds(annotations, labels, num_folds):
    indices = list(range(len(annotations)))
    np.random.shuffle(indices)
    fold_size = len(annotations) // num_folds
    remainder = len(annotations) % num_folds
    folds = []
    start_index = 0
    for i in range(num_folds):
        end_index = start_index + fold_size + (1 if i < remainder else 0)
        fold_indices = indices[start_index:end_index]
        fold_annotations = [annotations[i] for i in fold_indices]
        fold_labels = [labels[i] for i in fold_indices]
        folds.append({'indexes': fold_indices, 'annotations': fold_annotations, 'labels': fold_labels})
        start_index = end_index
    return folds

# Leggi le annotazioni combinate
with open(total_annotations_file, 'r') as f:
    total_annotations = f.read().splitlines()
total_labels = [ann.split()[-1] for ann in total_annotations]
total_annotations = [ann.split()[0] for ann in total_annotations]

# Crea i fold
folds_1_1 = create_folds(total_annotations, total_labels, 2)
folds_base = create_folds(total_annotations, total_labels, 3)
folds_subject = create_folds(total_annotations, total_labels, 6)

# Stampa la struttura dei fold
print("Folds 1:1:", len(folds_1_1[0]['indexes']))#588 = 1175 (sequenze totali) // 2
print("Folds Base:", len(folds_base[0]['indexes']))#392 1175 // 3
print("Folds Subject:", len(folds_subject[0]['indexes']))#196 1175 // 6

# Salva i fold in file .pckl
with open(os.path.join(output_dataset_path, 'annotations_table_2_cross-action_folds_1_1_jn20.pckl'), 'wb') as f:
    pickle.dump(folds_1_1, f)

with open(os.path.join(output_dataset_path, 'annotations_table_2_cross-action_folds_jn20.pckl'), 'wb') as f:
    pickle.dump(folds_base, f)

with open(os.path.join(output_dataset_path, 'annotations_table_2_cross-person_folds_jn20.pckl'), 'wb') as f:
    pickle.dump(folds_subject, f)
