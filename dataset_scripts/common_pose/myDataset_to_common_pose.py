# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import argparse
import pickle
import json
from sklearn.model_selection import KFold

# Numero di giunti nel dataset originale (25)
original_joints_num = 25

# Numero di giunti nel formato common_pose (20)
common_pose_joints_num = 20

# Definizione dei giunti nel formato common_pose
if common_pose_joints_num == 20:
    common_pose_joints = [ 'Wrist' ] +\
                        ['TMCP', 'TDIP', 'TTIP'] +\
                        [ '{}{}'.format(finger, part) for finger in  ['I', 'M', 'R', 'P' ] \
                         for part in ['MCP', 'PIP', 'DIP', 'TIP'] ]

# Mappa i giunti del tuo dataset ai giunti common_pose
my_joints = { s:joint_num for joint_num,s in enumerate('Wrist, TCMC, TMCP, TDIP, TTIP, ICMC, IMCP, IPIP, IDIP, ITIP, MCMC, MMCP, MPIP, MDIP, MTIP, RCMC, RMCP, RPIP, RDIP, RTIP, PCMC, PMCP, PPIP, PDIP, PTIP'.split(', ')) }
common_pose_joint_inds = [ my_joints[s] for s in common_pose_joints ]

# Parser degli argomenti
parser = argparse.ArgumentParser()
parser.add_argument('--annotation_file', type=str, required=True, help='Path al file delle annotazioni contenente i percorsi dei CSV e le etichette delle azioni')
parser.add_argument('--include_velocity_acceleration', action='store_true', help='Includi velocità e accelerazione nei dati di output')
parser.add_argument('--include_additional_features', action='store_true', help='Includi dati addizionali (angoli articolari, pinch distance) nei dati di output')
parser.add_argument('--include_rotations', action='store_true', help='Includi le rotazioni dei giunti nei dati di output')
parser.add_argument('--single_hand', action='store_true', help='Utilizza i dati di una sola mano')
parser.add_argument('--create_folds', action='store_true', help='Crea i fold per la valutazione del modello')
parser.add_argument('--cut_config', type=str, required=True, help='Path al file di configurazione dei tagli delle sequenze')
args = parser.parse_args()

# Carica il file di configurazione dei tagli
with open(args.cut_config, 'r') as f:
    cut_config = json.load(f)

# Percorsi dei dataset
output_dataset_path = './datasets/common_pose/myDataset/'
os.makedirs(output_dataset_path, exist_ok=True)

# File di annotazioni di input
annotation_base_path = os.path.dirname(args.annotation_file)
with open(args.annotation_file, 'r') as annotation_file:
    lines = annotation_file.readlines()

# Estrai e processa i dati da ogni file CSV di annotazioni
annotations = []
for line in lines:
    if args.single_hand:
        csv_rel_path, label = line.strip().split()
        csv_path = os.path.join(annotation_base_path, csv_rel_path)
        data = pd.read_csv(csv_path,
            sep=';',
            decimal=',', 
            encoding='utf-8',
            error_bad_lines=False  
            )
        print(f"Caricati dati CSV: {csv_path}")
        print(f"Dimensioni data: {data.shape}")
        annotations.append((csv_path, label))
    else:
        left_csv_rel_path, right_csv_rel_path, label = line.strip().split()
        left_csv_path = os.path.join(annotation_base_path, left_csv_rel_path)
        right_csv_path = os.path.join(annotation_base_path, right_csv_rel_path)
        left_data = pd.read_csv(left_csv_path)
        right_data = pd.read_csv(right_csv_path)
        print(f"Caricati dati CSV: {left_csv_path} e {right_csv_path}")
        print(f"Dimensioni left_data: {left_data.shape}, Dimensioni right_data: {right_data.shape}")
        annotations.append((left_csv_path, right_csv_path, label))

    # Filtra solo le colonne di interesse
    base_features = ['_X', '_Y', '_Z']
    rotation_features = ['rotation_X', 'rotation_Y', 'rotation_Z']
    velocity_acceleration_features = ['velocity_X', 'velocity_Y', 'velocity_Z', 'acceleration_X', 'acceleration_Y', 'acceleration_Z']
    additional_features = ['angle', 'angular_velocity', 'angular_acceleration', 'pinch_distance']

    feature_columns_per_joint = base_features
    if args.include_rotations:
        feature_columns_per_joint += rotation_features
    if args.include_velocity_acceleration:
        feature_columns_per_joint += velocity_acceleration_features

    if args.single_hand:
        feature_columns = [col for col in data.columns if any(feature in col for feature in feature_columns_per_joint)]
        additional_feature_columns = [col for col in data.columns if any(feature in col for feature in additional_features)] if args.include_additional_features else []
        data_filtered = data[feature_columns]
        additional_features_data = data[additional_feature_columns]
        print(f"Dimensioni data_filtered: {data_filtered.shape}")
        if args.include_additional_features:
            print(f"Dimensioni additional_features_data: {additional_features_data.shape}")
    else:
        feature_columns = [col for col in left_data.columns if any(feature in col for feature in feature_columns_per_joint)]
        additional_feature_columns = [col for col in left_data.columns if any(feature in col for feature in additional_features)] if args.include_additional_features else []
        left_data_filtered = left_data[feature_columns]
        right_data_filtered = right_data[feature_columns]
        left_additional_features = left_data[additional_feature_columns]
        right_additional_features = right_data[additional_feature_columns]
        print(f"Dimensioni left_data_filtered: {left_data_filtered.shape}, Dimensioni right_data_filtered: {right_data_filtered.shape}")
        if args.include_additional_features:
            print(f"Dimensioni left_additional_features: {left_additional_features.shape}, Dimensioni right_additional_features: {right_additional_features.shape}")

        if len(left_data_filtered) != len(right_data_filtered):
            raise ValueError("Il numero di frame nelle due mani non corrisponde. Controllare i file: {} e {}".format(left_csv_path, right_csv_path))

        left_data_filtered = left_data_filtered.reindex(sorted(left_data_filtered.columns), axis=1)
        right_data_filtered = right_data_filtered.reindex(sorted(right_data_filtered.columns), axis=1)
        combined_joints_data = pd.concat([left_data_filtered, right_data_filtered], axis=1)
        combined_additional_features = pd.concat([left_additional_features, right_additional_features], axis=1) if args.include_additional_features else None
        print(f"Dimensioni combined_joints_data: {combined_joints_data.shape}")
        if args.include_additional_features:
            print(f"Dimensioni combined_additional_features: {combined_additional_features.shape}")

    # Taglia la sequenza in base ai valori di inizio e fine specificati nel file di configurazione
    #TODO: deve prendere i nomi dal file e non direttamente un singolo nome
    csv_filename = "MANUS_data/grab-dx-sx_2024-12-18_12-23-48_federico_R.csv" 
    if csv_filename in cut_config:
        start_frame = cut_config[csv_filename]["start_frame"]
        end_frame = cut_config[csv_filename]["end_frame"]
    else:
        print(f"Errore: {csv_path} non trovato nel file di configurazione dei tagli.")
        continue

    if args.single_hand:
        data_filtered = data_filtered.iloc[start_frame:end_frame]
        if args.include_additional_features:
            additional_features_data = additional_features_data.iloc[start_frame:end_frame]
    else:
        combined_joints_data = combined_joints_data.iloc[start_frame:end_frame]
        if combined_additional_features is not None:
            combined_additional_features = combined_additional_features.iloc[start_frame:end_frame]

    if args.single_hand:
        num_frames = len(data_filtered)
        num_columns = data_filtered.shape[1]
        num_features_per_joint = num_columns // original_joints_num
        print(f"num_frames: {num_frames}, num_columns: {num_columns}, num_features_per_joint: {num_features_per_joint}")
        print(f"Expected size: {num_frames * original_joints_num * num_features_per_joint}, Actual size: {data_filtered.values.size}")
        data_np = data_filtered.values.reshape((num_frames, original_joints_num, num_features_per_joint))
        print(f"Dimensioni data_np: {data_np.shape}")
    else:
        num_frames = len(combined_joints_data)
        num_columns = combined_joints_data.shape[1]
        num_features_per_joint = num_columns // (original_joints_num * 2)
        print(f"num_frames: {num_frames}, num_columns: {num_columns}, num_features_per_joint: {num_features_per_joint}")
        print(f"Expected size: {num_frames * original_joints_num * 2 * num_features_per_joint}, Actual size: {combined_joints_data.values.size}")
        combined_joints_data_np = combined_joints_data.values.reshape((num_frames, original_joints_num * 2, num_features_per_joint))
        print(f"Dimensioni combined_joints_data_np: {combined_joints_data_np.shape}")

    custom_name = 'single_0_action'
    output_filename = f'{custom_name}_combined_skeleton.txt'
    new_skel_path = os.path.join(output_dataset_path, output_filename)
    os.makedirs(os.path.dirname(new_skel_path), exist_ok=True)
    with open(new_skel_path, 'w') as f:
        for i in range(num_frames):
            if args.single_hand:
                frame_joints_flat = ' '.join(map(str, data_np[i, common_pose_joint_inds].flatten()))
                if args.include_additional_features:
                    additional_features_flat = ' '.join(map(str, additional_features_data.iloc[i]))
                    f.write(frame_joints_flat + ' ' + additional_features_flat + '\n')
                else:
                    f.write(frame_joints_flat + '\n')
            else:
                frame_joints_flat = ' '.join(map(str, combined_joints_data_np[i, common_pose_joint_inds].flatten()))
                if combined_additional_features is not None:
                    additional_features_flat = ' '.join(map(str, combined_additional_features.iloc[i]))
                    f.write(frame_joints_flat + ' ' + additional_features_flat + '\n')
                else:
                    f.write(frame_joints_flat + '\n')

    print(f"Dati combinati salvati in: {new_skel_path}")

    total_annotations_file = './dataset_scripts/myDataset/total_annotations.txt'
    os.makedirs(os.path.dirname(total_annotations_file), exist_ok=True)
    with open(total_annotations_file, 'a') as store:
        store.write(f'{new_skel_path} {label}\n')

print("Conversione completata e file di annotazioni generati con successo.")

# Funzione per creare i fold
def create_folds(annotations, labels, num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    folds = []
    for train_index, test_index in kf.split(annotations):
        fold = {
            'train': [annotations[i] for i in train_index],
            'test': [annotations[i] for i in test_index],
            'train_labels': [labels[i] for i in train_index],
            'test_labels': [labels[i] for i in test_index]
        }
        folds.append(fold)
    return folds

# Leggi le annotazioni combinate
with open(total_annotations_file, 'r') as f:
    total_annotations = f.readlines()

total_labels = [ann.split()[-1] for ann in total_annotations]
total_annotations = [ann.split()[0] for ann in total_annotations]

# Verifica del numero di azioni
unique_labels = np.unique(total_labels)
if len(unique_labels) == 1:
    print("Una sola azione trovata. Saltando la creazione dei fold.")
else:
    if args.create_folds:
        # Crea i fold
        folds_1_1 = create_folds(total_annotations, total_labels, 2)
        folds_base = create_folds(total_annotations, total_labels, 3)
        folds_subject = create_folds(total_annotations, total_labels, 6)

        # Stampa la struttura dei fold
        print("Folds 1:1:", len(folds_1_1[0]['train']))  # 588 = 1175 (sequenze totali) // 2
        print("Folds Base:", len(folds_base[0]['train']))  # 392 1175 // 3
        print("Folds Subject:", len(folds_subject[0]['train']))  # 196 1175 // 6

        # Salva i fold in file .pckl
        with open(os.path.join(output_dataset_path, 'annotations_table_2_cross-action_folds_1_1_jn25.pckl'), 'wb') as f:
            pickle.dump(folds_1_1, f)

        with open(os.path.join(output_dataset_path, 'annotations_table_2_cross-action_folds_jn25.pckl'), 'wb') as f:
            pickle.dump(folds_base, f)

        with open(os.path.join(output_dataset_path, 'annotations_table_2_cross-action_folds_subject_jn25.pckl'), 'wb') as f:
            pickle.dump(folds_subject, f)
    else:
        print("Creazione dei fold disabilitata.")
