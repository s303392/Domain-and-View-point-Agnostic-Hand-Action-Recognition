# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import argparse

# Script progettato per convertire i dati CSV delle mani (sinistra e destra) in un formato comune utilizzabile dal modello TCN

joints_num = 20

# Definizione dei giunti per le due mani
joint_names = 'Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP'
fp_joints = {s: joint_num for joint_num, s in enumerate(joint_names.split(', '))}

# Definizione dei giunti comuni da mantenere
if joints_num == 20:
    common_pose_joints = ['Wrist'] + ['TPIP', 'TDIP', 'TTIP'] + [f'{finger}{part}' for finger in ['I', 'M', 'R', 'P'] for part in ['MCP', 'PIP', 'DIP', 'TIP']]

common_pose_joint_inds = [fp_joints[s] for s in common_pose_joints]

# Parser degli argomenti
parser = argparse.ArgumentParser(description='Process CSV data for hand skeletons.')
parser.add_argument('--reduce_joints', action='store_true', help='Flag per ridurre i giunti al formato comune')
parser.add_argument('--annotation_file', type=str, required=True, help='Path al file delle annotazioni contenente i percorsi dei CSV e le etichette delle azioni')
args = parser.parse_args()

custom_name = 'single_0_action'

# Percorsi dei dataset
output_dataset_path = './datasets/common_pose/myDataset/'
os.makedirs(output_dataset_path, exist_ok=True)

# Leggi il file delle annotazioni
annotation_base_path = os.path.dirname(args.annotation_file)
with open(args.annotation_file, 'r') as annotation_file:
    lines = annotation_file.readlines()

# Elaborazione di ogni riga del file di annotazioni
for line in lines:
    left_csv_rel_path, right_csv_rel_path, label = line.strip().split()

    # Costruzione dei percorsi assoluti per i file CSV
    left_csv_path = os.path.join(annotation_base_path, left_csv_rel_path)
    right_csv_path = os.path.join(annotation_base_path, right_csv_rel_path)

    # Caricamento dei dati CSV delle mani sinistra e destra
    left_data = pd.read_csv(left_csv_path)
    right_data = pd.read_csv(right_csv_path)

    # Filtraggio delle colonne per ottenere solo le posizioni dei giunti (X, Y, Z)
    joint_columns = [col for col in left_data.columns if any(axis in col for axis in ['_X', '_Y', '_Z'])]
    left_data_filtered = left_data[joint_columns]
    right_data_filtered = right_data[joint_columns]

    # Verifica che il numero di frame sia lo stesso per entrambe le mani
    assert len(left_data_filtered) == len(right_data_filtered), "Il numero di frame nelle due mani non corrisponde."

    # Combina i dati delle due mani lungo l'asse delle feature
    combined_data = pd.concat([left_data_filtered, right_data_filtered], axis=1)

    # Ristrutturazione dei dati
    num_frames = len(combined_data)
    num_columns = combined_data.shape[1]

    # Verifica che il numero di colonne sia divisibile per 3
    if num_columns % 3 != 0:
        raise ValueError("Il numero di colonne non è divisibile per 3, i dati potrebbero non essere formattati correttamente.")

    # Calcola il numero di giunti in base al numero di colonne
    num_joints = num_columns // 3

    # Ristrutturazione dei dati in (num_frames, num_joints, 3)
    combined_data_np = combined_data.values.reshape((num_frames, num_joints, 3))  # num_joints giunti con 3 dimensioni (x, y, z)

    # Selezione dei giunti comuni se richiesto
    dati_output = combined_data_np
    if args.reduce_joints:
        # Assicurati che gli indici dei giunti comuni siano corretti rispetto al numero di giunti effettivi
        if max(common_pose_joint_inds) >= num_joints:
            raise ValueError("Gli indici dei giunti comuni non corrispondono al numero di giunti nei dati.")
        dati_output = combined_data_np[:, common_pose_joint_inds, :]

    # Salva i dati rielaborati in un nuovo file di output
    output_filename = f'{custom_name}_combined_skeleton_common_pose.txt' if args.reduce_joints else f'{custom_name}_combined_skeleton_full_pose.txt'
    new_skel_path = os.path.join(output_dataset_path, output_filename)
    os.makedirs(os.path.dirname(new_skel_path), exist_ok=True)
    with open(new_skel_path, 'w') as f:
        for frame_joints in dati_output:
            frame_joints_flat = ' '.join(map(str, frame_joints.flatten()))
            f.write(frame_joints_flat + '\n')

    # Creazione di un file di annotazioni
    annotations_file = os.path.join('dataset_scripts/myDataset/', 'total_annotations.txt')
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
    with open(annotations_file, 'a') as store:
        store.write(f'{new_skel_path} {label}\n')
