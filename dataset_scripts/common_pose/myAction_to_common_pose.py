import os
import sys
import numpy as np
import pandas as pd
import argparse

def main():
    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Converti un CSV nel formato common_pose")
    parser.add_argument('--single_csv', type=str, required=True, help='Path al file CSV della singola azione')
    args = parser.parse_args()

    # Percorso file e output
    csv_path = args.single_csv
    output_dir = './datasets/common_pose/testSingleAction/'
    os.makedirs(output_dir, exist_ok=True)

    # Numero di giunti
    original_joints_num = 25
    common_pose_joints = ['Wrist'] + ['TMCP', 'TDIP', 'TTIP'] + [
        '{}{}'.format(finger, part) for finger in ['I', 'M', 'R', 'P']
        for part in ['MCP', 'PIP', 'DIP', 'TIP']
    ]
    
    my_joints = {s: i for i, s in enumerate(
        'Wrist, TCMC, TMCP, TDIP, TTIP, ICMC, IMCP, IPIP, IDIP, ITIP, MCMC, MMCP, MPIP, MDIP, MTIP, RCMC, RMCP, RPIP, RDIP, RTIP, PCMC, PMCP, PPIP, PDIP, PTIP'.split(', ')
    )}
    common_pose_joint_inds = [my_joints[s] for s in common_pose_joints]

    try:
        # Caricare il CSV
        data = pd.read_csv(csv_path, sep=';', decimal=',', encoding='utf-8')

        # Filtrare solo le colonne di interesse
        base_features = ['_X', '_Y', '_Z']
        feature_columns = [col for col in data.columns if any(feature in col for feature in base_features)]
        data_filtered = data[feature_columns]

        # Convertire in formato numpy
        num_frames = len(data_filtered)
        num_columns = data_filtered.shape[1]
        num_features_per_joint = num_columns // original_joints_num
        data_np = data_filtered.values.reshape((num_frames, original_joints_num, num_features_per_joint))

        # Creare il file di output
        input_filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f'{input_filename}_combined_skeleton.txt'
        new_skel_path = os.path.join(output_dir, output_filename)

        with open(new_skel_path, 'w') as f:
            for i in range(num_frames):
                frame_joints_flat = ' '.join(map(str, data_np[i, common_pose_joint_inds].flatten()))
                f.write(frame_joints_flat + '\n')

        print(new_skel_path)  # Output del percorso finale per subprocess
        sys.exit(0)

    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
