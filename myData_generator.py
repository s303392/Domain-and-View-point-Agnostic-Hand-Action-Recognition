#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Data Generator to Handle Manus Dataset Variations

This version is modified to work with different versions of the Manus dataset. It can handle variations with optional features like velocity, acceleration, and additional features based on flags.
"""
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import scipy.ndimage.interpolation as inter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.special import comb
import numpy as np
import pandas as pd

class MyDataGenerator():

    def __init__(self, 
                 max_seq_len,
                 scale_by_torso, temporal_scale, 
                 use_rotations,
                 use_relative_coordinates, 
                 use_jcd_features, use_coord_diff, 
                 use_bone_angles, use_bone_angles_diff,
                 skip_frames = [],
                 noise = None,
                 dataset = '',
                 joints_format='common_minimal',
                 rotation_noise = None,
                 use_velocity_acceleration=False,
                 use_additional_features=False,
                 **kargs):

        self.use_relative_coordinates = use_relative_coordinates
        self.use_jcd_features = use_jcd_features
        self.use_coord_diff = use_coord_diff
        self.use_bone_angles = use_bone_angles
        self.use_bone_angles_diff = use_bone_angles_diff
        self.use_velocity_acceleration = use_velocity_acceleration
        self.use_additional_features = use_additional_features
        
        assert use_rotations in ['by_positive', 'by_batch', 'by_sample', None], 'Rotation mode [{}] not handled'.format(use_rotations)
        self.use_rotations = use_rotations
        self.rotation_noise = rotation_noise

        if joints_format == 'common_minimal':
            self.min_common_joints = [0, 9, 4, 8, 12, 16, 20]  # Wrist, MMCP, TTIP, ITIP, MTIP, RTIP, PTIP
            self.joints_num = 7
        elif joints_format == 'common':
            self.joints_num = 25
        else:
            raise ValueError('joints_format {} not handled'.format(joints_format))

        self.joints_format = joints_format
        self.max_seq_len = max_seq_len
        self.scale_by_torso = scale_by_torso
        self.temporal_scale = temporal_scale
        self.noise = noise
        self.rotation_noise = rotation_noise
        self.skip_frames = skip_frames
        self.dataset = dataset

        self.base_joints_dim = 3  # Position (X, Y, Z)
        self.velocity_acceleration_dim = 3 if use_velocity_acceleration else 0  # Velocity and Acceleration (X, Y, Z each)
        self.additional_features_dim = 20 if use_additional_features else 0  # Additional features
        self.total_features_per_joint = self.base_joints_dim + self.velocity_acceleration_dim

    def load_skel_coords(self, filename):
        if filename.endswith('.csv'):
            skel = pd.read_csv(filename, sep=';', decimal=',', encoding='utf-8', error_bad_lines=False).values
        else:
            with open(filename, 'r') as f:
                skel = f.read().splitlines()
            skel = np.array([list(map(float, line.split())) for line in skel])

        num_frames = skel.shape[0]
        num_joint_features = self.joints_num * self.total_features_per_joint
        skel_joints = skel[:, :num_joint_features].reshape(num_frames, self.joints_num, self.total_features_per_joint)

        print(f"Loaded skeleton joints shape: {skel_joints.shape}")

        if self.joints_format == 'common_minimal':
            skel_joints = skel_joints[:, self.min_common_joints, :]

        skel_additional = skel[:, num_joint_features:] if self.use_additional_features else None

        return skel_joints, skel_additional

    def scale_skel(self, skel):
        torso_dists = np.linalg.norm(skel[:, 9] - skel[:, 0], axis=1)  # MMCP - Wrist
        for i in range(skel.shape[0]):
            rel = 1.0 / torso_dists[i] if torso_dists[i] != 0 else 1
            skel[i] = skel[i] * rel
        return skel

    def get_pose_data_v2(self, body, validation, rotation_matrix=None):
        # 1. Remove frames without predictions
        body = body[np.all(~np.all(body == 0, axis=2), axis=1)]

        # 2. Select skipping frames
        if len(self.skip_frames) > 0:
            sk = np.random.choice(self.skip_frames)
        else:
            sk = 1

        # 3. Reduce or extend the movement by interpolation
        if not validation and self.temporal_scale is not False:
            orig_new_frames = len(body)
            temporal_scale = list(self.temporal_scale)
            temporal_scale[0] = int(temporal_scale[0] * orig_new_frames)
            temporal_scale[1] = int(temporal_scale[1] * orig_new_frames)
            new_num_frames = np.random.randint(*temporal_scale)
            new_num_frames = max(new_num_frames, 2 * sk)
            zoom_factor = new_num_frames / orig_new_frames
            body = inter.zoom(body, (zoom_factor, 1, 1), mode='nearest')

        # 4. Reduce frame rate
        if len(self.skip_frames) > 0:
            if validation:
                sk_init = 0
            else:
                sk_init = np.random.randint(sk)
            if len(body[sk_init::sk]) >= 2:
                body = body[sk_init::sk]

        # 5. Modify movement speed
        if self.max_seq_len > 0:
            body = self.zoom_to_max_len(body)
        elif self.max_seq_len < 0:
            if not validation:
                start = np.random.randint(max(len(body) - abs(self.max_seq_len) + 1, 1))
                end = start + abs(self.max_seq_len)
                body = body[start:end]
            else:
                start = max(0, (len(body) - abs(self.max_seq_len)) // 2)
                end = start + abs(self.max_seq_len)
                body = body[start:end]

        # 6. Scale by torso
        if self.scale_by_torso:
            body = self.scale_skel(body)

        # 7. Add random noise and scales again
        if not validation and self.noise is not None:
            if self.noise[0] == 'uniform':
                noise = np.random.uniform(low=-self.noise[1], high=self.noise[1], size=body.shape)
            elif self.noise[0] == 'normal':
                noise = np.random.normal(loc=0, scale=self.noise[1], size=body.shape)
            else:
                raise ValueError('noise type [{}] not handled'.format(self.noise[0]))
            body = body + noise
            if self.scale_by_torso:
                body = self.scale_skel(body)

        # 8. Rotate sequence
        if not validation and self.use_rotations is not None:
            if rotation_matrix is None:
                rotation_matrix = self.get_random_rotation_matrix()
            body = self.rotate_sequence(body, rotation_matrix)

        # 9. Rotation noise
        if not validation and self.rotation_noise is not None and self.rotation_noise > 0:
            rotation_matrix = self.get_constrained_rotation_matrix(self.rotation_noise)
            body = self.rotate_sequence(body, rotation_matrix)

        # 10. Get movement features
        num_frames = len(body)
        pose_features = []

        if self.use_relative_coordinates:
            rel_coordinates = self.get_relative_coordinates(body)
            pose_features.append(np.reshape(rel_coordinates, (num_frames, self.joints_num * self.joints_dim)))

        if self.use_jcd_features:
            jcd_features = self.get_jcd_features(body, num_frames)
            pose_features.append(jcd_features)

        if self.use_coord_diff:
            speed_features = body[1:] - body[:-1]
            speed_features = np.reshape(speed_features, (num_frames - 1, self.joints_num * self.joints_dim))
            speed_features = np.concatenate([np.expand_dims(speed_features[0], axis=0), speed_features], axis=0)
            pose_features.append(speed_features)

        if self.use_bone_angles or self.use_bone_angles_diff:
            bone_angles = self.get_body_spherical_angles(body)
            if self.use_bone_angles_diff:
                bone_angles_diff = bone_angles[1:] - bone_angles[:-1]
                bone_angles_diff = np.concatenate([np.expand_dims(bone_angles_diff[0], axis=0), bone_angles_diff], axis=0)
                pose_features.append(bone_angles_diff)
            if self.use_bone_angles:
                pose_features.append(bone_angles)

        pose_features = np.concatenate(pose_features, axis=1).astype('float32')

        return pose_features

    def get_relative_coordinates(self, skels):
        skels_centers = (skels[:, 9, :] + skels[:, 0, :]) / 2  # MMCP - Wrist
        return skels - np.expand_dims(skels_centers, axis=1)

    def get_jcd_features(self, p, num_frames):
        jcd = []
        iu = np.triu_indices(self.joints_num, 1, self.joints_num)
        for f in range(num_frames):
            d_m = cdist(p[f], p[f], 'euclidean')
            d_m = d_m[iu]
            jcd.append(d_m)
        jcd = np.stack(jcd)
        return jcd

    def get_bone_spherical_angles(self, v):
        elevation = np.arctan2(v[:, 2], np.sqrt(v[:, 0]**2 + v[:, 1]**2))
        azimuth = np.arctan2(v[:, 1], v[:, 0])
        return np.column_stack([elevation, azimuth])

    def get_body_spherical_angles(self, body):
        angles = np.column_stack([self.get_bone_spherical_angles(body[:, i+1] - body[:, i]) for i in range(self.joints_num - 1)])
        return angles

    def triplet_data_generator(self, pose_annotations_file, batch_size, validation, classification, num_classes, **kwargs):
        def read_annotations():
            pose_files = {}
            with open(pose_annotations_file, 'r') as f:
                for line in f:
                    filename, label = line.split()
                    label = int(label)
                    if label in pose_files:
                        pose_files[label].append(filename)
                    else:
                        pose_files[label] = [filename]
            for k in pose_files.keys():
                np.random.shuffle(pose_files[k])
            return pose_files

        pose_files = read_annotations()

        while True:
            batch_samples = []
            batch_labels = []
            batch_additional_features = []

            for _ in range(batch_size):
                label = np.random.choice(list(pose_files.keys()))
                filename = pose_files[label].pop()

                p_joints, p_additional = self.load_skel_coords(filename)
                sample_joints, sample_additional = self.get_pose_data_v2(p_joints, validation)

                batch_samples.append(sample_joints)
                batch_labels.append(label)
                if self.use_additional_features:
                    batch_additional_features.append(sample_additional)

            X_joints = pad_sequences(batch_samples, maxlen=self.max_seq_len, padding='post', dtype='float32')
            y_clf = to_categorical(batch_labels, num_classes=num_classes) if classification else batch_labels

            if self.use_additional_features:
                X_additional = pad_sequences(batch_additional_features, maxlen=self.max_seq_len, padding='post', dtype='float32')
                yield [X_joints, X_additional], y_clf
            else:
                yield X_joints, y_clf

if __name__ == '__main__':
    gen_params = {
        'max_seq_len': 32,
        'scale_by_torso': True,
        'use_velocity_acceleration': True,
        'use_additional_features': True,
        'joints_format': 'common_minimal',
        'temporal_scale': (0.8, 1.2),
        'noise': ('normal', 0.03),
        'rotation_noise': 20,
        'use_rotations': 'by_positive',
        'use_relative_coordinates': True,
        'use_jcd_features': True, 
        'use_coord_diff': True,
        'use_bone_angles': True,
        'use_bone_angles_diff': True,
        'skip_frames': [2,3],
        'dataset': 'CP_',
    }

    data_gen = MyDataGenerator(**gen_params)
    gen_params = {
        'pose_annotations_file': './datasets/common_pose/myDataset/total_annotations.txt',
        'batch_size': 6,
        'validation': False,
        'classification': True,
        'num_classes': 10,
    }

    triplet_gen = data_gen.triplet_data_generator(**gen_params)
    for i in range(3):
        batch_X, batch_Y = next(triplet_gen)
        if data_gen.use_additional_features:
            print(f'Batch {i} - X joints shape: {batch_X[0].shape}, X additional shape: {batch_X[1].shape}, Y shape: {len(batch_Y)}')
        else:
            print(f'Batch {i} - X joints shape: {batch_X.shape}, Y shape: {len(batch_Y)}')
