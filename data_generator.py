#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:49:55 2020

@author: asabater
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.ndimage.interpolation as inter
from scipy.special import comb
from scipy.spatial.distance import cdist
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.sequence import pad_sequences


# from __OLD_data_generator_obj import DataGenerator_Hand

"""

Triplet generator
Aplicar random rotations. Igual para todas clases del batch o independiente para cada tipo de clase
Location invariance. Cannot use world/camera coordinates. Usar coordenadas relativas a la mano o no usar

Visualizar rotaciones

"""

#index, middle, ring, pinky
class DataGenerator():

    def __init__(self, 
                    max_seq_len,
                    scale_by_torso, temporal_scale, 
                    use_rotations,
                    
                    use_relative_coordinates, 
                    use_jcd_features, use_coord_diff, 
                    # use_coords_raw, use_coords, use_jcd_diff, 
                    use_bone_angles,                # use_bone_angles_cent,
                    use_bone_angles_diff,
                    
                    skip_frames = [],
                    noise = None,
                    dataset = '',
                    joints_format='common_minimal',
                    rotation_noise = None,
                    **kargs):

        self.use_relative_coordinates = use_relative_coordinates
        self.use_jcd_features = use_jcd_features
        self.use_coord_diff = use_coord_diff
        self.use_bone_angles = use_bone_angles
        self.use_bone_angles_diff = use_bone_angles_diff
        
        assert use_rotations in ['by_positive', 'by_batch', 'by_sample', None], 'Rotation mode [{}] not handled'.format(use_rotations)
        self.use_rotations = use_rotations
        self.rotation_noise = rotation_noise
        
        if joints_format == 'common':
#  F-PHAB (21 giunti)         
#   0   |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |  20 |
# [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]

# common_pose (20 giunti)
#   0   |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |  11 | 12 | 13 | 14  |  15 |  16 |  17 |  18 |  19 |
# [Wrist,TPIP, TDIP, TTIP, IMCP, IPIP, IDIP, ITIP, MMCP, MPIP, MDIP, MTIP, RMCP,RPIP, RDIP, RTIP, PMCP, PPIP, PDIP, PTIP]
            self.wrist_kp, self.middle_base_kp = 0, 8
            self.thumb_base_kp, self.index_base_kp, self.ring_base_kp = 1, 4, 12
            self.joints_num = 20
            connecting_joint = [1, # wrist
                                0, 1, 2, # thumb
                                0, 4, 5, 6, # index
                                0, 8, 9, 10, # middle
                                0, 12, 13, 14, # ring
                                0, 16, 17, 18 # pinky
                                ]
        elif joints_format == 'common_minimal':
            self.min_common_joints = [0, 8,     # Wrist, middle_base
                                      3,7,11,15,19,         # Finger tops
                                      ]
            self.joints_num = 7
            connecting_joint = [2,0,     # wrist
                                 1,1,1,1,1]
            self.wrist_kp, self.middle_base_kp = 0, 1
        
        self.joints_format = joints_format
        print(' * Using joints format:', joints_format)
        
        self.max_seq_len = max_seq_len
        # self.joints_num = joints_num
        self.joints_dim = 3
        # self.center_skels = center_skels
        self.scale_by_torso = scale_by_torso
        self.temporal_scale = temporal_scale
        # self.use_scaler = use_scaler
        # self.use_jcd_features = use_jcd_features
        # self.use_coord_diff = use_coord_diff
        # self.use_coords_raw = use_coords_raw
        # self.use_coords = use_coords
        # self.use_jcd_diff = use_jcd_diff
        # self.use_bone_angles = use_bone_angles
        # self.use_bone_angles_diff = use_bone_angles_diff
        # self.use_bone_angles_cent = use_bone_angles_cent
        self.skip_frames = skip_frames
        
        self.connecting_joint = connecting_joint
#Indice 0: L'articolazione 0 (polso) è connessa all'articolazione 2.
#Indice 1: L'articolazione 1 (base del dito medio) è connessa all'articolazione 0 (polso).
#Indici 2-6: Le articolazioni 2-6 (le punte delle dita) sono tutte connesse all'articolazione 1 (base del dito medio).
        if connecting_joint is not None: self.num_feats = self.get_num_feats()
        else: self.num_feats = None
        
        if noise is not None:
            self.add_coord_noise = True
            self.noise_type, self.noise_strength = noise[0], noise[1]
            print('Noise will be applied to training:', noise)
        else: self.add_coord_noise = False
        

#carica le coordinate scheletriche da un file (seleziona 7 joint) e le trasforma in un array Numpy
    def load_skel_coords(self, filename):
        with open(filename, 'r') as f: skel = f.read().splitlines()
        skel = np.array(list(map(str.split, skel)))
        # skel = np.reshape(skel, (len(skel), self.joints_num, self.joints_dim)).astype('float32')
        skel = np.reshape(skel, (len(skel), skel.shape[1]//self.joints_dim, self.joints_dim)).astype('float32')
        if self.joints_format == 'common_minimal':
            skel = skel[:, self.min_common_joints, :]
        return skel        

#scala le coordinate scheletriche in base alla distanza tra il polso e la base del dito
    def scale_skel(self, skel):
        torso_dists = np.linalg.norm(skel[:,self.middle_base_kp] - skel[:,self.wrist_kp], axis=1)     # length between wrist and middle finger base
        for i in range(skel.shape[0]): 
            rel = 1.0 / torso_dists[i] if torso_dists[i] != 0 else 1
            skel[i] = skel[i] * rel
        return skel
        

#total feature 54 -> (7 x 3 relative hand coordinates, 7 x 3 coordinate difference features, and 6 x 2 bone angle differences)
    def get_num_feats(self):
        num_feats = 0
        if self.use_bone_angles: 
            num_feats += (len(self.connecting_joint)-1)*2
        if self.use_bone_angles_diff: #6*2 - These features describe the rotation direction (with respect to the world coordinates) and rotation speed of each bone
            num_feats += (len(self.connecting_joint)-1)*2
        if self.use_jcd_features: 
            num_feats += int(comb(self.joints_num,2))
        if self.use_coord_diff: #7*3 - These features describe the translation direction and speed of each coordinate for each of the 3 axes
            num_feats += self.joints_num * self.joints_dim
        if self.use_relative_coordinates: #7*3
            num_feats += self.joints_num * self.joints_dim
        return num_feats
  

    # Crop movement to max_seq_len frames
    def zoom_to_max_len(self, p, force=False):  
        # Resize movement
        num_frames = p.shape[0]
        if force or num_frames > self.max_seq_len:
            # Zoom -> crop movement
            p_new = np.zeros([self.max_seq_len, self.joints_num, self.joints_dim], dtype="float32") 
            for m in range(self.joints_num):
                for n in range(self.joints_dim):
                    # smooth coordinates
                    # p_new[:,m,n] = medfilt(p_new[:,m,n], 3)
                    # Zoom coordinates to fit the max_seq_len_shape
                    p_new[:,m,n] = inter.zoom(p[:,m,n], self.max_seq_len/num_frames)[:self.max_seq_len]   # , mode='nearest'
        else:
            p_new = p
        return p_new

#vengono calcolate le differenze tra le coordinate delle articolazioni -> l'obiettivo è descrivere la struttura instantanea della mano
    def get_jcd_features(self, p, num_frames):
        # Get joint distances
        jcd = []
        iu = np.triu_indices(self.joints_num, 1, self.joints_num)
        for f in range(num_frames): 
            d_m = cdist(p[f],p[f],'euclidean')       
            d_m = d_m[iu] 
            jcd.append(d_m)
        jcd = np.stack(jcd) 
        return jcd
    
#utili per calcolare gli angoli delle ossa
    def get_bone_spherical_angles(self, v):
        elevation = np.arctan2(v[:,2], np.sqrt(v[:,0]**2 + v[:,1]**2))
        azimuth = np.arctan2(v[:,1], v[:,0])
        return np.column_stack([elevation, azimuth])
    def get_body_spherical_angles(self, body):
        angles = np.column_stack([ self.get_bone_spherical_angles(body[:, i+1] - body[:, i]) for i in range(len(self.connecting_joint)-1) ])
        return angles

    def average_wrong_frame_skels(self, skels):
        good_frames = np.all(~np.all(skels==0, axis=2), axis=1)
        for num_frame, gf in enumerate(good_frames):
            if gf: continue
            if num_frame == 0: skels[num_frame] = skels[num_frame+1]
            elif num_frame == len(skels)-1: skels[num_frame] = skels[num_frame-1]
            else: skels[num_frame] = (skels[num_frame+1] + skels[num_frame-1])/2
        return skels

    
    
  
    def get_random_rotation_matrix(self):
        return R.random().as_matrix()
    def get_constrained_rotation_matrix(self, angle_noise):
        return R.from_euler('xyz', np.random.uniform(-angle_noise,angle_noise,[3]), degrees=True).as_matrix()
    def rotate_sequence(self, skels, rot_matrix): 
        return np.matmul(skels, rot_matrix)
    
    # Move skels to the coordinate center. Coordinates relative to the palm center
    # estrae le coordinate della base delle dite e poi quelle del polso per ogni frame e 
    # calcola il punto medio che rappresenterà il centro del palmo
    def get_relative_coordinates(self, skels):
        skels_centers = (skels[:, self.middle_base_kp, :] + skels[:, self.wrist_kp, :])/2
        return skels - np.expand_dims(skels_centers, axis=1) 
    #Sottrae il centro del palmo dalle coordinate delle articolazioni, 
    # rendendole relative al centro del palmo
        
    
    
    
    # Input sequence -> (num_frame, num_joints, joints_dim)
    def get_pose_data_v2(self, body, validation, rotation_matrix=None):
    
        # Stampa i dati originali
        #print("Original body data:", body)
        #print("Original body shape:", body.shape)

        # 1. Remove frames without predictions
        #body = body[np.all(~np.all(body==0, axis=2), axis=1)]
        #print("Body shape after removing frames without predictions:", body.shape)
        #body è un array di elementi che rappresentano un frame che contiene coordinate 3D 
        # delle articolazioni della mano [num_frames, joints_num, joints_dim]        

# =============================================================================
# DATA AUGMENTATION
#   Skip frames, temporal scaling, sequence cropping, 
#   scale by torso, random noise, sequence rotation
# =============================================================================
        # Slect skipping frames (viene messo a 3 nei modelli pre-trainati)
        if len(self.skip_frames) > 0:
            sk = np.random.choice(self.skip_frames)
        else: sk = 1
        
        
        # 2. Reduce or extend the movement by interpolation. 
        # Ensures that the final movement will have at least 2 frames after skipping
# Aiuta a gestire la lunghezza delle sequenze. 
# In scenari pratici, le azioni possono avere durate diverse; 
# questa tecnica garantisce che il modello impari solo dalle parti rilevanti della sequenza.
        if not validation and self.temporal_scale is not False:
            orig_new_frames = len(body)
            temporal_scale = list(self.temporal_scale)
            temporal_scale[0] = int(temporal_scale[0]*orig_new_frames)
            temporal_scale[1] = int(temporal_scale[1]*orig_new_frames)
            # new_num_frames = min(np.random.randint(*temporal_scale), max_seq_len)
            new_num_frames = np.random.randint(*temporal_scale)
            new_num_frames = max(new_num_frames, 2*sk)
            
            zoom_factor = new_num_frames/orig_new_frames
            body = inter.zoom(body, (zoom_factor,1,1), mode='nearest') 
            

        # 3. Reduce frame rate
        # Ensures that the final movement will have at least 2 frames after skipping
# Riduce la quantità di dati ridondanti, 
# velocizzando l'addestramento e migliorando l'efficienza del modello. 
# Inoltre, rende il modello più robusto nel caso di sequenze video con framerate variabili.
        if len(self.skip_frames) > 0:
            # sk = np.random.choice(self.skip_frames)
            if validation: sk_init = 0
            else: sk_init = np.random.randint(sk)
            if len(body[sk_init::sk]) >= 2: body = body[sk_init::sk]


        # 4. Modify movement speed
# Consente di simulare sequenze con durate e velocità diverse, 
# rendendo il modello meno sensibile alla velocità del movimento. 
# Questo è utile per riconoscere azioni che possono essere eseguite a ritmi diversi.
        if self.max_seq_len > 0:
            body = self.zoom_to_max_len(body)
        elif self.max_seq_len < 0:
            if not validation:
                start = np.random.randint(max(len(body)-abs(self.max_seq_len)+1, 1))
                end = start + abs(self.max_seq_len)
                body = body[start:end]
            else:
                start = max(0, (len(body) - abs(self.max_seq_len)) // 2)
                end = start + abs(self.max_seq_len)
                body = body[start:end]
        #print("Body shape after modifying movement speed:", body.shape)

        # 5. Scale by torso
        if self.scale_by_torso: body = self.scale_skel(body)        
        
        # 6. Add random noise and scales again
# Introduce variabilità nelle coordinate, simulando errori di acquisizione dei dati dai sensori. 
# Questo rende il modello più resiliente a dati rumorosi o imprecisi.        
        if not validation and self.add_coord_noise:
            # print('Adding coord noise')
            if self.noise_type == 'uniform':
                noise = np.random.uniform(low=-self.noise_strength, high=self.noise_strength, size=body.shape)
            elif self.noise_type == 'normal':
                noise = np.random.normal(loc=0, scale=self.noise_strength, size=body.shape)
            else: raise ValueError('noise type [{}] not handled'.format(self.noise_type))
            body = body + noise
            if self.scale_by_torso: body = self.scale_skel(body)        
           
        
        # Rotate sequence
        if not validation and self.use_rotations is not None:
            if rotation_matrix is None: rotation_matrix = self.get_random_rotation_matrix()
            # print('Rotating', self.use_rotations, rotation_matrix[0])
            body = self.rotate_sequence(body, rotation_matrix)
            
        # Rotation noise
# Permette al modello di gestire variazioni di punto di vista, 
# che sono comuni in applicazioni reali (ad esempio, diverse angolazioni della telecamera).
        if not validation and self.rotation_noise is not None and \
            self.rotation_noise is not False and  self.rotation_noise>0:
            rotation_matrix = self.get_constrained_rotation_matrix(self.rotation_noise)
            body = self.rotate_sequence(body, rotation_matrix)

        
        
# =============================================================================
# FEATURE GENERATION
#         8. Get movement features
#             Relative coordinates
#             JCD, coord_diff
#             bone_angles, bone_angles_diff
# =============================================================================

        num_frames = len(body)
        #print("Number of frames:", num_frames)
        pose_features = []
        
        if self.use_relative_coordinates:
            rel_coordinates = self.get_relative_coordinates(body)
            #print("Relative coordinates shape:", rel_coordinates.shape)
            pose_features.append(np.reshape(rel_coordinates, (num_frames, self.joints_num * self.joints_dim)))

        if self.use_jcd_features:
            jcd_features = self.get_jcd_features(body, num_frames)
            #print("JCD features shape:", jcd_features.shape)
            pose_features.append(jcd_features)
            
        if self.use_coord_diff:
            if num_frames > 1:
                speed_features = body[1:] - body[:-1]
                #print("Speed features shape before reshape:", speed_features.shape)
                speed_features = np.reshape(speed_features, (num_frames-1, self.joints_num*self.joints_dim))
                #print("Speed features shape after reshape:", speed_features.shape)
                speed_features = np.concatenate([np.expand_dims(speed_features[0], axis=0), speed_features], axis=0)
                #print("Speed features shape after concatenate:", speed_features.shape)
                pose_features.append(speed_features)
            else:
                speed_features = np.zeros((num_frames, self.joints_num*self.joints_dim))
                #print("Speed features shape (zeros):", speed_features.shape)
                pose_features.append(speed_features)        
        
        if self.use_bone_angles or self.use_bone_angles_diff:
            bone_angles = self.get_body_spherical_angles(body)
            #print("Bone angles shape:", bone_angles.shape)
            if self.use_bone_angles_diff:
                if num_frames > 1:
                    bone_angles_diff = bone_angles[1:] - bone_angles[:-1]
                    #print("Bone angles diff shape before concatenate:", bone_angles_diff.shape)
                    bone_angles_diff = np.concatenate([np.expand_dims(bone_angles_diff[0], axis=0), bone_angles_diff], axis=0)
                    #print("Bone angles diff shape after concatenate:", bone_angles_diff.shape)
                    pose_features.append(bone_angles_diff)
                else:
                    bone_angles_diff = np.zeros((num_frames, (len(self.connecting_joint)-1)*2))
                    #print("Bone angles diff shape (zeros):", bone_angles_diff.shape)
                    pose_features.append(bone_angles_diff)
            if self.use_bone_angles:
                pose_features.append(bone_angles)        
            

        # Create features array -> (num_frames, num_features)
        pose_features = np.concatenate(pose_features, axis=1).astype('float32')
        #print("Pose features shape:", pose_features.shape)

            
        
        return pose_features
    
    

    # Triplet data generator -> genera batch di dati per l'addestramento. Ogni batch è composto da
    #campioni di diverse classi. La funzione legge le annotazioni, carica i dati delle pose, applica
    #trasformazioni e augmentazioni e prepara i batch di dati.

    # Each batch is composed by K=4 samples of P=batch_size/K different classes
    # if max_seq_len == 0 -> samples inside a batch are zero-padded to fit their inner max length. 
    #                           Longer sequences are zoomed out to fit max_seq_len
    # if max_seq_len > 0 -> samples inside a batch are zoomed-out to fit max_seq_len
    # if max_seq_len < 0 -> samples bigger than max_seq_len are randomly cropped to fit -max_seq_len
    # @threadsafe_generator	
    def triplet_data_generator(self, pose_annotations_file, 
                               batch_size, 
                               in_memory_generator, 
                               validation,
                               decoder, reverse_decoder,
                               triplet, 
                               classification, num_classes,
                               
                               
                               # skip_frames = [],
                               average_wrong_skels = True,
                               is_tcn=False,
                               K=4,
                               in_memory_skels=False,
                               sample_repetitions=1,
                               **kwargs):
        
        
            # Reads the annotations and stores them into a dict by label. Annotations are shuffled
            def read_annotations():
                pose_files = {} #le chiavi sono le etichette e i valori sono liste di nomi di file
                with open(pose_annotations_file, 'r') as f: 
                    for line in f:
                        filename, label = line.split()
                        label = int(label)
                        if label in pose_files: pose_files[label].append(filename)
                        else: pose_files[label] = [filename]
                for k in pose_files.keys(): np.random.shuffle(pose_files[k])
                return pose_files
        
            # Return a random sample with the given label or a random one if there is no 
            # more samples with that label
            def get_random_sample(label):
                if label in pose_files and len(pose_files[label]) > 0:
                    return pose_files[label].pop(), label
                else: 
                #se non ci sono più campioni con quella etichetta, viene selezionata una nuova etichetta casuale
                    if label in pose_files: del pose_files[label]
                    new_label = np.random.choice(list(pose_files.keys()))
                    return get_random_sample(new_label)        
            
            if in_memory_generator: 
                print(' ** Data Generator | data will be cached | Validation: {} **'.format(validation))
                cached_data = {}    
            if in_memory_skels: 
                print(' ** Data Generator | skeleton sequences be cached | Validation: {} **'.format(validation))
                cached_skels = {}
                
            if validation: sample_repetitions = 1
    
            
            if validation:
                batch_size = batch_size // K
                K = 1

            #K = (numero di campioni per classe)
            #P = (numero di classi per batch)
            #sample_repetitions = (numero di ripetizioi per campione)
            assert batch_size % K == 0
            P = batch_size // K
            pose_files = read_annotations()
            print('*************', K, P, batch_size, self.use_rotations)

            #Con classification a true -> le etichette delle classi vengono convertite in una rappresentazione
            # one-hot, e quest'ultime vengono aggiunte all'array Y.
            if classification:
                total_labels = sorted(list(pose_files.keys()))
                labels_dict = { l:i for i,l in enumerate(total_labels) }
                
                
            rotation_matrix = None
            print(self.use_rotations)
            print(' *** batch_size: {} - K: {} - P: {} - sample_repetitions: {}'.format(
                batch_size, K, P, sample_repetitions))
            
            while True:
                if sum([ len(v) for v in pose_files.values() ]) < batch_size:
                    pose_files = read_annotations()
                    
                batch_labels = []
                batch_samples = []
                if classification: y_clf = []
                if not validation and self.use_rotations == 'by_batch': rotation_matrix = self.get_random_rotation_matrix()
               
                
                # if triplet and triplet_individual_labels: label_ind: 0
                #Se triplet è True -> vengono generati gruppi di campioni positivi e negativi per ogni classe
                # utile per apprendere rappresentazioni discriminative
                if triplet:
                    # Positive pairs rotated together must have the same label
                    # Samples not rotated, rotated equally within batch or rotated randomly must have the original label
                    triplet_labels = []
                    if self.use_rotations == 'by_positive': triplet_label_ind = 0
                
                for num_p in range(P):      # For each group of triplet classes
                    # Get a random positive class
                    # label_iter = np.random.choice(list(pose_files.keys()))  
                    if triplet:
                        available_classes = [ c for c in pose_files.keys() if c not in list(set(batch_labels)) ]
                    if not triplet or len(available_classes) == 0:
                        available_classes = list(pose_files.keys())
                    label_iter = np.random.choice(available_classes)  # Random positive class
                    
                    if not validation and self.use_rotations == 'by_positive': 
                        rotation_matrix = self.get_random_rotation_matrix()
                        triplet_label_ind += 1
                    
                    for i in range(K):              # For each positive sample within positive group     
                        filename, label = get_random_sample(label_iter)
                        
                        for num_rep in range(sample_repetitions):
                            if classification:      # Get classification y_true
                                label_cat = to_categorical(labels_dict[int(label)], num_classes=num_classes)                
                        
                            if in_memory_generator and filename in cached_data.keys():
                                # Get sample from cache
                                sample = cached_data[filename]
                            else:
                                if in_memory_skels and filename in cached_skels:
                                    p = cached_skels[filename]
                                else:
                                    # Calculate (and store) new sample features
                                    p = self.load_skel_coords(filename)
                                    if average_wrong_skels: p = self.average_wrong_frame_skels(p)
                                    if in_memory_skels: cached_skels[filename] = p
                                sample = self.get_pose_data_v2(p, validation, rotation_matrix=rotation_matrix)
                                if in_memory_generator: cached_data[filename] = sample
                                
                            batch_samples.append(sample)
                            batch_labels.append(label)
                            if triplet:
                                if not validation and self.use_rotations == 'by_positive': triplet_labels.append(triplet_label_ind)
                                else: triplet_labels.append(label)
                            
                            if classification: y_clf.append(label_cat)   
                            
                    
                
                # Pack triplet labels and classification y_true
                if triplet: 
                    batch_labels = np.stack(batch_labels)       # for triplets
                    triplet_labels = np.stack(triplet_labels)       # for triplets

                #vedere sopra cosa comporta classification a true
                if classification: y_clf = np.stack(y_clf).astype('int')              # for classification                
                    
                X, Y, sample_weights = [], [], {}
                X = pad_sequences(batch_samples, abs(self.max_seq_len), padding='pre', dtype='float32')    # Pack NN input            
                    
                # if triplet: Y.append(batch_labels)
                if triplet: Y.append(triplet_labels)
                if classification: Y.append(y_clf)
                #se decoder è True -> il generatore prepara i dati per un compito di decodifica sequenziale
                if decoder:
                    decoder_data = [ bs[::-1] for bs in batch_samples ] if reverse_decoder else batch_samples
                    padding = 'pre' if is_tcn else 'post'
                    # decoder_data = pad_sequences(decoder_data, padding='post', dtype='float32')
                    decoder_data = pad_sequences(decoder_data, padding=padding, dtype='float32')
                    Y.append(decoder_data)
                    sample_weights['output_{}'.format(len(Y))] =  (decoder_data[:, :, 0] != 0).astype('float32')
                    
                    # if reverse_decoder: Y.append(batch_samples[:, ::-1, :])
                    # else: Y.append(batch_samples)
                    # sample_weights['output_{}'.format(len(Y))] =  (Y[-1][:, :, 0] != 0).astype('float32')
                    
                Y = np.concatenate(Y)
                yield X, Y              
           
        # return aux()


if __name__ == '__main__':

    joints_num = 20
    gen_params = {'max_seq_len': 32,
                    'scale_by_torso': True, 
                    # 'use_rotations': None, 
                    'use_rotations': 'by_positive', 
                    # 'use_rotations': 'by_batch', 
                    # 'use_rotations': 'by_sample', 
                    'rotation_noise': 20,
                    
                    'use_relative_coordinates': True,
                    'use_jcd_features': True, 
                    'use_coord_diff': True,
                    'use_bone_angles': True,
                    'use_bone_angles_diff': True,
                    
                    'skip_frames': [2,3],
                    # 'skip_frames': [],
                    'temporal_scale': (0.8,1.2), 
                    # 'temporal_scale': False, 
                    'dataset': 'CP_',
                    # 'noise': None,
                    # 'noise': ('normal', 0.03),
                    # 'noise': ('uniform', 0.03),
                    'joints_format': 'mpii' if joints_num==21 else 'common',

                    }
    
    
    
    data_gen = DataGenerator(**gen_params)

    body = np.random.rand(4, joints_num, 3)
    p = data_gen.get_pose_data_v2(body.copy(), validation = False)
    print(p.shape)
    
    self = data_gen
    

    gen_params = {
            'pose_annotations_file': './dataset_scripts/common_pose/annotations/F_PHAB/annotations_train_jn20.txt',
            'batch_size': 6,
            'in_memory_generator': True, 
            # 'validation': True,
            'validation': False,
            'decoder': None, 'reverse_decoder': None,
            'triplet': True,
            'classification': False,
            'num_classes': 45,
            
            'sample_repetitions': 1,    
            'K': 2
        }

    triplet_gen = data_gen.triplet_data_generator(**gen_params)
    for i in range(3):
        batch_X, batch_Y = next(triplet_gen)
        # batch_X, batch_Y, batch_sample_weights = next(triplet_gen)
        # batch_X, batch_Y, batch_sample_weights, batch_rot = next(triplet_gen)
        batch_Y = batch_Y[0]