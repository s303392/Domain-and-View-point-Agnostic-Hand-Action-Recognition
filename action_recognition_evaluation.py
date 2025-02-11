#!/usr/bin/env python3
# -*- coding: utf -8 -*-
"""
Created on Thu? Jan 14 10:19:18 2021

@author: asabater
""" 

import random
import os
import pickle
import numpy as np
from tqdm import tqdm
import time
import sys
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data_generator import DataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences

#file progettato per valutare le prestazioni del modello.

knn_neighbors = [1,3,5,7,9,11]
# aug_loop = [0,10,20,40]
aug_loop = [0,40]
num_augmentations = max(aug_loop)
#num_augmentations = 0
#num_augmentations: si riferisce al numero di versioni augmentate 
# delle sequenze di azioni originali 
weights = 'distance'

crop_train = 80000
crop_test = np.inf

np.random.seed(0)


#funzione che valuta le prestazioni del modello su diversi foldi di dati utilizzando
# un classificatore k-NN.
def evaluate_folds(folds_data, embs, total_labels, num_augmentations=0, embs_aug=None, 
                   evaluate_all_folds=True, leave_one_out=True, groupby=None, return_sequences=False):
    
    res = {}
    all_preds = {}
    all_true = {}
    num_folds = len(folds_data) if evaluate_all_folds else 1
    print(f"Number of folds: {num_folds}")

    for num_fold in range(num_folds):
        if leave_one_out:
            train_indexes =  np.concatenate([ f['indexes'] for i,f in enumerate(folds_data) if i != num_fold])
            test_indexes = folds_data[num_fold]['indexes']
            print(f"Fold {num_fold} (leave_one_out=True) - Number of test sequences: {len(test_indexes)}")
        else:
            train_indexes = folds_data[num_fold]['indexes']
            test_indexes =  np.concatenate([ f['indexes'] for i,f in enumerate(folds_data) if i != num_fold])
            print(f"Fold {num_fold} (leave_one_out=False) - Number of test sequences: {len(test_indexes)}")
        X_train = embs[train_indexes]
        X_test = embs[test_indexes]
        y_train = total_labels[train_indexes]
        y_test = total_labels[test_indexes]
        
        if groupby is not None:
            _, groups_test_true = groupby[train_indexes], groupby[test_indexes]
            
        if num_augmentations > 0:
            X_train = np.concatenate([X_train] + [ embs_aug[i][train_indexes] for i in range(num_augmentations) ])
            y_train = np.concatenate([ y_train for i in range(num_augmentations+1) ])
 
        if return_sequences:
            if groupby is not None: groups_test_true = np.array([ y for y,seq in zip(groups_test_true, X_test) for _ in range(len(seq)) ])
            y_train = [ y for y,seq in zip(y_train, X_train) for _ in range(len(seq)) ]
            y_test = [ y for y,seq in zip(y_test, X_test) for _ in range(len(seq)) ]
            X_train = np.concatenate(X_train)
            X_test = np.concatenate(X_test)
            
        if groupby is not None and len(y_test) > crop_test:
            print('Cropping test results:', len(y_test))
            idx = np.random.choice(np.arange(len(y_test)), crop_test, replace=False)
            y_test = np.array(y_test)[idx].tolist()
            X_test = X_test[idx]
            groups_test_true = groups_test_true[idx]
 
        if len(y_train) > crop_train: 
            idx = np.random.choice(np.arange(len(y_train)), crop_train, replace=False)
            X_train = X_train[idx]
            y_train = np.array(y_train)[idx]

        res[num_fold] = {}
        all_preds[num_fold] = {}
        all_true[num_fold] = {}
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=8, weights=weights).fit(X_train, y_train)
        for n in knn_neighbors:
            knn = knn.set_params(**{'n_neighbors': n})
            
            t = time.time()
            if groupby is not None: 
                preds_proba = knn.predict_proba(X_test)
                classes = sorted(list(set(y_train)))
                groups = list(set(groups_test_true))
                g_true, g_preds = [], []
                for g in groups:
                    g_true.append(g.split('_')[1])
                    g_inds = np.where(groups_test_true == g)
                    
                    g_pred = preds_proba[g_inds].mean(axis=0)
                    g_preds.append(classes[np.where(g_pred == g_pred.max())[0][0]])
                    
                acc = accuracy_score(g_true, g_preds)
            else: 
                preds = knn.predict(X_test)
                acc = accuracy_score(y_test, preds)
                all_preds[num_fold][n] = preds
                all_true[num_fold][n] = y_test
            res[num_fold][n] = acc
            tf = time.time()
            print(' ** Classification time ** num_fold [{}] | k [{}] | X_test [{}] | time [{:.3f}s] | ms per sequence [{:.3f}]'.format(num_fold, n, 
                                                                       len(X_test), tf-t, (tf-t)*1000/len(X_test)))

    res = { n:np.mean([ res[num_fold][n] for num_fold in range(num_folds) ]) for n in knn_neighbors }
    
    return res, all_preds, all_true
        
# %%

# =============================================================================
# F-PHAB
# =============================================================================

#carica i dati del dataset F-PHAB, inclusi le annotazioni e i fold di valutazione
def load_fphab_data():
    # Load all annotations dal file total_annotations_jn20.txt, il quale contiene:
    # per ogni riga una sequenza di azione: percorso del file di scheletro e id azione
    annotations_store_folder = './dataset_scripts/F_PHAB/paper_tables_annotations'
    with open(os.path.join(annotations_store_folder, 'total_annotations_jn{}.txt'.format(20)), 'r') as f: 
        total_annotations = f.read().splitlines()
    total_labels = np.stack([ l.split()[-1] for l in total_annotations ])
    #Format of each line of skeleton.txt: t x_1 y_1 z_1 x_2 y_2 z_2 ... x_21 y_21 z_21
    #where t is the frame number and x_i y_i z_i are the world coordinates (in mm) of joint i at frame t.
    
    total_annotations = [ l.split()[0] for l in total_annotations ]
    
    
    # Load evaluation folds
    folds_1_1 = pickle.load(open(os.path.join(annotations_store_folder, 'annotations_table_2_cross-action_folds_1_1_jn{}.pckl'.format(20)), 'rb'))
    folds_base = pickle.load(open(os.path.join(annotations_store_folder, 'annotations_table_2_cross-action_folds_jn{}.pckl'.format(20)), 'rb'))
    folds_subject = pickle.load(open(os.path.join(annotations_store_folder, 'annotations_table_2_cross-person_folds_jn{}.pckl'.format(20)), 'rb'))
    return total_annotations, total_labels, folds_1_1, folds_base, folds_subject


from joblib import Parallel, delayed

#genera le sequenze di azioni utilizzando il DataGenerator
def load_actions_sequences_data_gen(total_annotations, num_augmentations, model_params, load_from_files=True, return_sequences=False):
    np.random.seed(0)
    data_gen = DataGenerator(**model_params)
    if return_sequences: data_gen.max_seq_len = 0
    
    if load_from_files: 
        skels_ann = [ data_gen.load_skel_coords(ann) for ann in total_annotations ]
    else: 
        skels_ann = total_annotations

    def get_pose_features(validation=False):
        action_sequences = [ data_gen.get_pose_data_v2(skels, validation=validation) for skels in skels_ann ]
        if not return_sequences: 
            action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')
        return action_sequences
    
    action_sequences = get_pose_features(validation=True)
    print('* Data sequences loaded')
    
    action_sequences_augmented = None
    if num_augmentations > 0:
        action_sequences_augmented = Parallel(n_jobs=8)(delayed(get_pose_features)(validation=False) for i in tqdm(range(num_augmentations)))
        print('* Data sequences augmented')
    else:
        print('* No augmentations applied')
    
    return action_sequences, action_sequences_augmented



def get_tcn_embeddings(model, action_sequences, action_sequences_augmented, return_sequences=False):
    t = time.time()
    
    #con False si utilizza il meccanismo di attenzione -> 
    # viene restituito solo l'ultimo elemento della sequenza
    model.set_encoder_return_sequences(return_sequences)
    # Get embeddings from all annotations
    if return_sequences: 
        # embs = np.array([ model.get_embedding(s[None]).numpy()[0] for s in action_sequences ])
        embs = [ model.get_embedding(s[None]) for s in action_sequences ]
    else: 
        embs = [ model.get_embedding(s) for s in np.array_split(action_sequences, max(1, len(action_sequences)//1)) ]
    print('* Embeddings calculated')
    if num_augmentations > 0:
        if return_sequences: 
            embs_aug = [ [ model.get_embedding(s[None]) for s in samples ] for samples in tqdm(action_sequences_augmented) ]
        else: 
            embs_aug = [ [ model.get_embedding(s) for s in np.array_split(samples, max(1, len(samples)//1)) ] for samples in tqdm(action_sequences_augmented) ]
        print('* Augmented embeddings calculated')
    else: embs_aug = None
    
    tf = time.time()
    sys.stdout.flush
    
    if return_sequences: embs = np.array([ e[0] for e in embs ])
    else: embs = np.concatenate([ e for e in embs ])
    if num_augmentations > 0:
        if return_sequences: embs_aug = np.array([ [ s[0] for s in samples ] for samples in embs_aug ])
        else: embs_aug = [ np.concatenate([ s for s in samples ]) for samples in embs_aug ]
    
    num_sequences = len(embs)
    if num_augmentations > 0: num_sequences += sum([ len(e) for e in embs_aug ])
    print(' ** Prediction time **   Secuences evaluated [{}] | time [{:.3f}s] | ms per sequence [{:.3f}]'.format(num_sequences, tf-t, (tf-t)*1000/num_sequences))
 
    return embs, embs_aug


def evaluate_fphab(aug_loop, folds_base, folds_1_1, folds_subject, embs, embs_aug, total_labels, knn_neighbors):
    total_res = {}
    all_preds = {}
    all_true = {}
    for n_aug in aug_loop:
        total_res[n_aug] = {}
        all_preds[n_aug] = {}
        all_true[n_aug] = {}
        print(n_aug, '1:3')
        total_res[n_aug]['1:3'], all_preds[n_aug]['1:3'], all_true[n_aug]['1:3'] = evaluate_folds(folds_base, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False)
        print(n_aug, '1:1')
        total_res[n_aug]['1:1'], all_preds[n_aug]['1:1'], all_true[n_aug]['1:1'] = evaluate_folds(folds_1_1, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False, evaluate_all_folds=False)
        print(n_aug, '3:1')
        total_res[n_aug]['3:1'], all_preds[n_aug]['3:1'], all_true[n_aug]['3:1'] = evaluate_folds(folds_base, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=True)
        print(n_aug, 'cross_sub')
        total_res[n_aug]['cross_sub'], all_preds[n_aug]['cross_sub'], all_true[n_aug]['cross_sub'] = evaluate_folds(folds_subject, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=True)

    return total_res, all_preds, all_true
 
#totale_res = {
   # 0: {  # No augmentations
   #     '1:3': { ... },
   #     '1:1': { ... },
   #    '3:1': { ... },
   #     'cross_sub': { ... }
   # },
   # 40: {  # With augmentations
   #     '1:3': { ... },
   #     '1:1': { ... },
   #     '3:1': { ... },
   #     'cross_sub': { ... }
   # }
#}
                            
#total_res[0].keys() -> (['1:3', '1:1', '3:1', 'cross_sub'])
def print_results(dataset_name, total_res, knn_neighbors, aug_loop, frame=True):
    if frame: print('-'*81)
    results = []
    for k in total_res[0].keys():
        max_acc = max((total_res[0][k][n], n) for n in knn_neighbors)
        max_aug_acc = max((total_res[na][k][n], n) for n in knn_neighbors for na in aug_loop)
        results.append('[{}] {:.1f} (k={}) / {:.1f} (k={})'.format(
            k, max_acc[0]*100, max_acc[1], max_aug_acc[0]*100, max_aug_acc[1]
        ))
    print('# | {} | {}'.format(dataset_name, ' | '.join(results)))
    if frame: print('-'*81)
    

def print_model_details(model_params):
    print("Model details of the model:")
    for key, value in model_params.items():
        print(f"{key}: {value}")


def create_action_mapping(file_path):
    action_mapping = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Training') or line.startswith('Test'):
                continue
            parts = line.strip().split()
            if '/' in parts[0]:
                action_name = parts[0].split('/')[1]  # Prendi solo il nome dell'azione
            else:
                action_name = parts[0]  # Prendi la parola prima dello spazio
            action_number = int(parts[-1])
            if action_number not in action_mapping:
                action_mapping[action_number] = action_name
    return action_mapping

def get_action_name(action_mapping, action_number):
    return action_mapping.get(action_number, "Unknown")

def save_embeddings_to_csv(embs, labels, output_path="embeddings.csv"):
    """
    Salva gli embeddings e le loro etichette in un file CSV.

    :param embs: array numpy con gli embeddings.
    :param labels: array numpy con le etichette delle azioni.
    :param output_path: percorso del file CSV in cui salvare i dati.
    """
    df = pd.DataFrame(embs)
    df["label"] = labels  # Aggiungiamo una colonna con le etichette
    df.to_csv(output_path, index=False)
    print(f"EMBEDDINGS SALVATI IN {output_path}")
    print(f"Shape degli embeddings: {embs.shape}")
    print(f"Shape delle etichette: {labels.shape}")

# SINGLE MY ACTION 

def map_model_params_to_mygenerator_params(model_params):
    gen_params = {
        'max_seq_len': model_params.get('max_seq_len', -32),
        'scale_by_torso': model_params.get('scale_by_torso', True),
        'use_velocity_acceleration': model_params.get('use_velocity_acceleration', False),
        'use_additional_features': model_params.get('use_additional_features', False),
        'joints_format': model_params.get('joints_format', 'common_minimal'),
        'temporal_scale': model_params.get('temporal_scale', (0.8, 1.2)),
        'noise': model_params.get('noise', None),
        'rotation_noise': model_params.get('rotation_noise', None),
        'use_rotations': model_params.get('use_rotations', None),
        'use_relative_coordinates': model_params.get('use_relative_coordinates', False),
        'use_jcd_features': model_params.get('use_jcd_features', False),
        'use_coord_diff': model_params.get('use_coord_diff', False),
        'use_bone_angles': model_params.get('use_bone_angles', False),
        'use_bone_angles_diff': model_params.get('use_bone_angles_diff', False),
        'skip_frames': model_params.get('skip_frames', []),
        'dataset': model_params.get('dataset', ''),
    }
    return gen_params

def load_single_action_data(annotation_file, model_params):
    gen_params = map_model_params_to_mygenerator_params(model_params)
    with open(annotation_file, 'r') as f:
        total_annotations = f.read().splitlines()
    total_labels = np.stack([l.split()[-1] for l in total_annotations])
    total_annotations = [l.split()[0] for l in total_annotations]
    data_gen = DataGenerator(**gen_params)
    action_sequences = []
    for ann in total_annotations:
        print(f"Loading skeleton coordinates from: {ann}")
        skel_joints = data_gen.load_skel_coords(ann)
        print(f"Skeleton joints shape: {skel_joints.shape}")
        pose_data = data_gen.get_pose_data_v2(skel_joints, validation=True)
        action_sequences.append(pose_data)
    print(f"Action sequence len: {len(action_sequences)}")
    action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')
    return action_sequences, total_labels

def get_embeddings(model, action_sequences):
    action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')
    action_sequences = np.array(action_sequences)
    embs = model.get_embedding(action_sequences)
    return embs

def evaluate_single_action(single_embs, single_labels, reference_embs, reference_labels):
    # combined_embs = np.concatenate([reference_embs, single_embs], axis=0)
    # combined_labels = np.concatenate([reference_labels, single_labels], axis=0)

    # print(f"Len combined_embs_0: {len(combined_embs)}")
    # print(f"Len combined_labels_0: {len(combined_labels)}")

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8, weights='distance').fit(reference_embs, reference_labels)
    preds = knn.predict(single_embs)
    acc = accuracy_score(single_labels, preds)
    return acc, preds, single_labels

# my Dataset

def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    annotations = [line.strip().split()[0] for line in lines]
    labels = np.array([int(line.strip().split()[1]) for line in lines])
    return annotations, labels

def get_embeddings_dataset(model, annotations, model_params):
    action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(annotations, num_augmentations, model_params)
    embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented)
    return embs, embs_aug

def evaluate_my_actions(model, model_params, my_actions_file, reference_actions_file, action_mapping, output_file):
    # Carica i dati di riferimento
    reference_annotations, reference_labels = load_annotations(reference_actions_file)
    reference_embs, ref_embs_aug = get_embeddings_dataset(model, reference_annotations, model_params)

    # Carica le tue registrazioni
    my_annotations, my_labels = load_annotations(my_actions_file)
    my_embs, my_embs_aug = get_embeddings_dataset(model, my_annotations, model_params)

    # Salva gli embeddings di riferimento in un CSV
    save_embeddings_to_csv(reference_embs, reference_labels, output_path="./results/embeddings/reference_embeddings.csv")

    # Salva gli embeddings delle tue registrazioni in un CSV
    save_embeddings_to_csv(my_embs, my_labels, output_path="./results/embeddings/my_embeddings.csv")

    print(f"Lunghezza ref_embs: {len(reference_embs)}")
    print(f"Lunghezza my_embs: {len(my_embs)}")
    
    if ref_embs_aug is not None:
        print(f"Lunghezza ref_embs_aug: {len(ref_embs_aug)}")
        # for i, aug_emb in enumerate(ref_embs_aug):
        #     print(f"Shape di ref_embs_aug[{i}]: {aug_emb.shape}")
    if my_embs_aug is not None:
        print(f"Lunghezza my_embs_aug: {len(my_embs_aug)}")
        # for i, aug_emb in enumerate(my_embs_aug):
        #     print(f"Shape di my_embs_aug[{i}]: {aug_emb.shape}")

    total_acc = 0
    num_evaluations = 0

    with open(output_file, 'w') as f:
        if my_embs_aug is not None:
            for aug_idx, aug_embs in enumerate(my_embs_aug):
                print(f"Valutazione delle sequenze augmentate {aug_idx + 1}")
                acc_aug, preds_aug, true_labels_aug = evaluate_single_action(aug_embs, my_labels, reference_embs, reference_labels)
                print(f"Accuratezza (augmentata {aug_idx + 1}): {acc_aug}")
                total_acc += acc_aug
                num_evaluations += 1
                f.write(f"Accuratezza (augmentata {aug_idx + 1}): {acc_aug}\n")
                for i in range(len(preds_aug)):
                    pred_action_name = get_action_name(action_mapping, int(preds_aug[i]))
                    true_action_name = get_action_name(action_mapping, int(true_labels_aug[i]))
                    f.write(f"Predizione (augmentata {aug_idx + 1}): {pred_action_name}, Etichetta corretta: {true_action_name}\n")
        else:
            # Valuta le tue registrazioni utilizzando il KNN addestrato sui dati di riferimento
            acc, preds, true_labels = evaluate_single_action(my_embs, my_labels, reference_embs, reference_labels)
            print(f"Accuratezza: {acc}")
            total_acc += acc
            num_evaluations += 1
            f.write(f"Accuratezza: {acc}\n")
            for i in range(len(preds)):
                pred_action_name = get_action_name(action_mapping, int(preds[i]))
                true_action_name = get_action_name(action_mapping, int(true_labels[i]))
                f.write(f"Predizione: {pred_action_name}, Etichetta corretta: {true_action_name}\n")

        # Calcola l'accuratezza totale combinata
        if num_evaluations > 0:
            total_acc /= num_evaluations
            print(f"ACCURATEZZA totale combinata: {total_acc}")
            f.write(f"ACCURATEZZA totale combinata: {total_acc}\n")

# %%

# =============================================================================
# SHREC 14/28
# =============================================================================

def load_shrec_data():
    base_path = './'
    shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('14', 20)
    shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('14', 20)
    with open(os.path.join(base_path+shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
    with open(os.path.join(base_path+shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
    
    # Load data annotations
    total_shrec_anns = shrec_train_anns + shrec_val_anns
    shrec_train_indexes = [ total_shrec_anns.index(ann) for ann in shrec_train_anns ]
    shrec_val_indexes = [ total_shrec_anns.index(ann) for ann in shrec_val_anns ]
    total_shrec_anns = np.stack([ l.split()[0] for l in total_shrec_anns ])

    # Create data folds
    shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('14', 20)
    shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('14', 20)
    with open(os.path.join(base_path+shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
    with open(os.path.join(base_path+shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
    total_shrec_labels_14 = np.stack([ l.split()[-1] for l in shrec_train_anns+shrec_val_anns ])

    #indici, annotazioni e etichette
    shrec_folds_14 = {
        0:{'indexes': shrec_train_indexes, 'annotations': total_shrec_anns[shrec_train_indexes], 'labels': total_shrec_labels_14[shrec_train_indexes]},
        1:{'indexes': shrec_val_indexes, 'annotations': total_shrec_anns[shrec_val_indexes], 'labels': total_shrec_labels_14[shrec_val_indexes]},
                   }
    
    # Create data folds
    shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('28', 20)
    shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('28', 20)
    with open(os.path.join(base_path+shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
    with open(os.path.join(base_path+shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
    total_shrec_labels_28 = np.stack([ l.split()[-1] for l in shrec_train_anns+shrec_val_anns ])

    #indici, annotazioni e etichette
    shrec_folds_28 = {
        0:{'indexes': shrec_train_indexes, 'annotations': total_shrec_anns[shrec_train_indexes], 'labels': total_shrec_labels_28[shrec_train_indexes]},
        1:{'indexes': shrec_val_indexes, 'annotations': total_shrec_anns[shrec_val_indexes], 'labels': total_shrec_labels_28[shrec_val_indexes]},
                   }
    
    return total_shrec_anns, shrec_folds_14, shrec_folds_28, total_shrec_labels_14, total_shrec_labels_28
 

def evaluate_shrec(aug_loop, shrec_folds_14, shrec_folds_28, embs, embs_aug, total_shrec_labels_14, total_shrec_labels_28, knn_neighbors):
    total_res_shrec = {}
    for n_aug in aug_loop:
        print('***', n_aug, '***')
        total_res_shrec[n_aug] = {}
        print(n_aug, '14')
        total_res_shrec[n_aug]['14'] = evaluate_folds(shrec_folds_14, embs, total_shrec_labels_14, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False, evaluate_all_folds=False)
        print(n_aug, '28')
        total_res_shrec[n_aug]['28'] = evaluate_folds(shrec_folds_28, embs, total_shrec_labels_28, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False, evaluate_all_folds=False)
    return total_res_shrec


# %%

# =============================================================================
# MSRA functions
# =============================================================================

def load_MSRA_data(model_params, n_splits=4, seq_perc=1, data_format = 'common_minimal'):
    from dataset_scripts.MSRA import load_data
    np.random.seed(0)
    if seq_perc == -1: total_data = load_data.actions_to_samples(load_data.load_data(data_format), -1)
    else: total_data = load_data.actions_to_samples(load_data.load_data(data_format), int(abs(model_params['max_seq_len']*model_params['skip_frames'][0]*seq_perc)))
    actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet = \
        load_data.get_folds(total_data, n_splits=n_splits)
    return actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet

def evaluate_MSRA(aug_loop, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet, 
                  embs, embs_aug, actions_labels, knn_neighbors, return_sequences=False):
    total_res = {}
    for n_aug in aug_loop:
        print('***', n_aug, '***')
        total_res[n_aug] = {}

        print(n_aug, 'posturenet')
        total_res[n_aug]['posturenet'] = evaluate_folds(folds_posturenet, embs, actions_labels, num_augmentations=n_aug, \
                                                        embs_aug=embs_aug, leave_one_out=False, \
                                                        evaluate_all_folds = False,
                                                        groupby=actions_label_sbj, return_sequences=return_sequences)
        print(n_aug, 'posturenet_online')
        total_res[n_aug]['posturenet_online'] = evaluate_folds(folds_posturenet, embs, actions_labels, num_augmentations=n_aug, \
                                                        embs_aug=embs_aug, leave_one_out=False, \
                                                        evaluate_all_folds = False,
                                                        groupby=None, return_sequences=return_sequences)


    return total_res


if __name__ == '__main__':
    # %%

    # =============================================================================
    # Load model
    # =============================================================================
    
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import prediction_utils
    import time
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate action recognition in different datasets')
    parser.add_argument('--path_model', type=str, help='path to the trained model')
    parser.add_argument('--loss_name', type=str, help='key to load weights')
    parser.add_argument('--eval_fphab', action='store_true', help='evaluate on F-PHAB splits')
    parser.add_argument('--eval_shrec', action='store_true', help='evaluate on SHREC splits')
    parser.add_argument('--eval_msra', action='store_true', help='evaluate on MSRA dataset')
    parser.add_argument('--single_action', type=str, help='path to the annotation file for a single action')
    parser.add_argument('--my_actions', type=str, help='path to the annotation file for your actions')
    parser.add_argument('--reference_actions', type=str, help='path to the annotation file for reference actions')
    args = parser.parse_args()

    model, model_params = prediction_utils.load_model(args.path_model, False, loss_name=args.loss_name)
    model_params['use_rotations'] = None
    
    print('* Model loaded')

    print_model_details(model_params)
    
    # Carica i dati di riferimento se specificati
    if args.reference_actions:
        reference_annotations, reference_labels = load_annotations(args.reference_actions)
        reference_embs, ref_embs_aug = get_embeddings_dataset(model, reference_annotations, model_params)
        print(f"Lunghezza reference_embs: {len(reference_embs)}")
    
    # myDATASET
    if args.my_actions and args.reference_actions:
        action_mapping_file_path = 'C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/datasets/myDataset/data_action_recognition.txt'
        action_mapping = create_action_mapping(action_mapping_file_path)

        OUTPUT_FILE = './results/classification/cross-domain/myDatasetClassification_CDAug_mod.txt'
        evaluate_my_actions(model, model_params, args.my_actions, args.reference_actions, action_mapping, OUTPUT_FILE)

    # SINGLE MY ACTION
    if args.single_action:
        file_path = 'C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/datasets/F-PHAB/data_split_action_recognition.txt'
        a_mapping = create_action_mapping(file_path)

        single_action_sequences, single_action_labels = load_single_action_data(args.single_action, model_params)
        single_action_embs = get_embeddings(model, single_action_sequences)

        if args.reference_actions:
            combined_embs = reference_embs
            combined_labels = reference_labels
            if num_augmentations > 0 and ref_embs_aug is not None:
                combined_embs = np.concatenate([reference_embs] + ref_embs_aug, axis=0)
                combined_labels = np.concatenate([reference_labels for _ in range(num_augmentations + 1)], axis=0)
        else:
            # Carico i dati reference di F-PHAB e calcolo i rispettivi embeddings
            total_annotations, total_labels, folds_1_1, folds_base, folds_subject = load_fphab_data()
            action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(total_annotations, num_augmentations, model_params)
            embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented)

            if num_augmentations > 0 and embs_aug is not None:
                combined_embs = np.concatenate([embs] + embs_aug, axis=0)
                combined_labels = np.concatenate([total_labels for _ in range(num_augmentations + 1)], axis=0)
            else:
                combined_embs = embs
                combined_labels = total_labels

        print('Len single_action_embs:{}', len(single_action_embs))
        print('Len label:{}', len(single_action_labels))

        # Valuto la singola azione
        acc, preds, true_labels = evaluate_single_action(single_action_embs, single_action_labels, combined_embs, combined_labels)
        print(f"Accuratezza: {acc}")
        pred_action_name = get_action_name(a_mapping, int(preds[0]))
        true_action_name = get_action_name(a_mapping, int(true_labels[0]))
        print(f"Predizione: {pred_action_name}, Etichetta corretta: {true_action_name}")

    # F-PHAB
    if args.eval_fphab:
        t = time.time()
        total_annotations, total_labels, folds_1_1, folds_base, folds_subject = load_fphab_data()
        print('Len total labels:{}', len(total_labels))#1175
        print('Len fold_1_1:{}', len(folds_1_1))#2
        print('Len fold_base:{}', len(folds_base))#3
        print('Len fold_subject:{}', len(folds_subject))#6
        action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(total_annotations, num_augmentations, model_params)
        print('Len action sequences:{}', len(action_sequences))#1175
        print(f"Shape action sequence: {action_sequences[0].shape}")#(32,54) #(frame, feature)
        print('Len action sequences augmented:{}', len(action_sequences_augmented))#10
        print(f"Shape action sequence aug: {action_sequences_augmented[0].shape}")#(1175,32,54) -> feature = (7 x 3 relative hand coordinates, 7 x 3 coordinate difference features, and 6 x 2 bone angle differences)
        embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented)
        print('Len embs:{}', len(embs))#1175
        print('Len embs_aug:{}', len(embs_aug))#10 (vettore con 10 embs augmentati)
        total_res_fphab, all_preds_fphab, all_true_fphab = evaluate_fphab(aug_loop, folds_base, folds_1_1, folds_subject, embs, embs_aug, total_labels, knn_neighbors)
        #print_results('F-PHAB   ', total_res_fphab, knn_neighbors, aug_loop)

        file_path = 'C:/Users/filip/Desktop/Politecnico/INGEGNERIA/TESI_loc/Sabater/Domain-and-View-point-Agnostic-Hand-Action-Recognition/datasets/F-PHAB/data_split_action_recognition.txt'
        a_mapping_fphab = create_action_mapping(file_path)
        #print(a_mapping_fphab) #{num_action: name_action, ..}        
        
        # Stampa un esempio di predizione con il nome dell'azione e la sua etichetta corretta
        for n_aug in all_preds_fphab:
            for fold_type in all_preds_fphab[n_aug]:
                for fold in all_preds_fphab[n_aug][fold_type]:
                    for k in all_preds_fphab[n_aug][fold_type][fold]:
                        preds = all_preds_fphab[n_aug][fold_type][fold][k]
                        true_labels = all_true_fphab[n_aug][fold_type][fold][k]
                        if len(preds) > 0:
                            i = random.choice(range(len(preds)))  # Seleziona un indice casuale
                            pred_action_name = get_action_name(a_mapping_fphab, int(preds[i]))
                            true_action_name = get_action_name(a_mapping_fphab, int(true_labels[i]))
                            print(f"Predizione: {pred_action_name}, Etichetta corretta: {true_action_name}")
                        break
                    break
                break
            break
                        
        # for n_aug in all_preds_fphab:
        #     print(f"Augmentation: {n_aug}")
        #     for fold_type in all_preds_fphab[n_aug]:
        #         print(f"  Fold type: {fold_type}")
        #         for fold in all_preds_fphab[n_aug][fold_type]:
        #             print(f"    Fold: {fold}")
        #             for k in all_preds_fphab[n_aug][fold_type][fold]:
        #                 print(f"      k={k}: {len(all_preds_fphab[n_aug][fold_type][fold][k])}")
        
        print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
        #print('Lunghezza total_res:{}', len(total_res_fphab))#2 (per gli aug_loop che è [0,10])
        

        del embs; del embs_aug; del action_sequences; del action_sequences_augmented

    # %%
    # SHREC
    if args.eval_shrec:
        t = time.time()
        total_shrec_annotations, shrec_folds_14, shrec_folds_28, total_shrec_labels_14, total_shrec_labels_28 = load_shrec_data()
        action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(total_shrec_annotations, num_augmentations, model_params)
        embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented)
        total_res_shrec = evaluate_shrec(aug_loop, shrec_folds_14, shrec_folds_28, embs, embs_aug, total_shrec_labels_14, total_shrec_labels_28, knn_neighbors)
        print_results('SHREC    ', total_res_shrec, knn_neighbors, aug_loop)
        print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
        
        del embs; del embs_aug; del action_sequences; del action_sequences_augmented;
    
    # %%
    # MSRA full -> return_sequences == True
    if args.eval_msra:
        t = time.time()
        model_params['skip_frames'] = [1]
        actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet = \
                                    load_MSRA_data(model_params, n_splits=4, 
                                                               # seq_perc=0.2, data_format=model_params['joints_format'])
                                                                seq_perc=-1, data_format=model_params['joints_format'])
        action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(actions_list, num_augmentations, model_params, 
                                                                           load_from_files=False, return_sequences=True)
        embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented, return_sequences=True)
        total_res_msra_full = evaluate_MSRA(aug_loop, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet,
                                            embs, embs_aug, actions_labels, knn_neighbors, return_sequences=True)
        print_results('MSRA     ', total_res_msra_full, knn_neighbors, aug_loop)
        print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
        del embs; del embs_aug; del action_sequences; del action_sequences_augmented;

    # %%
    
    print('='*80)
    print('='*80)
    print('='*80)
    print('='*80)
    if args.eval_fphab: print_results('F-PHAB   ', total_res_fphab, knn_neighbors, aug_loop, frame=False)
    if args.eval_shrec: print_results('SHREC    ', total_res_shrec, knn_neighbors, aug_loop, frame=False)
    if args.eval_msra: print_results( 'MSRA     ', total_res_msra_full, knn_neighbors, aug_loop, frame=False)
    
    print()
    print(args.loss_name, args.path_model)
    
# %%
