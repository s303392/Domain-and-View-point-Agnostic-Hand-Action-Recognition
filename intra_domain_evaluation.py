import os
import numpy as np
from tensorflow.keras.models import load_model
from data_generator import DataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import prediction_utils

# Configurazioni
MODEL_PATH = './pretrained_models/intradom_SHREC/labels_28'
ANNOTATION_FILE = './dataset_scripts/mySHREC-17/tot_SHREC_annotation.txt'
OUTPUT_RESULTS = './classification_results.txt'
#MAX_SEQ_LEN = 32  # Lunghezza massima delle sequenze
JOINTS_NUM = 7  # Numero di giunti in formato common_minimal

# # Parametri per il DataGenerator (derivati dal modello preaddestrato)
# DATA_GEN_PARAMS = {
#     "max_seq_len": MAX_SEQ_LEN,
#     "scale_by_torso": True,
#     "temporal_scale": [0.6, 1.4],
#     "use_rotations": None,
#     "use_relative_coordinates": True,
#     "use_jcd_features": False,
#     "use_coord_diff": True,
#     "use_bone_angles": False,
#     "use_bone_angles_diff": True,
#     "joints_format": "common_minimal",
#     "rotation_noise": 10,
#     "noise": ["uniform", 0.04],
#     "skip_frames": [3]
# }

# Carica il modello preaddestrato
model, model_params = prediction_utils.load_model(MODEL_PATH, return_sequences=True, loss_name=None)
#print("MODEL LOADED:", model_params)

def load_annotations(annotation_file):
    """
    Carica il file di annotazione SHREC_annotation.txt.
    """
    with open(annotation_file, "r") as f:
        lines = f.readlines()
    annotations = [(line.split()[0], int(line.split()[1])) for line in lines]
    return annotations

annotations = load_annotations(ANNOTATION_FILE)

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

file_path = './datasets/SHREC2017/data_action_recognition.txt'
a_mapping_fphab = create_action_mapping(file_path)

# Inizializza il DataGenerator
data_gen = DataGenerator(**model_params['backbone_params'])

def preprocess_sequences(annotations, data_gen):
    """
    Preprocessa le sequenze scheletriche utilizzando il DataGenerator.
    """
    sequences = []
    labels = []
    max_seq_len = abs(model_params['backbone_params']['max_seq_len'])  # Accedi a max_seq_len da backbone_params
    for filepath, label in annotations:
        skeleton_sequence = data_gen.load_skel_coords(filepath)  # Carica i dati
        preprocessed_sequence = data_gen.get_pose_data_v2(skeleton_sequence, validation=True)  # Preprocessing
        padded_sequence = pad_sequences([preprocessed_sequence], maxlen=max_seq_len, dtype="float32", padding="pre")
        sequences.append(padded_sequence[0])  # Aggiunge la sequenza preprocessata
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequences, labels = preprocess_sequences(annotations, data_gen)

def evaluate_model(model, sequences, labels):
    """
    Predizione e valutazione
    """
    # Effettua la previsione
    probabilities = model.predict(sequences)
    
    # Stampa il tipo di probabilities
    print(f'Type of probabilities: {type(probabilities)}')
    
    # Se probabilities è una lista, stampa il contenuto
    if isinstance(probabilities, list):
        print(f'Length of probabilities list: {len(probabilities)}')
        for i, prob in enumerate(probabilities):
            print(f'Shape of probabilities [{i}]: {np.array(prob).shape}')
    
    # Converti la lista in un array numpy se necessario
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)
    
    # Rimuovi la dimensione extra se necessario
    probabilities = np.squeeze(probabilities)
    
    # Assicurati che le probabilità abbiano la forma corretta
    if probabilities.shape[0] != len(sequences):
        raise ValueError(f'Expected probabilities to have shape ({len(sequences)}, ...), but got {probabilities.shape}')
    
    predictions = np.argmax(probabilities, axis=1)
    
    # # Stampa un campione delle etichette predette
    # print(f'Sample of predictions before offset: {predictions[:10]}')
    
    # Aggiungi un offset di 1 alle etichette predette (sono mappate da 0 a 27 e non da 1 a 28 come le true label)
    predictions = predictions + 1
    
    # # Stampa un campione delle etichette predette dopo l'offset
    # print(f'Sample of predictions after offset: {predictions[:10]}')
    
    # Verifica la lunghezza e la forma delle previsioni e delle etichette
    print(f'Shape of sequences: {sequences.shape}')
    print(f'Shape of labels: {labels.shape}')
    print(f'Shape of predictions: {predictions.shape}')
    
    accuracy = accuracy_score(labels, predictions)
    return predictions, accuracy, probabilities

predictions, accuracy, probabilities = evaluate_model(model, sequences, labels)

# Stampa i risultati
print("Risultati della valutazione del modello")
print(f'Accuracy: {accuracy:.2f}')

# Salva i risultati nel file di output
with open(OUTPUT_RESULTS, "w") as f:
    for i, (pred, true_label) in enumerate(zip(predictions, labels)):
        pred_action_name = get_action_name(a_mapping_fphab, int(pred))
        true_action_name = get_action_name(a_mapping_fphab, int(true_label))
        #print(f"Predizione: {pred_action_name}, Etichetta corretta: {true_action_name}")
        f.write(f"Sample {i}: True Label: {true_action_name}, Predicted: {pred_action_name} \n ")
        #f.write(f"Sample {i}: True Label: {true_label}, Predicted: {pred} \n ")
    f.write(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")

if __name__ == "__main__":
    pass
