import sys
import argparse
import os
import subprocess
import socket
import time

# Configurazione IP e porta del robot (modifica questi valori)
ROBOT_IP = "127.0.0.1"  # Sostituisci con l'IP del robot
ROBOT_PORT = 5000           # Sostituisci con la porta corretta

# Parser per accettare il file CSV come input
parser = argparse.ArgumentParser(description="Esegue l'intera pipeline di elaborazione")
parser.add_argument('--csv_file', type=str, required=True, help="Percorso del file CSV della singola azione")
parser.add_argument('--output_dir', type=str, default="./results", help="Directory di output per i risultati")
args = parser.parse_args()

# Assicurarsi che la cartella di output esista
os.makedirs(args.output_dir, exist_ok=True)

# Percorsi assoluti agli script
script_dir = os.path.abspath("./dataset_scripts/common_pose")  
script_myAction = os.path.join(script_dir, "myAction_to_common_pose.py")
script_det_coordinates = os.path.abspath("./det_coordinates/det_coordinates.py")
script_action_recognition = os.path.abspath("./action_recognition_evaluation.py")

# Percorsi aggiuntivi richiesti dall'ultimo step
path_model = "./pretrained_models/xdom_summarization"
loss_name = "mixknn_best"
reference_actions = os.path.abspath("./dataset_scripts/myDataset/ref_ComPose_annotations.txt")

# Step 1: Conversione in common_pose
print("Eseguendo la conversione in common_pose...")
try:
    subprocess.run(["python", script_myAction, "--single_csv", args.csv_file], check=True)

    common_pose_file = os.path.join(
        "./datasets/common_pose/testSingleAction",
        os.path.basename(args.csv_file).replace(".csv", "_combined_skeleton.txt")
    )

    if not os.path.exists(common_pose_file):
        raise FileNotFoundError(f"Errore: il file {common_pose_file} non è stato creato.")
except Exception as e:
    print(f"Errore durante la conversione in common_pose: {e}", file=sys.stderr)
    sys.exit(1)

# Step 2: Calcolo delle coordinate
print("Calcolando le coordinate di inizio e fine azione...")
coordinates_file = os.path.join(args.output_dir, "coordinates_obj", "action_coordinates.txt")
try:
    subprocess.run(["python", script_det_coordinates, "--csv_file", args.csv_file], check=True)
except Exception as e:
    print(f"Errore durante il calcolo delle coordinate: {e}", file=sys.stderr)
    sys.exit(1)

# Step 3: Creazione del file di annotazione per l'ultimo step
annotation_file = os.path.join(args.output_dir, "single_action_annotation.txt")
try:
    with open(annotation_file, "w") as f:
        f.write(f"{common_pose_file} 9\n")  
except Exception as e:
    print(f"Errore durante la creazione del file di annotazione: {e}", file=sys.stderr)
    sys.exit(1)

# Step 4: Classificazione dell'azione
print("Classificando l'azione...")
try:
    subprocess.run([
        "python", script_action_recognition,
        "--path_model", path_model,
        "--loss_name", loss_name,
        "--single_action", annotation_file,
        "--reference_actions", reference_actions,
    ], check=True)
except Exception as e:
    print(f"Errore durante la classificazione dell'azione: {e}", file=sys.stderr)
    sys.exit(1)

# Step 5: Lettura dei dati e invio al robot
print("Preparazione dei dati per l'invio al robot...")
try:
    action_output_file = os.path.join(args.output_dir, "recognized_action.txt")
    
    # Aspetta fino a quando il file di output viene generato
    while not os.path.exists(action_output_file):
        time.sleep(0.1)  

    with open(action_output_file, "r") as f:
        action_name = f.read().strip()

    if not action_name:
            raise ValueError("Errore: il file recognized_action.txt è vuoto.")

    with open(coordinates_file, "r") as f:
        lines = f.readlines()
        if len(lines) < 6:
            raise ValueError("Errore: il file action_coordinates.txt non contiene abbastanza righe.")
        start_coords = lines[1].strip().split(": ")[1]
        end_coords = lines[3].strip().split(": ")[1]
        object_name = lines[5].strip()


    message = f"{action_name}|{object_name}|{start_coords}|{end_coords}|"
    print(f"Inviando dati al robot: {message}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ROBOT_IP, ROBOT_PORT))
    client_socket.sendall(message.encode('utf-8'))

    response = client_socket.recv(1024).decode('utf-8')
    print("Risposta dal robot:", response)

    client_socket.close()
except Exception as e:
    print(f"Errore durante l'invio dei dati al robot: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Pipeline completata con successo! Dati inviati al robot {ROBOT_IP}:{ROBOT_PORT}")
