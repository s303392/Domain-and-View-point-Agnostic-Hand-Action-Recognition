import sys
import os
import subprocess
import socket
import threading
import time
import csv

# Configurazione IP e porta del server TCP
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12345
BUFFER_SIZE = 1024  # Dimensione del buffer per ricevere i dati

# Flag per controllare il ciclo del server
running = True

# Funzione per ricevere i dati da Unity e salvarli in un file CSV
def receive_data_from_unity():
    global running
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server TCP in ascolto su {SERVER_IP}:{SERVER_PORT}...")

    conn, addr = server_socket.accept()
    print(f"Connessione accettata da: {addr}")

    with open("temp_realtime_data.csv", "w", newline="") as f:
        csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        while running:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                break
            data_decoded = data.decode("utf-8").strip()

            # Validazione dei dati ricevuti
            if validate_data_row(data_decoded):
                csv_writer.writerow(data_decoded.split(";"))  
            else:
                print(f"Riga malformata: {data_decoded}")  

    conn.close()
    server_socket.close()
    print("Connessione chiusa.")

# Funzione per validare che i dati abbiano il numero corretto di colonne (78 colonne)
def validate_data_row(data_row):
    columns = data_row.split(";")
    return len(columns) == 78  

# Avvia un thread per ricevere i dati da Unity
thread = threading.Thread(target=receive_data_from_unity)
thread.start()

# Attendi che i dati siano stati ricevuti
thread.join()

# Configurazione IP e porta del robot (modifica questi valori)
ROBOT_IP = "127.0.0.1"  # Sostituisci con l'IP del robot
ROBOT_PORT = 5000       # Sostituisci con la porta corretta

# Assicurarsi che la cartella di output esista
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

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
temp_csv_file = "temp_realtime_data.csv"
try:
    subprocess.run(["python", script_myAction, "--single_csv", temp_csv_file], check=True)

    common_pose_file = os.path.join(
        "./datasets/common_pose/testSingleAction",
        os.path.basename(temp_csv_file).replace(".csv", "_combined_skeleton.txt")
    )

    if not os.path.exists(common_pose_file):
        raise FileNotFoundError(f"Errore: il file {common_pose_file} non è stato creato.")
except Exception as e:
    print(f"Errore durante la conversione in common_pose: {e}", file=sys.stderr)
    sys.exit(1)

# Step 2: Calcolo delle coordinate
print("Calcolando le coordinate di inizio e fine azione...")
coordinates_file = os.path.join(output_dir, "coordinates_obj", "action_coordinates.txt")
try:
    subprocess.run(["python", script_det_coordinates, "--csv_file", temp_csv_file], check=True)
except Exception as e:
    print(f"Errore durante il calcolo delle coordinate: {e}", file=sys.stderr)
    sys.exit(1)

# Step 3: Creazione del file di annotazione per l'ultimo step
annotation_file = os.path.join(output_dir, "single_action_annotation.txt")
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
    action_output_file = os.path.join(output_dir, "recognized_action.txt")
    
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
