import socket
import subprocess
import threading
import sys
import time
import csv
import os

# Configurazione IP e porta del server TCP
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12345
BUFFER_SIZE = 1024

#testServer
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 5000

# robot e.Do
# ROBOT_IP = "192.168.1.58"
# ROBOT_PORT = 8585

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

# Flag per controllare il ciclo del server
running = True

# Global variable for robot socket
robot_socket = None

def validate_data_row(row):
    return len(row.split(";")) == 78

def handle_client(conn, addr):
    global running
    print(f"Connessione accettata da: {addr}")

    while running:
        data = conn.recv(BUFFER_SIZE).decode("utf-8").strip()

        if not data:
            print("Connessione chiusa dal client Unity.")
            break

        if data == "START":
            print("Inizio registrazione dati")
            csv_dir = os.path.normpath("./temp_data")
            os.makedirs(csv_dir, exist_ok=True)
            csv_filename = os.path.join(csv_dir, f"temp_realtime_data_{int(time.time())}.csv")
            with open(csv_filename, "w", newline="") as f:
                csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                while True:
                    data = conn.recv(BUFFER_SIZE)
                    if not data:
                        break
                    data_decoded = data.decode("utf-8").strip()
                    if data_decoded == "STOP":
                        print("Fine ricezione dati, avvio pipeline ML")
                        break
                    if validate_data_row(data_decoded):
                        csv_writer.writerow(data_decoded.split(";"))
                    else:
                        print(f"Riga malformata: {data_decoded}")

            # Step 1: Conversione dati
            print("Eseguendo la conversione in common_pose...")
            try:
                subprocess.run(["python", script_myAction, "--single_csv", csv_filename], check=True)
                common_pose_file = os.path.join(
                    "./datasets/common_pose/testSingleAction",
                    os.path.basename(csv_filename).replace(".csv", "_combined_skeleton.txt")
                )
                if not os.path.exists(common_pose_file):
                    raise FileNotFoundError(f"Errore: il file {common_pose_file} non è stato creato.")
            except Exception as e:
                print(f"Errore durante la conversione dati: {e}", file=sys.stderr)
                continue

            # Step 2: Calcolo delle coordinate
            print("Calcolando le coordinate di inizio e fine azione...")
            try:
                subprocess.run(["python", script_det_coordinates, "--csv_file", csv_filename], check=True)
                coordinates_file = os.path.join(output_dir, "coordinates_obj", "action_coordinates.txt")
            except Exception as e:
                print(f"Errore durante il calcolo delle coordinate: {e}", file=sys.stderr)
                continue

            # Step 3: Creazione annotazioni per classificazione
            annotation_file = os.path.join(output_dir, "single_action_annotation.txt")
            try:
                with open(annotation_file, "w") as f:
                    f.write(f"{common_pose_file} 9\n")
            except Exception as e:
                print(f"Errore durante la creazione del file di annotazione: {e}", file=sys.stderr)
                continue

            # Step 4: Classificazione azione
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
                continue

            # Step 5: Lettura dei dati e invio al robot
            print("Preparazione dei dati per l'invio al robot...")
            action_name = "UNKNOWN"
            recognized_action_file = os.path.join(output_dir, "recognized_action.txt")
            while not os.path.exists(recognized_action_file):
                time.sleep(0.1)
            with open(recognized_action_file, "r") as f:
                action_name = f.read().strip()

            with open(coordinates_file, "r") as f:
                lines = f.readlines()
                if len(lines) < 6:
                    raise ValueError("Errore: il file action_coordinates.txt non contiene abbastanza righe.")
                start_coords = lines[1].strip().split(": ")[1]
                end_coords = lines[3].strip().split(": ")[1]
                object_name = lines[5].strip()

            message = f"{action_name}|{object_name}|{start_coords}|{end_coords}|"
            print(f"Inviando dati al robot: {message}")

            # Invia messaggio al robot (thread per non chiudere la connessione)
            robot_thread = threading.Thread(target=send_to_robot, args=(message,conn))
            robot_thread.start()

            # # Invia il messaggio al client Unity
            # conn.sendall((message +"\n").encode('utf-8'))
            # # Invia una risposta di completamento al client Unity
            # time.sleep(0.1)
            # conn.sendall(b"Processing complete")
            # print("Pipeline completata. Pronta per la prossima azione.")
        else:
            print(f"Comando sconosciuto ricevuto: {data}")

    #conn.shutdown(socket.SHUT_WR)
    #conn.close()

def listen_for_keypress():
    global running
    while running:
        key = input()
        if key.lower() == 'k':
            print('Tasto "k" premuto, chiusura del server...')
            running = False
            break

def send_to_robot(message, conn):
    global robot_socket
    try:
        robot_socket.sendall(message.encode("utf-8"))
        response = robot_socket.recv(1024).decode('utf-8')
        print(f"Risposta robot: {response}")  # response -> messaggio ricevuto

        #FORSE HA SENSO CHE MANDO QUI MESSAGGIO A UNITY ?
        # Invia il messaggio al client Unity
        conn.sendall((message +"\n").encode('utf-8'))
        # Invia una risposta di completamento al client Unity
        time.sleep(0.1)
        conn.sendall(b"Processing complete")
        print("Pipeline completata. Pronta per la prossima azione.")

    except Exception as e:
        print(f"Errore durante l'invio dei dati al robot: {e}", file=sys.stderr)

def handle_robot_communication():
    global robot_socket
    robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robot_socket.connect((ROBOT_IP, ROBOT_PORT))
    print("Connessione con il robot stabilita.")

def main_server():
    global running
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(5) #5 è la lunghezza massima della coda di connesioni in attesa
    print(f"Server TCP in ascolto su {SERVER_IP}:{SERVER_PORT}...")

    # Avvia un thread per ascoltare l'input della tastiera
    thread = threading.Thread(target=listen_for_keypress)
    thread.start()

    # Avvia un thread per la comunicazione con il robot
    robot_thread = threading.Thread(target=handle_robot_communication)
    robot_thread.start()

    while running:
        try:
            server_socket.settimeout(1.0)
            conn, addr = server_socket.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
        except socket.timeout:
            continue
        except socket.error:
            if not running:
                break

    server_socket.close()
    robot_socket.close()
    print("Server chiuso.")

if __name__ == '__main__':
    main_server()
