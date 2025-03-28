import socket
import threading
import sys

# Flag per controllare il ciclo del server
running = True

def listen_for_keypress():
    global running
    while running:
        key = input()
        if key.lower() == 'k':
            print('Tasto "k" premuto, chiusura del server...')
            running = False
            server_socket.close()
            break

# Creazione del server TCP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 5000))  #la porta uguale a quella di `run_pipeline.py`
server_socket.listen(1)

# Avvia un thread per ascoltare l'input della tastiera
thread = threading.Thread(target=listen_for_keypress)
thread.start()

print("Server in attesa di connessioni...")

while running:
    try:
        conn, addr = server_socket.accept()
        print(f"Connessione ricevuta da {addr}")
        
        while running:
            try:
                # Ricezione del messaggio
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break
                print("Dati ricevuti:", data)
                
                # Invia una risposta di conferma al client
                conn.sendall(b"ACK")  # Il robot risponderebbe con un ACK simile
            except socket.error:
                break
        conn.close()
    except socket.error:
        if not running:
            break

print("Server chiuso.")
