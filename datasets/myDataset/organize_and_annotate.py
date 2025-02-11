import os
import shutil
import argparse

# Configurazione delle etichette numeriche per le azioni
ACTION_LABELS = {
    "push": 0,
    "pull": 1,
    "pickplace": 2,
    "upsideDown": 3,
    "plug": 4,
    "unplug": 5,
    "screw": 6,
    "unscrew": 7
}

def organize_files(raw_data_path, organized_data_path):
    """
    Organizza i file CSV non strutturati in una struttura gerarchica.
    """
    print(f"Organizzando i file da: {raw_data_path} a: {organized_data_path}")
    if not os.path.exists(organized_data_path):
        os.makedirs(organized_data_path)
    
    moved_files = []
    
    # Scansiona tutti i file nella cartella non strutturata
    for file_name in os.listdir(raw_data_path):
        if file_name.endswith(".csv"):
            # Estrai informazioni dal nome del file
            # Nome file esempio: pickplace_dx_Luca001_30-01_LucaTracker_R.csv
            parts = file_name.replace(".csv", "").split("_")
            if len(parts) < 4:
                print(f"Nome file non valido: {file_name}. Saltato.")
                continue
            
            action, hand, name, date_profile = parts[0], parts[1], parts[2], parts[3]
            
            # Verifica che l'azione sia valida
            if action not in ACTION_LABELS:
                print(f"Azione non valida: {action}. Saltata.")
                continue
            
            # Percorso destinazione
            hand_folder = "dx" if hand == "dx" else "sx"
            action_dir = os.path.join(organized_data_path, action) #prima cartella datasetPath/action/
            hand_dir = os.path.join(action_dir, hand_folder) #sottocartella datasetPath/action/hand/
            os.makedirs(hand_dir, exist_ok=True)
            
            # Sposta il file nella nuova struttura
            src_file = os.path.join(raw_data_path, file_name)
            dst_file = os.path.join(hand_dir, file_name)
            shutil.move(src_file, dst_file)
            moved_files.append((action, hand_folder, file_name))
            print(f"File spostato: {src_file} -> {dst_file}")
    
    print("Organizzazione completata.\n")
    return moved_files

def generate_annotation_file(organized_data_path, annotation_file_path, moved_files):
    """
    Genera il file annotation_file.txt con percorsi e label.
    """
    print(f"Generando il file di annotazione: {annotation_file_path}")
    
    existing_annotations = set()
    if os.path.exists(annotation_file_path):
        with open(annotation_file_path, "r") as annotation_file:
            for line in annotation_file:
                existing_annotations.add(line.strip())
    
    with open(annotation_file_path, "a") as annotation_file:
        for action, hand_folder, file_name in moved_files:
            label = ACTION_LABELS.get(action, None)
            if label is None:
                print(f"Azione non trovata nel dizionario: {action}. Saltata.")
                continue
            
            relative_path = os.path.join("MANUS_data", action, hand_folder, file_name).replace('\\', '/')
            annotation_entry = f"{relative_path} {label}"
            if annotation_entry not in existing_annotations:
                annotation_file.write(f"{annotation_entry}\n")
                print(f"Aggiunto al file di annotazione: {annotation_entry}")
    
    print(f"File di annotazione generato: {annotation_file_path}\n")

def main():
    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Organizza file CSV e genera il file annotation_file.txt.")
    parser.add_argument("--raw_data", type=str, required=True, help="Percorso della cartella con i file CSV non organizzati.")
    parser.add_argument("--organized_data", type=str, required=True, help="Percorso della cartella per la struttura organizzata.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Percorso del file di annotazione da generare.")
    args = parser.parse_args()

    # Organizza i file CSV nella struttura gerarchica
    moved_files = organize_files(args.raw_data, args.organized_data)
    
    # Genera il file annotation_file.txt
    generate_annotation_file(args.organized_data, args.annotation_file, moved_files)

if __name__ == "__main__":
    main()
