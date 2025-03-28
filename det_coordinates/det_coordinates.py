import numpy as np
import json
import pandas as pd
import argparse
import os

# # Caricamento dei dati dal file JSON
# with open("./det_coordinates/objects_coordinates/WorkArea.json", "r") as f:
#     data = json.load(f)

# # Definizione dei punti della scrivania
# A = np.array(data["A"]["coords"])
# B = np.array(data["B"]["coords"])
# C = np.array(data["C"]["coords"])

# # Calcolo dei vettori che definiscono il piano
# v1 = B - A
# v2 = C - A

# # Calcolo della normale al piano (prodotto vettoriale)
# normal = np.cross(v1, v2)
# normal = normal / np.linalg.norm(normal)  # Normalizzazione

# # Normale attesa per un piano perfettamente orizzontale (quindi la y punta verso l'alto)
# expected_normal = np.array([0, 1, 0])

# # Calcolo dell'angolo di inclinazione tra la normale calcolata e quella attesa
# dot_product = np.dot(normal, expected_normal)
# angle_radians = np.arccos(dot_product)  # Angolo in radianti
# angle_degrees = np.degrees(angle_radians)  # Conversione in gradi

# # Calcolo dell'asse di rotazione per riallineare il piano
# rotation_axis = np.cross(normal, expected_normal)
# rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalizziamo

# # Creiamo la matrice di rotazione usando la formula di Rodrigues
# K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
#               [rotation_axis[2], 0, -rotation_axis[0]],
#               [-rotation_axis[1], rotation_axis[0], 0]])

# I = np.identity(3)
# rotation_matrix = I + np.sin(angle_radians) * K + (1 - np.cos(angle_radians)) * np.dot(K, K)

# # Stampa della matrice di rotazione
# print("Matrice di Rotazione:")
# print(rotation_matrix)

# Numero di frame da mediare per l'inizio e la fine dell'azione
N = 2  

# Parser per accettare il file CSV come input
def parse_arguments():
    parser = argparse.ArgumentParser(description="Calcolo delle coordinate di inizio e fine azione")
    parser.add_argument('--csv_file', type=str, required=True, help="Percorso del file CSV della singola azione")
    return parser.parse_args()

# Percorso del file contenente gli oggetti
OBJECTS_FILE = "./det_coordinates/objects_coordinates/object_coordinates.json"

def load_objects(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # for obj in data.values():
    #     obj["coords"] = np.array(obj["coords"]) * 1000  # Converte la lista in np.array e scala da metri a millimetri
    #     obj["coords"] = np.array([-obj["coords"][0], obj["coords"][2], obj["coords"][1]])  # Conversione corretta SteamVR → Blender
    return data

def find_closest_object(hand_coords, objects):
    min_distance = float('inf')
    closest_object = None
    for obj_name, obj_data in objects.items():
        distance = np.linalg.norm(hand_coords - obj_data["coords"])
        print(f"Object: {obj_name}, Distance: {distance}, Hand Coords: {hand_coords}, Object Coords: {obj_data['coords']}")
        if distance < min_distance:
            min_distance = distance
            closest_object = obj_name
    return closest_object

def get_joint_coordinates(df, joint, N):
    if joint == "Wrist":
        start_coords = np.mean(df.iloc[:N][["Hand_X", "Hand_Y", "Hand_Z"]].values, axis=0)
        end_coords = np.mean(df.iloc[-N:][["Hand_X", "Hand_Y", "Hand_Z"]].values, axis=0)
    elif joint == "Index_TIP":
        start_coords = np.mean(df.iloc[:N][["Index_TIP_X", "Index_TIP_Y", "Index_TIP_Z"]].values, axis=0)
        end_coords = np.mean(df.iloc[-N:][["Index_TIP_X", "Index_TIP_Y", "Index_TIP_Z"]].values, axis=0)
    elif joint == "Palm_Center":
        palm_joints = ["Index_MCP_X", "Index_MCP_Y", "Index_MCP_Z", "Middle_MCP_X", "Middle_MCP_Y", "Middle_MCP_Z", 
                       "Ring_MCP_X", "Ring_MCP_Y", "Ring_MCP_Z", "Pinky_MCP_X", "Pinky_MCP_Y", "Pinky_MCP_Z"]
        start_coords = np.mean(df.iloc[:N][palm_joints].values.reshape(N, -1, 3), axis=(0, 1))
        end_coords = np.mean(df.iloc[-N:][palm_joints].values.reshape(N, -1, 3), axis=(0, 1))
    
    # Conversione delle coordinate da Blender a SteamVR e da millimetri a metri
    start_coords = np.array([-start_coords[0], start_coords[1], start_coords[2]]) / 1000
    end_coords = np.array([-end_coords[0], end_coords[1], end_coords[2]]) / 1000
    
    print(f"Start coordinates (converted): {start_coords}")
    print(f"End coordinates (converted): {end_coords}")
    
    return start_coords, end_coords

# Caricare gli oggetti
def main():
    args = parse_arguments()
    csv_file_path = args.csv_file
    objects = load_objects(OBJECTS_FILE)
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Il file CSV {csv_file_path} non esiste.")
    
    df = pd.read_csv(csv_file_path, sep=";", decimal=",")
    
    def convert_numeric(df):
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float, errors='ignore')
        return df

    df = convert_numeric(df)

    # Trova l'oggetto più vicino alle coordinate iniziali della mano
    start_hand_coords = np.mean(df.iloc[:N][["Hand_X", "Hand_Y", "Hand_Z"]].values, axis=0)
    start_hand_coords = np.array([-start_hand_coords[0], start_hand_coords[1], start_hand_coords[2]]) / 1000
    print(f"Start hand coordinates: {start_hand_coords}")
    closest_object = find_closest_object(start_hand_coords, objects)
    print(f"Closest object: {closest_object}")
    
    # Le coordinate di inizio sono quelle dell'oggetto
    #start_coords = objects[closest_object]["coords"]
    
    # Le coordinate di fine dipendono dal tipo di interazione
    interaction_type = objects[closest_object]["interaction"]
    start_coords, end_coords = get_joint_coordinates(df, interaction_type, N)
    
    # Applicare la matrice di rotazione alle coordinate finali
    # start_coords = np.dot(rotation_matrix, start_coords)
    # end_coords = np.dot(rotation_matrix, end_coords)
    
    # print(f"Start coordinates (rotated): {start_coords}")
    # print(f"End coordinates (rotated): {end_coords}")
    
    output_dir = "./results/coordinates_obj"
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_PATH = os.path.join(output_dir, "action_coordinates.txt")
    with open(OUTPUT_PATH, "w") as f:
        f.write("Inizio azione:\n")
        f.write(f"Hand X,Y,Z (corretto): {start_coords.tolist()}\n")
        
        f.write("Fine azione:\n")
        f.write(f"Hand X,Y,Z (corretto): {end_coords.tolist()}\n")
        
        f.write("Oggetto piu' vicino:\n")
        f.write(f"{closest_object}\n")
    
    print(f"Analisi completata. Risultati salvati in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
