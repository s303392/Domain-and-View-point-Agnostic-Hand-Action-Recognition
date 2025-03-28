import openvr
import json
import os

# Inizializza OpenVR
vr_system = openvr.init(openvr.VRApplication_Other)

# Percorso del file JSON per salvare le coordinate
output_file = "./det_coordinates/objects_coordinates/object_coordinates.json"

# Caricare dati esistenti se il file esiste
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        object_positions = json.load(f)
else:
    object_positions = {}


def get_tracker_positions():
    "Ottiene una singola lettura delle coordinate dei Vive Tracker attivi."
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )
    tracker_positions = {}

    for device_index, pose in enumerate(poses):
        if pose.bDeviceIsConnected:
            device_class = vr_system.getTrackedDeviceClass(device_index)
            if device_class == openvr.TrackedDeviceClass_GenericTracker:
                matrix = pose.mDeviceToAbsoluteTracking
                x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]  # Estrarre coordinate X, Y, Z
                tracker_positions[device_index] = [x, z, y]

    return tracker_positions


# Ottenere una lettura unica delle coordinate
print("Scansione dei tracker in corso...")
positions = get_tracker_positions()

# Definire le interazioni possibili
interaction_types = ["Wrist", "Index_TIP", "Palm_Center"]

# Associare un nome e un'interazione agli oggetti
print("\nAssegna un nome e un tipo di interazione agli oggetti tracciati:")
for tracker_id, coords in positions.items():
    object_name = input(f"Inserisci il nome per il tracker {tracker_id} (X: {coords[0]}, Y: {coords[1]}, Z: {coords[2]}): ")
    if object_name in object_positions:
        print(f"L'oggetto {object_name} esiste già, aggiornando le coordinate.")
    interaction = input(f"Seleziona il tipo di interazione per {object_name} (Wrist, Index_TIP, Palm_Center): ")
    while interaction not in interaction_types:
        print("Scelta non valida. Scegli tra: Wrist, Index_TIP, Palm_Center.")
        interaction = input(f"Seleziona il tipo di interazione per {object_name}: ")
    
    object_positions[object_name] = {"coords": coords, "interaction": interaction}

# Salvare i dati nel file JSON
with open(output_file, "w") as f:
    json.dump(object_positions, f, indent=4)

print(f"\nMappatura completata! Le coordinate sono state salvate in {output_file}")

# Chiudere OpenVR
openvr.shutdown()
