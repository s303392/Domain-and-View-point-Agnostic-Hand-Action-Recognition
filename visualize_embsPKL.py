import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import pickle
import plotly.express as px
import os


#TODO: HO AGGIUNTO IL SALVATAGGIO DI UN PKL PER OGNI SINGOLA AZIONE, MODIFICARE IL CODICE PER VEDERLA(ovviamente sarà l'ultima registrata)

# Configurazione
EMBEDDINGS_PKL = "./results/reference_embs.pkl"  # Percorso del file PKL con i vettori
LABELS_TXT = "./dataset_scripts/myDataset/ref_ComPose_annotations.txt"  # Percorso del file TXT con le etichette
N_COMPONENTS = 3  # Numero di dimensioni da ridurre
PERPLEXITY = 30  # Parametro di t-SNE
RANDOM_SEED = 42  # Fissiamo il seed per riproducibilità

# Caricamento degli embeddings dal file .pkl
with open(EMBEDDINGS_PKL, 'rb') as f:
    reference_embs, _ = pickle.load(f)  # Carica solo i reference embeddings

# Verifica il numero di embeddings caricati
print(f"Numero di reference embeddings caricati: {len(reference_embs)}")

# Caricamento delle etichette dal file .txt
labels = []
with open(LABELS_TXT, 'r') as f:
    for line in f:
        # Ignora righe vuote o righe non valide
        if line.strip():
            try:
                # Supponiamo che l'etichetta sia l'ultimo elemento della riga
                label = line.strip().split()[-1]
                labels.append(label)
            except IndexError:
                print(f"Riga non valida nel file delle etichette: {line.strip()}")

# Verifica che il numero di embeddings corrisponda al numero di etichette
if len(reference_embs) != len(labels):
    print(f"Numero di embeddings: {len(reference_embs)}")
    print(f"Numero di etichette: {len(labels)}")
    raise ValueError("Il numero di embeddings nel file .pkl non corrisponde al numero di etichette nel file .txt.")

# Caricamento del mapping numeri -> nomi dal file delle annotazioni
label_mapping = {}
with open(LABELS_TXT, 'r') as f:
    for line in f:
        # Supponiamo che il formato sia: "path/to/file numero"
        parts = line.strip().split()
        if len(parts) >= 2:
            label_number = parts[-1]  # L'ultimo elemento è il numero
            label_name = os.path.basename(parts[0]).split('_')[0]  # Estrai il nome dal percorso
            label_mapping[label_number] = label_name

# Converti le etichette numeriche in nomi
labels_named = [label_mapping[label] for label in labels]

# Verifica che il mapping sia corretto
print(f"Mapping delle etichette: {label_mapping}")

# Standardizzazione degli embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(reference_embs)

# PCA
pca = PCA(n_components=N_COMPONENTS)
embeddings_pca = pca.fit_transform(embeddings_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=RANDOM_SEED)
embeddings_tsne = tsne.fit_transform(embeddings_scaled)

# UMAP
umap_reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
embeddings_umap = umap_reducer.fit_transform(embeddings_scaled)

# Funzione per la visualizzazione 2D con Plotly
def plot_embeddings_2d(embeddings, labels, title):
    df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2"])
    df_vis["Label"] = labels

    fig = px.scatter(df_vis, x="Dim1", y="Dim2", color="Label", title=title, labels={"Label": "Label"})
    fig.update_layout(legend=dict(title="Label"))
    fig.show()

# Funzione per la visualizzazione 3D con Plotly
def plot_embeddings_3d(embeddings, labels, title):
    df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2", "Dim3"])
    df_vis["Label"] = labels

    fig = px.scatter_3d(df_vis, x="Dim1", y="Dim2", z="Dim3", color="Label", title=title, labels={"Label": "Label"})
    fig.update_layout(legend=dict(title="Label"))
    fig.show()

# Funzione per creare una figura con la legenda separata
def plot_legend(labels_named, title="Legenda"):
    unique_labels = sorted(set(labels_named))  # Trova le etichette uniche
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Genera una palette di colori

    # Crea una figura vuota
    fig, ax = plt.subplots(figsize=(4, len(unique_labels) * 0.5))  # Altezza proporzionale al numero di etichette
    ax.axis('off')  # Rimuovi gli assi

    # Aggiungi la legenda
    legend_handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10) 
                      for color in colors]
    ax.legend(legend_handles, unique_labels, title=title, loc='center', frameon=False)

    # Salva la figura
    plt.tight_layout()
    plt.savefig("legend.png")  # Salva la legenda come immagine
    plt.show()

# Visualizzazione
plot_embeddings_2d(embeddings_tsne, labels_named, "t-SNE - Riduzione a 2D")
plot_embeddings_3d(embeddings_umap, labels_named, "UMAP - Riduzione a 3D")

plot_legend(labels_named, title="Legenda delle Etichette")
plt.show()