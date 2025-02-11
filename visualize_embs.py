import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import plotly.express as px

# Configurazione
REFERENCE_CSV = "./results/embeddings/reference_embeddings.csv"  # Percorso del file CSV con embeddings di riferimento
MY_CSV = "./results/embeddings/my_embeddings.csv"  # Percorso del file CSV con i tuoi embeddings
LABELS_FILE = "./datasets/myDataset/data_action_recognition.txt"  # Percorso del file con la corrispondenza delle etichette
N_COMPONENTS = 3  # Numero di dimensioni da ridurre
PERPLEXITY = 30  # Parametro di t-SNE
RANDOM_SEED = 42  # Fissiamo il seed per riproducibilità

# Caricamento della corrispondenza delle etichette
label_mapping = {}
with open(LABELS_FILE, 'r') as f:
    for line in f:
        name, number = line.strip().split()
        label_mapping[int(number)] = name

# Caricamento degli embeddings di riferimento
df_ref = pd.read_csv(REFERENCE_CSV)
embeddings_ref = df_ref.iloc[:, :-1].values         
labels_ref = df_ref.iloc[:, -1].values

# Caricamento dei tuoi embeddings
df_my = pd.read_csv(MY_CSV)
embeddings_my = df_my.iloc[:, :-1].values  
labels_my = df_my.iloc[:, -1].values

# Combina gli embeddings e le etichette
embeddings = np.vstack((embeddings_ref, embeddings_my))
labels = np.concatenate((labels_ref, labels_my))

# Mappa i numeri delle etichette ai loro nomi
labels_named = [label_mapping[label] for label in labels]

# Standardizzazione degli embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# PCA
pca = PCA(n_components=N_COMPONENTS)
embeddings_pca = pca.fit_transform(embeddings_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=RANDOM_SEED)
embeddings_tsne = tsne.fit_transform(embeddings_scaled)

# UMAP 
umap_reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
embeddings_umap = umap_reducer.fit_transform(embeddings_scaled)

# Funzione per la visualizzazione 2D con Matplotlib
# def plot_embeddings_2d(embeddings, title):
#     df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2"])
#     df_vis["Label"] = labels_named

#     fig = plt.figure(figsize=(12, 8))
#     gs = GridSpec(1, 2, width_ratios=[4, 1])

#     ax0 = fig.add_subplot(gs[0])
#     scatter_plot = sns.scatterplot(data=df_vis, x="Dim1", y="Dim2", hue="Label", palette="Set1", alpha=0.7, ax=ax0)
#     ax0.set_title(title)
#     ax0.legend().remove() 

#     ax1 = fig.add_subplot(gs[1])
#     handles, legend_labels = scatter_plot.get_legend_handles_labels()
#     if len(handles) > 0 and len(legend_labels) > 0:
#         ax1.legend(handles=handles, labels=df_vis["Label"].unique().tolist(), loc='center', ncol=1)
#     ax1.axis('off')  # Rimuovi gli assi dalla figura della legenda

#     plt.tight_layout()

def plot_embeddings_2d(embeddings, title):
    df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2"])
    df_vis["Label"] = labels_named

    fig = px.scatter(df_vis, x="Dim1", y="Dim2", color="Label", title=title, labels={"Label": "Label"})
    fig.update_layout(legend=dict(title="Label"))
    fig.show()

# Funzione per la visualizzazione 3D
def plot_embeddings_3d(embeddings, title):
    df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2", "Dim3"])
    df_vis["Label"] = labels_named

    fig = px.scatter_3d(df_vis, x="Dim1", y="Dim2", z="Dim3", color="Label", title=title, labels={"Label": "Label"})
    fig.update_layout(legend=dict(title="Label"))
    fig.show()

# Visualizzazione
#plot_embeddings_3d(embeddings_pca, "PCA - Riduzione a 3D")
plot_embeddings_2d(embeddings_tsne, "t-SNE - Riduzione a 2D")
plot_embeddings_3d(embeddings_umap, "UMAP - Riduzione a 3D")
plt.show()
