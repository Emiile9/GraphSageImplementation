# ============================================================
# generate_datasets.py
# Génération de graphes réels pour GraphSAGE
# ============================================================

import os
import sys
import urllib.request
import gzip
import shutil
from datetime import datetime

import torch
import networkx as nx


# ------------------------------------------------------------
# 0. Imports projet (robustes)
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = CURRENT_DIR

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import GraphSageDataset


# ------------------------------------------------------------
# 1. Dossier de génération horodaté
# ------------------------------------------------------------
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_DIR = os.path.join(SRC_DIR, "generation_dataset", TIMESTAMP)
RAW_DIR = os.path.join(BASE_DIR, "raw")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

print(f"Génération dans : {BASE_DIR}")


# ------------------------------------------------------------
# 2. Datasets réels (SNAP et équivalents GraphSAGE)
# ------------------------------------------------------------
DATASETS = {
    "ca-GrQc": {
        "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "description": "Collaboration scientifique (arXiv)"
    },
    "ca-HepTh": {
        "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "description": "Citations scientifiques (High Energy Physics)"
    },
    "facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "description": "Réseau social Facebook"
    },
    "reddit": {
        "url": "https://snap.stanford.edu/data/soc-RedditHyperlinks-title.tsv",
        "description": "Réseau d’interactions Reddit (hyperliens entre posts)"
    },
    "ppi": {
        "url": "https://snap.stanford.edu/data/ppi-human.txt.gz",
        "description": "Réseau d’interactions protéine-protéine (PPI)"
    },
}


# ------------------------------------------------------------
# 3. Edge list -> NetworkX Graph (robuste)
# ------------------------------------------------------------
def load_graph_from_edgelist(path: str) -> nx.Graph:
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G


# ------------------------------------------------------------
# 4. Pipeline principal
# ------------------------------------------------------------
for name, info in DATASETS.items():
    print("=" * 70)
    print(f"Dataset : {name}")
    print(f"Description : {info['description']}")

    gz_path = os.path.join(RAW_DIR, f"{name}.gz")
    txt_path = os.path.join(RAW_DIR, f"{name}.txt")
    graph_path = os.path.join(GRAPH_DIR, f"{name}_graph.pt")

    # Téléchargement
    if not os.path.exists(txt_path):
        print("Téléchargement...")
        if info["url"].endswith(".gz"):
            urllib.request.urlretrieve(info["url"], gz_path)
            with gzip.open(gz_path, "rb") as f_in:
                with open(txt_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            urllib.request.urlretrieve(info["url"], txt_path)

    # Construction ou chargement du graphe
    if os.path.exists(graph_path):
        print("Chargement du graphe existant")
        G = torch.load(graph_path)
    else:
        print("Construction du graphe NetworkX")
        G = load_graph_from_edgelist(txt_path)
        torch.save(G, graph_path)
        print(f"Graphe sauvegardé : {graph_path}")

    print(f"Nombre de noeuds : {G.number_of_nodes()}")
    print(f"Nombre d'arêtes : {G.number_of_edges()}")

    # Test avec la version finale de GraphSageDataset
    dataset = GraphSageDataset(
        G=G,
        num_pairs=1000,
        walk_length=5,
        context_size=2,
        num_neg=10
    )

    u, v_pos, v_neg = dataset[0]
    print("Test GraphSageDataset OK :", u.item(), v_pos.item(), v_neg.shape)

print("=" * 70)
print("Tous les datasets ont été générés et validés correctement.")
