
# Utilisation de GraphSAGE pour l’extraction de caractéristiques

Ce guide décrit les étapes nécessaires pour charger un graphe et préparer les données afin d’entraîner un modèle **GraphSAGE** dans un cadre non supervisé, à l’aide d’un échantillonnage par marches aléatoires (Random Walks).

L’objectif est de générer des paires de nœuds structurellement proches, utilisées pour apprendre des représentations vectorielles de nœuds.

---

## 1. Chargement du graphe

Le graphe est chargé depuis un fichier `.pt`, généralement un objet PyTorch ou NetworkX sérialisé.

```python
import torch

# Chargement du graphe depuis le fichier source
G = torch.load(
    "src/generation_dataset/<timestamp>/graphs/ca-GrQc_graph.pt"
)
````

---

## 2. Création du dataset GraphSAGE

La classe `GraphSageDataset` génère les paires de nœuds nécessaires à l’apprentissage non supervisé.
Elle s’appuie sur des marches aléatoires pour identifier des nœuds proches du point de vue structurel, et applique un échantillonnage négatif.

```python
from dataset import GraphSageDataset

dataset = GraphSageDataset(
    G=G,
    num_pairs=10000,  # Nombre total de paires (u, v) générées
    walk_length=5,    # Longueur des marches aléatoires
    context_size=2,   # Taille de la fenêtre de contexte
    num_neg=20        # Nombre d’exemples négatifs par paire positive
)
```

---

## 3. Configuration du DataLoader

Le `DataLoader` permet de charger les données par lots (batches) et d’itérer efficacement lors de l’entraînement du modèle.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True
)
```

Ce `DataLoader` peut ensuite être utilisé directement dans la boucle d’entraînement du modèle GraphSAGE.


