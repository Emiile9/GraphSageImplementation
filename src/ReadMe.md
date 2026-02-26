# Utilisation de GraphSAGE en Mini-Batch

Ce guide décrit les étapes nécessaires pour charger un graphe et entraîner un modèle **GraphSAGE** en mini-batch à l’aide d’un échantillonnage de voisins.

L’objectif est d’apprendre des représentations vectorielles de nœuds en agrégeant l’information de leurs voisins à chaque couche du modèle.

---

## 1. Chargement du graphe

Le graphe est chargé depuis un fichier `.pt`, généralement un objet NetworkX sérialisé avec PyTorch.

```python
import torch

# Chargement du graphe depuis le fichier source
G = torch.load(
    "src/generation_dataset/<timestamp>/graphs/ppi.pt",
    weights_only=False
)
```

Chaque nœud doit contenir un attribut `"x"` correspondant à son vecteur de features :

```python
G.nodes[node]["x"] = torch.tensor([...])
```

Le script convertit automatiquement cet attribut en `"features"` pour compatibilité avec l’algorithme mini-batch :

```python
for node in G.nodes():
    G.nodes[node]["features"] = G.nodes[node]["x"]
```

---

## 2. Configuration du modèle GraphSAGE

Le modèle est basé sur une architecture à 2 couches.

Chaque couche applique :

- Un échantillonnage de voisins
- Une agrégation (Mean, MaxPooling ou LSTM)
- Une transformation linéaire suivie d’une activation ReLU

Configuration interne :

```python
hidden_dim = feat_dim
depth = 2

W = nn.ModuleList([
    nn.Linear(2 * hidden_dim, hidden_dim)
    for _ in range(depth)
])
```

---

## 3. Agrégateurs disponibles

```text
mean   -> Mean Aggregator
max    -> Max Pooling Aggregator
lstm   -> LSTM Aggregator
```

---

## 4. Configuration de l’échantillonnage

À chaque couche, un nombre fixe de voisins est échantillonné :

```python
sample_size = [args.sample_size for _ in range(depth)]
```

Cela permet de :

- Réduire le coût mémoire
- Rendre l'entraînement scalable sur grands graphes

---

## 5. Lancement depuis le terminal

Le script principal se lance avec :

```bash
python main.py \
--graph_path <chemin_vers_graphe.pt> \
--aggregator <mean|max|lstm> \
--sample_size <int> \
--epochs <int> \
--batch_size <int>
```

```text
--graph_path   Chemin vers le fichier .pt du graphe
--aggregator   Type d’agrégateur : mean | max | lstm
--sample_size  Nombre de voisins échantillonnés par couche
--epochs       Nombre d’époques d’entraînement
--batch_size   Taille des batches
```

---

## 6. Exemple complet

```bash
python main.py \
--graph_path src/generation_dataset/2026-02-10_14-21-11/graphs/ppi.pt \
--aggregator mean \
--sample_size 5 \
--epochs 10 \
--batch_size 128
```

---

## 7. Entraînement

L’entraînement est lancé via :

```python
train(
    model,
    G,
    device,
    sampling_size=sample_size,
    epochs=args.epochs,
    learning_rate=3e-4,
    batch_size=args.batch_size
)
```

Le paramètre `batch_size` contrôle le nombre de nœuds traités par itération.

---

## 8. Remarques importantes

- La dimension cachée est égale à la dimension des features du graphe.
- La profondeur du modèle est fixée à 2 couches.
- Le learning rate est fixé à `3e-4` (modifiable dans le code).
- Tous les nœuds doivent contenir un attribut `"x"`.