"""
Fichier utilitaire pour les fonctions diverses
"""

import numpy as np
import networkx as nx


def get_n_neighbours(graph: nx.Graph, node, n: int):
    """
    Retourne n voisins aléatoires d'un noeud donné dans un graphe représenté par une liste d'adjacence.

    Parameters:
    graph (nx.Graph): Dictionnaire représentant la liste d'adjacence du graphe.
    node: Le noeud dont on veut obtenir les voisins.
    n (int): Le nombre de voisins à retourner.

    Returns:
    list: Une liste des n voisins aléatoires du noeud spécifié.
    """
    if not graph.has_node(node):
        return []

    neighbors = list(graph.neighbors(node))
    if len(neighbors) <= n:
        return neighbors
    rng = np.random.default_rng()
    return list(rng.choice(neighbors, n, replace=False))
