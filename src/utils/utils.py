"""
Fichier utilitaire pour les fonctions diverses
"""
import numpy as np

def get_n_neighbours(graph, node, n):
    """
    Retourne n voisins aléatoires d'un noeud donné dans un graphe représenté par une liste d'adjacence.

    Parameters:
    graph (dict): Dictionnaire représentant la liste d'adjacence du graphe.
    node: Le noeud dont on veut obtenir les voisins.
    n (int): Le nombre de voisins à retourner.

    Returns:
    list: Une liste des n voisins aléatoires du noeud spécifié.
    """
    if node not in graph:
        return []

    neighbors = graph[node]
    if len(neighbors) <= n:
        return neighbors
    return list(np.random.choice(neighbors, n, replace=False))

