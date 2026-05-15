import random
import torch
from torch.utils.data import Dataset


class GraphSageDataset(Dataset):
    """
    Transformation d'un graphe networkx en données exploitables par PyTorch

    --- Paramètres ---
    G : Graphe networkx
    num_pairs : nombre de paires générées à partir du graphe (chaque paire forme un individu de notre dataset)
    walk_length : longueur des random walks
    context_size : distance entre deux noeuds d'une même random walk pour être considérés comme proches (paire positive)
    num_neg : nombre de noeuds non-proches utilisés pour la fonction de perte
    """
    
    def __init__(
        self,
        G,
        num_pairs=10000,
        walk_length=5,
        context_size=2,
        num_neg=20,
    ):
        self.G = G

        self.nodes = list(G.nodes())
        self.num_nodes = len(self.nodes)

        # Adjacence avec les vrais node IDs du graphe
        self.adj_list = {
            node: list(G.neighbors(node))
            for node in self.nodes
        }

        self.num_pairs = num_pairs
        self.walk_length = walk_length
        self.context_size = context_size
        self.num_neg = num_neg

        # Distribution négative degree^(3/4) indexée par position dans self.nodes
        degrees = torch.tensor(
            [len(self.adj_list[node]) for node in self.nodes],
            dtype=torch.float
        )
        prob_neg = degrees.pow(0.75)
        prob_neg /= prob_neg.sum()
        self.prob_neg = prob_neg

        # Pré-génération des paires positives (stockées comme vrais node IDs)
        self.u_nodes, self.pos_nodes = self._generate_positive_pairs()

    def _generate_positive_pairs(self):
        # Random-walks sur les vrais node IDs
        u_nodes, pos_nodes = [], []

        while len(u_nodes) < self.num_pairs:
            start = random.choice(self.nodes)
            walk = [start]

            for _ in range(self.walk_length - 1):
                cur = walk[-1]
                neighbors = self.adj_list[cur]
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))

            for i, u in enumerate(walk):
                for j in range(
                    max(0, i - self.context_size),
                    min(len(walk), i + self.context_size + 1),
                ):
                    if i == j:
                        continue

                    u_nodes.append(u)
                    pos_nodes.append(walk[j])

                    if len(u_nodes) >= self.num_pairs:
                        break
                if len(u_nodes) >= self.num_pairs:
                    break

        return torch.tensor(u_nodes), torch.tensor(pos_nodes)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        u = self.u_nodes[idx]
        pos = self.pos_nodes[idx]

        # multinomial renvoie des positions → convertir en vrais node IDs
        neg_positions = torch.multinomial(self.prob_neg, self.num_neg, replacement=True)
        nodes_tensor = torch.tensor(self.nodes)
        neg = nodes_tensor[neg_positions]

        return u, pos, neg
