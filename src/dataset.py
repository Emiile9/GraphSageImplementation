import random
import torch
from torch.utils.data import Dataset


class GraphSageDataset(Dataset):
    def __init__(
        self,
        G,
        num_pairs=10000,
        walk_length=5,
        context_size=2,
        num_neg=20,
    ):
        self.G = G

        # --- CHANGEMENT ICI ---
        # Re-mapping des noeuds vers des indices continus 0..N-1
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        self.adj_list = {
            self.node_to_idx[node]: [
                self.node_to_idx[neigh] for neigh in G.neighbors(node)
            ]
            for node in self.nodes
        }
        # --- FIN DU CHANGEMENT ---

        self.num_nodes = len(self.nodes)
        self.num_pairs = num_pairs
        self.walk_length = walk_length
        self.context_size = context_size
        self.num_neg = num_neg

        # Distribution négative degree^(3/4)
        degrees = torch.tensor(
            [len(self.adj_list[i]) for i in range(self.num_nodes)],
            dtype=torch.float
        )
        prob_neg = degrees.pow(0.75)
        prob_neg /= prob_neg.sum()
        self.prob_neg = prob_neg

        # Pré-génération des paires positives
        self.u_nodes, self.pos_nodes = self._generate_positive_pairs()

    def _generate_positive_pairs(self):
        u_nodes, pos_nodes = [], []

        while len(u_nodes) < self.num_pairs:
            start = random.randrange(self.num_nodes)
            walk = [start]

            for _ in range(self.walk_length - 1):
                cur = walk[-1]
                if len(self.adj_list[cur]) == 0:
                    break
                walk.append(random.choice(self.adj_list[cur]))

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

        neg = torch.multinomial(
            self.prob_neg,
            self.num_neg,
            replacement=True,
        )

        return u, pos, neg
