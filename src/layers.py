import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, neighbors_features):
        """
        Args:
            neighbors_features (torch.Tensor): Tensor de forme (batch_size, num_neighbors, feature_dim)

        Returns:
            torch.Tensor: Tensor de forme (batch_size, feature_dim)
        """
        return torch.mean(neighbors_features, dim=1)