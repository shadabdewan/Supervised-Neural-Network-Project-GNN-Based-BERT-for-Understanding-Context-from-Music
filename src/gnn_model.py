import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class ChordGraphSAGE(nn.Module):
    """
    GraphSAGE model for music genre classification from chord-transition graphs.
    
    Architecture:
    - Stacked SAGEConv layers for neighborhood aggregation.
    - Global Mean Pooling to aggregate node representations into a single graph embedding.
    - Linear classification head mapping graph embeddings to genre logits.
    """
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(ChordGraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # First SAGEConv layer: input_dim -> hidden_dim
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        # Middle SAGEConv layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Last SAGEConv layer: hidden_dim -> hidden_dim (before global pooling)
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Final Linear readout layer mapping graph embedding to 8 genre classes
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_embedding=False):
        """
        Forward pass.

        Returns either the graph embedding or logits depending on `return_embedding`.
        """
        # Note: PyTorch Geometric's standard SAGEConv does not utilize edge_weight by
        # default in mean/max/gcn aggregation.
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embed = global_mean_pool(x, batch)

        if return_embedding:
            return graph_embed

        logits = self.classifier(graph_embed)
        return logits