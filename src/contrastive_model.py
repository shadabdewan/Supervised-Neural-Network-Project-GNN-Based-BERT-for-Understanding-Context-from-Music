from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertLyricsEncoder
from gnn_model import ChordGraphSAGE


class ContrastiveGNNBERT(nn.Module):
  """Cross-modal contrastive learning model aligning chord graph representations

  (GraphSAGE) with text caption representations (BERT) using a shared
  embedding space and symmetric InfoNCE loss.
  """

  def __init__(
      self,
      embedding_dim: int = 128,
      gnn_hidden_dim: int = 64,
      freeze_bert_layers: bool = True,
  ):
    super().__init__()

    # 1. Encoders
    self.gnn_encoder = ChordGraphSAGE(
      input_dim=12,
      hidden_dim=gnn_hidden_dim,
      output_dim=gnn_hidden_dim,
      num_layers=2,
    )
    self.bert_encoder = BertLyricsEncoder()

    # Freeze all BERT layers except the last 2 (same strategy as Task 3)
    if freeze_bert_layers:
      for param in self.bert_encoder.bert.parameters():
        param.requires_grad = False
      for layer in self.bert_encoder.bert.encoder.layer[-2:]:
        for param in layer.parameters():
          param.requires_grad = True

    # 2. Projection Heads (map modal features to shared embedding space)
    self.graph_projection = nn.Linear(gnn_hidden_dim, embedding_dim)
    self.text_projection = nn.Linear(768, embedding_dim)

  def encode_graph(self, graph_batch) -> torch.Tensor:
    """Extracts graph embedding via GraphSAGE, projects to shared dimension, and

    L2-normalizes.
    """
    # GraphSAGE forward pass (mean pooling produces graph-level representation g)
    g = self.gnn_encoder(
      graph_batch.x,
      graph_batch.edge_index,
      graph_batch.edge_weight,
      graph_batch.batch,
      return_embedding=True,
    )
    proj = self.graph_projection(g)
    # L2-normalization ensures cosine similarity equals dot product
    norm_g = F.normalize(proj, p=2, dim=-1)
    return norm_g

  def encode_text(
      self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> torch.Tensor:
    """Extracts text [CLS] embedding via BERT, projects to shared dimension, and

    L2-normalizes.
    """
    cls_embedding, _ = self.bert_encoder(input_ids, attention_mask)
    proj = self.text_projection(cls_embedding)
    # L2-normalization ensures cosine similarity equals dot product
    norm_t = F.normalize(proj, p=2, dim=-1)
    return norm_t

  def forward(
      self,
      graph_batch,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      temperature: float = 0.07,
  ):
    """Computes symmetric InfoNCE loss across graph and text modalities.

    ========================================================================
    INFONCE LOSS EXPLANATION (Symmetric Cross-Entropy):
    ========================================================================
    1. Cosine Similarities:
       Since graph_embeddings and text_embeddings are L2-normalized, their
       dot product (G @ T.T) computes pairwise cosine similarities across
       the batch. Scaling by 1 / temperature controls the sharpness of the
       softmax distribution.

    2. Positive Pairs:
       For batch index i, the true match is at diagonal position (i, i).
       Thus, targets are labels = arange(batch_size) = [0, 1, 2, ..., N-1].

    3. Bidirectional Optimization:
       - loss_g2t (Graph-to-Text): Treats graph_i as the query and asks:
         "Which text caption matches this graph?" (Rows are logit vectors).
       - loss_t2g (Text-to-Graph): Treats caption_i as the query and asks:
         "Which chord graph matches this text?" (Columns are logit vectors).

    The total loss is the average of both directions:
       Loss = (loss_g2t + loss_t2g) / 2
    ========================================================================
    """
    # Encode both modalities
    graph_embeds = self.encode_graph(graph_batch)  # [batch_size, embedding_dim]
    text_embeds = self.encode_text(
        input_ids, attention_mask
    )  # [batch_size, embedding_dim]

    batch_size = graph_embeds.size(0)
    device = graph_embeds.device

    # Scaled similarity matrix [batch_size, batch_size]
    similarity_matrix = (graph_embeds @ text_embeds.T) / temperature

    # Ground truth targets: entry (i, i) is the positive pair
    labels = torch.arange(batch_size, device=device)

    # Compute loss in both directions
    loss_g2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2g = F.cross_entropy(similarity_matrix.T, labels)

    # Symmetric InfoNCE loss
    loss = (loss_g2t + loss_t2g) / 2.0

    return loss, similarity_matrix