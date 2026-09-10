import math
from bert_encoder import BertLyricsEncoder
from gnn_model import ChordGraphSAGE
import torch
import torch.nn as nn


def freeze_bert_except_last_layers(bert_model, num_layers_trainable=2):
  """Freezes all BERT parameters except for the last `num_layers_trainable` layers."""
  # Freeze base embeddings and all encoders first
  for param in bert_model.parameters():
    param.requires_grad = False

  # Unfreeze the last N layers of the transformer encoder
  if hasattr(bert_model, "bert") and hasattr(bert_model.bert, "encoder"):
    for layer in bert_model.bert.encoder.layer[-num_layers_trainable:]:
      for param in layer.parameters():
        param.requires_grad = True


class GNNOnlyModel(nn.Module):
  """Baseline 1: Uses GraphSAGE audio chord representations only."""

  def __init__(
      self,
      hidden_dim: int = 64,
      num_classes: int = 8,
      freeze_bert_layers: bool = True,
  ):
    super(GNNOnlyModel, self).__init__()
    self.gnn = ChordGraphSAGE(
        input_dim=12, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2
    )
    self.classifier = nn.Linear(hidden_dim, num_classes)

  def forward(self, batch):
    g = self.gnn(
        batch.x, batch.edge_index, batch.edge_weight, batch.batch, return_embedding=True
    )
    logits = self.classifier(g)
    return logits


class BERTOnlyModel(nn.Module):
  """Baseline 2: Uses BERT transcriptions [CLS] representations only."""

  def __init__(
      self,
      bert_model_name: str = "bert-base-uncased",
      num_classes: int = 8,
      freeze_bert_layers: bool = True,
  ):
    super(BERTOnlyModel, self).__init__()
    self.bert_encoder = BertLyricsEncoder(
        model_name=bert_model_name,
        freeze_bert=False,
    )
    if freeze_bert_layers:
      freeze_bert_except_last_layers(
          self.bert_encoder, num_layers_trainable=2
      )

    self.classifier = nn.Linear(768, num_classes)

  def forward(self, input_ids, attention_mask):
    cls_emb, _ = self.bert_encoder(input_ids, attention_mask)
    logits = self.classifier(cls_emb)
    return logits


class ConcatFusionModel(nn.Module):
  """Fusion 1: Concatenates GraphSAGE graph embedding with BERT [CLS] embedding."""

  def __init__(
      self,
      hidden_dim: int = 64,
      bert_model_name: str = "bert-base-uncased",
      num_classes: int = 8,
      freeze_bert_layers: bool = True,
  ):
    super(ConcatFusionModel, self).__init__()
    self.gnn = ChordGraphSAGE(
        input_dim=12, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2
    )
    self.bert_encoder = BertLyricsEncoder(
        model_name=bert_model_name, freeze_bert=False
    )

    if freeze_bert_layers:
      freeze_bert_except_last_layers(
          self.bert_encoder, num_layers_trainable=2
      )

    fusion_dim = hidden_dim + 768
    self.classifier = nn.Linear(fusion_dim, num_classes)

  def forward(self, batch, input_ids, attention_mask, return_embedding=False):
    g = self.gnn(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    cls_emb, _ = self.bert_encoder(input_ids, attention_mask)

    fused_emb = torch.cat([g, cls_emb], dim=-1)

    if return_embedding:
      return fused_emb

    logits = self.classifier(fused_emb)
    return logits


class CrossAttentionFusionModel(nn.Module):
  """Fusion 2: GraphSAGE query attends over token-level BERT hidden states."""

  def __init__(
      self,
      hidden_dim: int = 64,
      bert_model_name: str = "bert-base-uncased",
      num_classes: int = 8,
      freeze_bert_layers: bool = True,
  ):
    super(CrossAttentionFusionModel, self).__init__()
    self.gnn = ChordGraphSAGE(
        input_dim=12, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2
    )
    self.bert_encoder = BertLyricsEncoder(
        model_name=bert_model_name, freeze_bert=False
    )

    if freeze_bert_layers:
      freeze_bert_except_last_layers(
          self.bert_encoder, num_layers_trainable=2
      )

    self.hidden_dim = hidden_dim

    # Linear projections for query, key, value scaled dot-product attention
    self.q_proj = nn.Linear(hidden_dim, hidden_dim)
    self.k_proj = nn.Linear(768, hidden_dim)
    self.v_proj = nn.Linear(768, hidden_dim)

    # Fusion takes g (hidden_dim) + attended_text (hidden_dim) = hidden_dim * 2
    self.classifier = nn.Linear(hidden_dim * 2, num_classes)

  def forward(self, batch, input_ids, attention_mask, return_embedding=False):
    # 1. Obtain Graph embedding g: (batch_size, hidden_dim)
    g = self.gnn(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

    # 2. Obtain BERT sequence states: (batch_size, seq_len, 768)
    _, full_last_hidden_state = self.bert_encoder(input_ids, attention_mask)

    # 3. Cross Attention Projections
    # Query: (batch_size, 1, hidden_dim)
    Q = self.q_proj(g).unsqueeze(1)
    # Key: (batch_size, seq_len, hidden_dim)
    K = self.k_proj(full_last_hidden_state)
    # Value: (batch_size, seq_len, hidden_dim)
    V = self.v_proj(full_last_hidden_state)

    # 4. Scaled Dot-Product Attention (QK^T / sqrt(d))
    # Scores: (batch_size, 1, seq_len)
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

    # 5. Mask out padding positions using attention_mask
    mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # 6. Weighted sum over values -> (batch_size, 1, hidden_dim) -> squeeze to (batch_size, hidden_dim)
    attended_text = torch.matmul(attn_weights, V).squeeze(1)

    # 7. Concatenate graph embedding and attended text representation
    fused_emb = torch.cat([g, attended_text], dim=-1)  # (batch_size, 64*2=128)

    if return_embedding:
      return fused_emb

    logits = self.classifier(fused_emb)
    return logits