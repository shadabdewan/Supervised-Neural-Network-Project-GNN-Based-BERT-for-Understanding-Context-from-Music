from pathlib import Path
from bert_encoder import tokenize_batch
from fusion_model import CrossAttentionFusionModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch_geometric.data import Batch

GENRE_NAMES = [
    "blues",
    "country",
    "disco",
    "hiphop",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def extract_fused_embeddings():
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  data_processed_dir = project_root / "data" / "processed"
  results_dir = project_root / "results"
  plots_dir = results_dir / "plots"
  plots_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cpu")

  # 1. Load Test Dataset and Checkpoint
  test_data = torch.load(
      data_processed_dir / "fusion_test_dataset.pt", weights_only=False
  )
  checkpoint_path = results_dir / "fusion_Cross-attention-fusion_best.pt"

  model = CrossAttentionFusionModel(hidden_dim=64, num_classes=8).to(device)
  model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  model.eval()

  # Batch inputs
  pyg_batch = Batch.from_data_list(test_data).to(device)
  lyrics_list = [d.lyrics for d in test_data]

  tokenized = tokenize_batch(
      lyrics_list,
      tokenizer=None,
      max_length=256,
      padding="max_length",
      truncation=True,
  )
  input_ids = tokenized["input_ids"].to(device)
  attention_mask = tokenized["attention_mask"].to(device)

  # 2. Forward pass to extract fused embeddings
  with torch.no_grad():
    fused_embeddings = model(
        pyg_batch, input_ids, attention_mask, return_embedding=True
    )

  embeddings_np = fused_embeddings.cpu().numpy()
  targets_np = pyg_batch.y.cpu().numpy()

  # 3. Dimensionality Reduction via t-SNE
  print("Running t-SNE reduction (perplexity=30, random_state=42)...")
  tsne = TSNE(n_components=2, perplexity=30, random_state=42)
  embeddings_2d = tsne.fit_transform(embeddings_np)

  # 4. Generate Scatter Plot
  plt.figure(figsize=(10, 8))
  colors = plt.cm.tab10(np.linspace(0, 1, 8))

  for genre_id, genre_name in enumerate(GENRE_NAMES):
    idx = np.where(targets_np == genre_id)[0]
    plt.scatter(
        embeddings_2d[idx, 0],
        embeddings_2d[idx, 1],
        color=colors[genre_id],
        label=genre_name,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
        s=60,
    )

  plt.title(
      "t-SNE Visualization of Fused Embeddings (Cross-Attention)",
      fontsize=14,
      pad=15,
  )
  plt.xlabel("t-SNE Dimension 1", fontsize=11)
  plt.ylabel("t-SNE Dimension 2", fontsize=11)
  plt.legend(title="True Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()

  save_path = plots_dir / "tsne_fusion_embeddings.png"
  plt.savefig(save_path, dpi=300)
  plt.close()

  # 5. Console Output
  print(
      "\nt-SNE plot shows how well the fused embeddings separate by genre -"
      " clusters indicate the fusion model has learned genre-discriminative"
      " representations."
  )
  print(f"Saved t-SNE plot to: {save_path.resolve()}")


if __name__ == "__main__":
  extract_fused_embeddings()