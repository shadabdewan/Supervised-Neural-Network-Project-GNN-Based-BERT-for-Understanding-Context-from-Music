import logging
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from transformers import BertTokenizer

from bert_encoder import tokenize_batch
from contrastive_model import ContrastiveGNNBERT

# Configure logging for batch errors
logging.basicConfig(
    filename="contrastive_training_errors.log",
    level=logging.ERROR,
    format="%(asctime)s | Batch Error: %(message)s",
)


def custom_contrastive_collate(data_list, tokenizer):
  """Custom collate function for PyG graphs and BERT text tokens.

  - Batches PyG Data graphs using PyG's Batch.from_data_list
  - Extracts captions from each Data object and tokenizes using BERT
  """
  # 1. Batch PyG graphs
  graph_batch = Batch.from_data_list(data_list)

  # 2. Collect raw caption strings from each sample in the batch
  captions = [data.caption for data in data_list]

  # 3. Tokenize captions using BertTokenizer with max_length=64
  tokenized = tokenize_batch(captions, tokenizer=tokenizer, max_length=64)

  return graph_batch, tokenized["input_ids"], tokenized["attention_mask"]


def save_loss_plot(train_losses, val_losses, plot_path):
  """Plots and saves the training and validation loss curves."""
  epochs = range(1, len(train_losses) + 1)
  plt.figure(figsize=(8, 5))
  plt.plot(epochs, train_losses, label="Train InfoNCE Loss", color="#1f77b4")
  plt.plot(epochs, val_losses, label="Val InfoNCE Loss", color="#ff7f0e")
  plt.xlabel("Epochs")
  plt.ylabel("InfoNCE Loss")
  plt.title("Task 4: Cross-Modal Contrastive Training Curve")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.savefig(plot_path, dpi=300)
  plt.close()


def main():
  # Setup paths
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent if script_dir.name == "src" else script_dir

  data_dir = project_root / "data" / "processed"
  results_dir = project_root / "results"
  checkpoints_dir = results_dir / "checkpoints"
  plots_dir = results_dir / "plots"

  checkpoints_dir.mkdir(parents=True, exist_ok=True)
  plots_dir.mkdir(parents=True, exist_ok=True)

  train_pt_path = data_dir / "contrastive_train_dataset.pt"
  val_pt_path = data_dir / "contrastive_val_dataset.pt"

  checkpoint_path = checkpoints_dir / "contrastive_checkpoint.pt"
  best_model_path = results_dir / "contrastive_best_model.pt"
  plot_path = plots_dir / "contrastive_training_curve.png"

  # 1. Load Dataset Splits
  if not train_pt_path.exists() or not val_pt_path.exists():
    print(
        f"Error: Dataset files not found in {data_dir}. Run"
        " contrastive_dataset_loader.py first."
    )
    return

  train_dataset = torch.load(train_pt_path)
  val_dataset = torch.load(val_pt_path)

  BATCH_SIZE = 8
  NUM_EPOCHS = 30
  TEMPERATURE = 0.15
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      collate_fn=lambda batch: custom_contrastive_collate(batch, tokenizer),
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      collate_fn=lambda batch: custom_contrastive_collate(batch, tokenizer),
  )

  device = torch.device("cpu")
  print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val pairs.")

  # 2. Instantiate Model & Optimizer with parameter groups
  model = ContrastiveGNNBERT(
      embedding_dim=128, gnn_hidden_dim=64, freeze_bert_layers=True
  ).to(device)

  # Separate parameters: use gentler learning rates for this small dataset.
  bert_params = list(model.bert_encoder.parameters())
  other_params = (
      list(model.gnn_encoder.parameters())
      + list(model.graph_projection.parameters())
      + list(model.text_projection.parameters())
  )

  optimizer = torch.optim.Adam([
      {"params": bert_params, "lr": 5e-6},
      {"params": other_params, "lr": 5e-4},
  ])

  start_epoch = 1
  train_loss_history = []
  val_loss_history = []
  best_val_loss = float("inf")

  # 3. Resume Checkpoint Logic
  if checkpoint_path.exists():
    print(f"Found existing checkpoint at {checkpoint_path}. Resuming...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    train_loss_history = checkpoint["train_loss_history"]
    val_loss_history = checkpoint["val_loss_history"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    if start_epoch > NUM_EPOCHS:
      print(
          f"Training already complete ({NUM_EPOCHS} epochs). Skipping"
          " training."
      )
      return

  print(f"Starting training from epoch {start_epoch} to {NUM_EPOCHS}...\n")

  # 4. Training Loop
  for epoch in range(start_epoch, NUM_EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    train_batches = 0

    for batch_idx, (
        graph_batch,
        input_ids,
        attention_mask,
    ) in enumerate(train_loader):
      try:
        graph_batch = graph_batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        loss, _ = model(
          graph_batch,
          input_ids,
          attention_mask,
          temperature=TEMPERATURE,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        train_batches += 1
      except Exception as e:
        err_msg = f"Epoch {epoch} | Train Batch {batch_idx}: {str(e)}"
        logging.error(err_msg)
        print(f"Warning: Skipped train batch due to error: {e}")

    avg_train_loss = (
        total_train_loss / train_batches if train_batches > 0 else 0.0
    )
    train_loss_history.append(avg_train_loss)

    # Validation Pass
    model.eval()
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
      for batch_idx, (
          graph_batch,
          input_ids,
          attention_mask,
      ) in enumerate(val_loader):
        try:
          graph_batch = graph_batch.to(device)
          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)

          loss, _ = model(
              graph_batch,
              input_ids,
              attention_mask,
              temperature=TEMPERATURE,
          )
          total_val_loss += loss.item()
          val_batches += 1
        except Exception as e:
          err_msg = f"Epoch {epoch} | Val Batch {batch_idx}: {str(e)}"
          logging.error(err_msg)
          print(f"Warning: Skipped val batch due to error: {e}")

    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
    val_loss_history.append(avg_val_loss)

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )
    if len(val_loss_history) > 1:
      previous_val_loss = val_loss_history[-2]
      change = avg_val_loss - previous_val_loss
      trend = "improved" if change < 0 else "worsened" if change > 0 else "unchanged"
      print(
        f"  Val trend: {trend} by {abs(change):.4f} "
        f"(previous: {previous_val_loss:.4f})"
      )

    # Save best model
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(
          {
              "epoch": epoch,
              "model_state": model.state_dict(),
              "val_loss": avg_val_loss,
          },
          best_model_path,
      )

    # Save full state checkpoint for resumability
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

  # 5. Plot Loss Curves
  save_loss_plot(train_loss_history, val_loss_history, plot_path)

  print("\n" + "=" * 50)
  print("CONTRASTIVE TRAINING COMPLETE")
  print("=" * 50)
  print(f"Best Val Loss     : {best_val_loss:.4f}")
  print(f"Best Model Saved  : {best_model_path.resolve()}")
  print(f"Checkpoint Saved  : {checkpoint_path.resolve()}")
  print(f"Loss Curve Saved  : {plot_path.resolve()}")
  print("=" * 50)


if __name__ == "__main__":
  main()