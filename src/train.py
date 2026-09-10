from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from gnn_model import ChordGraphSAGE

# Genre mapping reference (alphabetical order)
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


def train_one_epoch(model, loader, criterion, optimizer, device):
  """Executes a single training epoch across all dataset batches."""
  model.train()
  total_loss = 0.0
  correct = 0
  total_samples = 0

  for batch in loader:
    batch = batch.to(device)
    optimizer.zero_grad()

    # Forward pass: obtain logits from model
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

    # Loss computation:
    # DEVIATION NOTE: Single-label multi-class classification using CrossEntropyLoss
    loss = criterion(out, batch.y)

    loss.backward()
    optimizer.step()

    total_loss += loss.item() * batch.num_graphs
    preds = out.argmax(dim=1)
    correct += (preds == batch.y).sum().item()
    total_samples += batch.num_graphs

  return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
  """Evaluates the model on validation or test dataset batches."""
  model.eval()
  total_loss = 0.0
  correct = 0
  total_samples = 0

  for batch in loader:
    batch = batch.to(device)
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    loss = criterion(out, batch.y)

    total_loss += loss.item() * batch.num_graphs
    preds = out.argmax(dim=1)
    correct += (preds == batch.y).sum().item()
    total_samples += batch.num_graphs

  return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def generate_test_report(model, loader, device):
  """Generates precision, recall, and F1 per class using sklearn classification_report."""
  model.eval()
  all_preds = []
  all_targets = []

  for batch in loader:
    batch = batch.to(device)
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    preds = out.argmax(dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_targets.extend(batch.y.cpu().numpy())

  return classification_report(
      all_targets, all_preds, target_names=GENRE_NAMES, digits=4
  )


def plot_training_curves(history, save_path):
  """Plots and saves loss and accuracy metrics over training epochs."""
  epochs = range(1, len(history["train_loss"]) + 1)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Loss plot
  ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
  ax1.plot(
      epochs,
      history["val_loss"],
      label="Val Loss",
      color="red",
      linestyle="--",
  )
  ax1.set_title("Cross-Entropy Loss History")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")
  ax1.legend()
  ax1.grid(True)

  # Accuracy plot
  ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="blue")
  ax2.plot(
      epochs,
      history["val_acc"],
      label="Val Accuracy",
      color="red",
      linestyle="--",
  )
  ax2.set_title("Classification Accuracy History")
  ax2.set_xlabel("Epochs")
  ax2.set_ylabel("Accuracy")
  ax2.legend()
  ax2.grid(True)

  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()


def main():
  # Robust Path Configuration (Resolves relative to project root)
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  data_dir = project_root / "data" / "processed"
  results_dir = project_root / "results"
  plots_dir = results_dir / "plots"

  results_dir.mkdir(parents=True, exist_ok=True)
  plots_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cpu")
  print(f"Running execution pipeline on target device: {device}")

  # 1. Load dataset splits
  print(f"Loading preprocessed dataset splits from {data_dir}...")
  train_dataset = torch.load(data_dir / "train_dataset.pt", weights_only=False)
  val_dataset = torch.load(data_dir / "val_dataset.pt", weights_only=False)
  test_dataset = torch.load(data_dir / "test_dataset.pt", weights_only=False)

  print(
      f"Loaded: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test"
      f" ({len(test_dataset)})"
  )

  # 2. PyG DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

  # 3. Instantiate model, loss, and optimizer
  model = ChordGraphSAGE(
      input_dim=12, hidden_dim=64, output_dim=8, num_layers=2, dropout=0.3
  ).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  best_val_acc = 0.0
  best_model_path = results_dir / "best_model.pt"

  history = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": [],
  }

  # 4. Training loop
  epochs = 50
  print("\nStarting GNN Training Process...")
  print("-" * 75)

  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train"
        f" Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc:"
        f" {val_acc*100:.2f}%"
    )

    # Checkpoint best model
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), best_model_path)
      print(
          f"  --> Saved new best model checkpoint (Val Acc:"
          f" {val_acc*100:.2f}%)"
      )

  print("-" * 75)
  print(f"Training completed. Best Validation Accuracy: {best_val_acc*100:.2f}%")

  # 5. Evaluate best model on test set
  print("\nLoading best model weights for final evaluation on Test Set...")
  model.load_state_dict(torch.load(best_model_path, weights_only=True))

  test_loss, test_acc = evaluate(model, test_loader, criterion, device)
  print(
      f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy:"
      f" {test_acc*100:.2f}%\n"
  )

  print("Detailed Classification Report (Test Set):")
  report = generate_test_report(model, test_loader, device)
  print(report)

  # 6. Generate training curve plots
  curve_plot_path = plots_dir / "training_curve.png"
  plot_training_curves(history, curve_plot_path)
  print(f"Saved training curves to: {curve_plot_path.resolve()}")


if __name__ == "__main__":
  main()