import csv
from pathlib import Path
import time
from cnn_model import MelSpecCNN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Same alphabetical mapping used in dataset_loader.py for the GNN model
GENRE_MAP = {
    "blues": 0,
    "country": 1,
    "disco": 2,
    "hiphop": 3,
    "metal": 4,
    "pop": 5,
    "reggae": 6,
    "rock": 7,
}
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


class MelSpectrogramDataset(Dataset):
  """PyTorch Dataset that loads .npy files on-demand and applies training-set normalization."""

  def __init__(
      self,
      file_entries: list,
      base_dir: Path,
      mean: float = 0.0,
      std: float = 1.0,
  ):
    self.file_entries = file_entries
    self.base_dir = base_dir
    self.mean = mean
    self.std = std

  def __len__(self):
    return len(self.file_entries)

  def __getitem__(self, idx):
    entry = self.file_entries[idx]
    npy_path = self.base_dir / entry["filename"]

    # Load spectrogram array shape (128, 1300)
    spec = np.load(npy_path).astype(np.float32)

    # Standardize feature values using training set statistics
    spec = (spec - self.mean) / (self.std + 1e-7)

    # Add channel dimension -> (1, 128, 1300)
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
    label_tensor = torch.tensor(GENRE_MAP[entry["genre"]], dtype=torch.long)

    return spec_tensor, label_tensor


def compute_dataset_stats(file_entries, base_dir):
  """Computes global pixel-level mean and std across training set .npy files."""
  print("Computing global mean and std across training set...")
  total_sum = 0.0
  total_sq_sum = 0.0
  total_pixels = 0

  for entry in file_entries:
    spec = np.load(base_dir / entry["filename"]).astype(np.float64)
    total_sum += spec.sum()
    total_sq_sum += (spec**2).sum()
    total_pixels += spec.size

  mean = total_sum / total_pixels
  var = (total_sq_sum / total_pixels) - (mean**2)
  std = np.sqrt(max(var, 1e-7))

  print(f"Calculated Train Stats -> Mean: {mean:.4f}, Std: {std:.4f}")
  return mean, std


def train_one_epoch(model, loader, criterion, optimizer, device):
  model.train()
  total_loss = 0.0
  correct = 0
  total_samples = 0

  for x_batch, y_batch in loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    optimizer.zero_grad()
    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    loss.backward()
    optimizer.step()

    total_loss += loss.item() * len(y_batch)
    preds = logits.argmax(dim=1)
    correct += (preds == y_batch).sum().item()
    total_samples += len(y_batch)

  return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
  model.eval()
  total_loss = 0.0
  correct = 0
  total_samples = 0

  for x_batch, y_batch in loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    total_loss += loss.item() * len(y_batch)
    preds = logits.argmax(dim=1)
    correct += (preds == y_batch).sum().item()
    total_samples += len(y_batch)

  return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def generate_test_report(model, loader, device):
  model.eval()
  all_preds = []
  all_targets = []

  for x_batch, y_batch in loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    logits = model(x_batch)
    preds = logits.argmax(dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_targets.extend(y_batch.cpu().numpy())

  return classification_report(
      all_targets, all_preds, target_names=GENRE_NAMES, digits=4
  )


def plot_training_curves(history, save_path):
  epochs = range(1, len(history["train_loss"]) + 1)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
  ax1.plot(
      epochs,
      history["val_loss"],
      label="Val Loss",
      color="red",
      linestyle="--",
  )
  ax1.set_title("CNN Cross-Entropy Loss")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")
  ax1.legend()
  ax1.grid(True)

  ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="blue")
  ax2.plot(
      epochs,
      history["val_acc"],
      label="Val Accuracy",
      color="red",
      linestyle="--",
  )
  ax2.set_title("CNN Classification Accuracy")
  ax2.set_xlabel("Epochs")
  ax2.set_ylabel("Accuracy")
  ax2.legend()
  ax2.grid(True)

  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()


def main():
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  data_dir = project_root / "data" / "processed" / "mel_spectrograms"
  labels_csv_path = data_dir / "labels.csv"

  results_dir = project_root / "results"
  plots_dir = results_dir / "plots"

  results_dir.mkdir(parents=True, exist_ok=True)
  plots_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cpu")
  print(f"Target execution device: {device}")

  # 1. Load manifest file
  file_entries = []
  with open(labels_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      file_entries.append(row)

  targets = [GENRE_MAP[e["genre"]] for e in file_entries]

  # 3. Stratified Split matching dataset_loader.py logic (seed=42, 80/10/10)
  train_entries, temp_entries, train_y, temp_y = train_test_split(
      file_entries,
      targets,
      test_size=0.20,
      stratify=targets,
      random_state=42,
  )

  val_entries, test_entries, _, _ = train_test_split(
      temp_entries,
      temp_y,
      test_size=0.50,
      stratify=temp_y,
      random_state=42,
  )

  # Compute training normalization stats
  train_mean, train_std = compute_dataset_stats(train_entries, data_dir)

  # 2. Instantiate PyTorch Datasets & DataLoaders
  train_dataset = MelSpectrogramDataset(
      train_entries, data_dir, mean=train_mean, std=train_std
  )
  val_dataset = MelSpectrogramDataset(
      val_entries, data_dir, mean=train_mean, std=train_std
  )
  test_dataset = MelSpectrogramDataset(
      test_entries, data_dir, mean=train_mean, std=train_std
  )

  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

  # 4. Instantiate Model, Optimizer, Loss
  model = MelSpecCNN(output_dim=8, dropout=0.3).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  best_val_acc = 0.0
  best_model_path = results_dir / "best_cnn_model.pt"

  history = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": [],
  }

  epochs = 50
  print("\nStarting CNN Model Training (50 Epochs)...")
  print("-" * 75)

  # 5. Training loop
  for epoch in range(1, epochs + 1):
    start_time = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    elapsed_time = time.time() - start_time

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Time: {elapsed_time:.2f}s | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
    )

    # Print CPU runtime estimation warning after epoch 1
    if epoch == 1:
      est_total_min = (elapsed_time * (epochs - 1)) / 60.0
      print(
          f"  [CPU Execution Notice] Epoch 1 took {elapsed_time:.1f}s."
          f" Estimated time for remaining 49 epochs: ~{est_total_min:.1f} min."
      )

    # Save checkpoint
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), best_model_path)
      print(f"  --> Saved best CNN model checkpoint (Val Acc: {val_acc*100:.2f}%)")

  print("-" * 75)
  print(f"Training completed. Best Validation Accuracy: {best_val_acc*100:.2f}%")

  # 6. Final Evaluation
  print("\nEvaluating best CNN model on Test Set...")
  model.load_state_dict(torch.load(best_model_path, weights_only=True))

  test_loss, cnn_test_acc = evaluate(model, test_loader, criterion, device)
  print(
      f"\nFinal CNN Test Loss: {test_loss:.4f} | Final CNN Test Accuracy:"
      f" {cnn_test_acc*100:.2f}%\n"
  )

  print("Classification Report (CNN Baseline - Test Set):")
  report = generate_test_report(model, test_loader, device)
  print(report)

  # 7. Save Training Curves Plot
  curve_plot_path = plots_dir / "cnn_training_curve.png"
  plot_training_curves(history, curve_plot_path)
  print(f"Saved CNN training curves to: {curve_plot_path.resolve()}")

  # 8. Side-by-side comparison readout
  # TODO: Update gnn_test_acc string below with the actual test accuracy from train.py run
  gnn_test_acc_str = "[FILL_IN_GNN_ACC_HERE]"
  print("\n" + "=" * 60)
  print("MODEL COMPARISON (TEST ACCURACY)")
  print("=" * 60)
  print(
      f"GNN test accuracy: {gnn_test_acc_str} | CNN test accuracy:"
      f" {cnn_test_acc*100:.2f}%"
  )
  print("=" * 60)


if __name__ == "__main__":
  main()