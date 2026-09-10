import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from bert_encoder import tokenize_batch
from fusion_dataset_loader import create_fusion_dataset
from fusion_model import (
    BERTOnlyModel,
    ConcatFusionModel,
    CrossAttentionFusionModel,
    GNNOnlyModel,
)

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


def custom_collate_fn(batch_list, model_type):
  """Custom collate logic returning PyG Batch graphs and tokenized lyrics inputs."""
  pyg_batch = Batch.from_data_list(batch_list)

  if model_type == "gnn":
    return pyg_batch, None, None

  # For BERT-only or fusion models, tokenize batch lyrics strings
  lyrics_list = [data.lyrics for data in batch_list]
  tokenizer = torch.hub.load_state_dict_from_url if False else None
  tokenized = tokenize_batch(
      lyrics_list,
      tokenizer=None,
      max_length=256,
      padding="max_length",
      truncation=True,
  )

  input_ids = tokenized["input_ids"]
  attention_mask = tokenized["attention_mask"]

  return pyg_batch, input_ids, attention_mask


def run_epoch(
    model, data_loader, criterion, optimizer, device, model_type, is_train=True
):
  if is_train:
    model.train()
  else:
    model.eval()

  total_loss = 0.0
  all_preds = []
  all_targets = []

  context = torch.enable_grad() if is_train else torch.no_grad()

  with context:
    for batch_data in data_loader:
      pyg_batch, input_ids, attention_mask = batch_data
      pyg_batch = pyg_batch.to(device)
      targets = pyg_batch.y.to(device)

      if input_ids is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

      if is_train:
        optimizer.zero_grad()

      # Forward pass route based on model parameters
      if model_type == "gnn":
        logits = model(pyg_batch)
      elif model_type == "bert":
        logits = model(input_ids, attention_mask)
      else:  # concat or cross_attn fusion
        logits = model(pyg_batch, input_ids, attention_mask)

      loss = criterion(logits, targets)

      if is_train:
        loss.backward()
        optimizer.step()

      total_loss += loss.item() * pyg_batch.num_graphs
      preds = logits.argmax(dim=1)

      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(targets.cpu().numpy())

  epoch_loss = total_loss / len(all_targets)
  epoch_acc = accuracy_score(all_targets, all_preds)

  return epoch_loss, epoch_acc, all_preds, all_targets


def save_epoch_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch_completed,
    best_val_accuracy,
    train_loss_history,
    train_acc_history,
    val_loss_history,
    val_acc_history,
):
  checkpoint = {
      "epoch": int(epoch_completed),
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "best_val_accuracy": float(best_val_accuracy),
      "train_loss_history": train_loss_history,
      "train_acc_history": train_acc_history,
      "val_loss_history": val_loss_history,
      "val_acc_history": val_acc_history,
      "rng_state": {
          "torch": torch.get_rng_state(),
          "numpy": np.random.get_state(),
      },
  }
  torch.save(checkpoint, checkpoint_path)


def train_and_evaluate_variant(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    results_dir,
    device,
    model_type,
):
  print(f"\n{'='*20} Training Model Variant: {model_name} {'='*20}")

  checkpoint_dir = results_dir / "checkpoints"
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pt"
  best_model_path = results_dir / f"fusion_{model_name}_best.pt"

  bert_params = []
  other_params = []

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if "bert_encoder" in name or "bert" in name:
      bert_params.append(param)
    else:
      other_params.append(param)

  optimizer = optim.Adam([
      {"params": bert_params, "lr": 2e-5},
      {"params": other_params, "lr": 1e-3},
  ])

  criterion = nn.CrossEntropyLoss()
  best_val_acc = 0.0
  train_loss_history = []
  train_acc_history = []
  val_loss_history = []
  val_acc_history = []
  start_epoch = 1

  if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      best_val_acc = checkpoint.get("best_val_accuracy", 0.0)
      train_loss_history = checkpoint.get("train_loss_history", [])
      train_acc_history = checkpoint.get("train_acc_history", [])
      val_loss_history = checkpoint.get("val_loss_history", [])
      val_acc_history = checkpoint.get("val_acc_history", [])
      saved_epoch = int(checkpoint.get("epoch", 0))
      if saved_epoch >= 10:
        print(f"{model_name} already fully trained - skipping to evaluation")
        if best_model_path.exists():
          model.load_state_dict(torch.load(best_model_path, weights_only=True))
        else:
          model.load_state_dict(checkpoint["model_state_dict"])
      else:
        start_epoch = saved_epoch + 1
        print(f"Resuming {model_name} from epoch {start_epoch}/10")

  if checkpoint_path.exists() and int(torch.load(checkpoint_path, map_location=device).get("epoch", 0)) >= 10:
    pass
  else:
    for epoch in range(start_epoch, 11):
      try:
        tr_loss, tr_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            model_type,
            is_train=True,
        )
        val_loss, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            model_type,
            is_train=False,
        )

        train_loss_history.append(float(tr_loss))
        train_acc_history.append(float(tr_acc))
        val_loss_history.append(float(val_loss))
        val_acc_history.append(float(val_acc))

        print(
            f"Epoch {epoch:02d}/10 | Train Loss: {tr_loss:.4f} | Train Acc:"
            f" {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc:"
            f" {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
          best_val_acc = val_acc
          torch.save(model.state_dict(), best_model_path)
          print(
              f"  --> Saved new best checkpoint for {model_name} (Val Acc:"
              f" {val_acc*100:.2f}%)"
          )

        save_epoch_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            best_val_acc,
            train_loss_history,
            train_acc_history,
            val_loss_history,
            val_acc_history,
        )

      except (Exception, KeyboardInterrupt) as exc:
        tag = "interrupt" if isinstance(exc, KeyboardInterrupt) else "error"
        print(
            f"Warning: {model_name} hit a {tag} during epoch {epoch}; saving recovery checkpoint and stopping this model. Error: {exc}"
        )
        save_epoch_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            max(epoch - 1, 0),
            best_val_acc,
            train_loss_history,
            train_acc_history,
            val_loss_history,
            val_acc_history,
        )
        break

  # Final Evaluation on Test Set
  print(f"\nEvaluating Best Checkpoint for {model_name} on Test Set...")
  if best_model_path.exists():
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
  elif checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])

  _, test_acc, test_preds, test_targets = run_epoch(
      model,
      test_loader,
      criterion,
      optimizer,
      device,
      model_type,
      is_train=False,
  )

  macro_f1 = f1_score(test_targets, test_preds, average="macro")
  micro_f1 = f1_score(test_targets, test_preds, average="micro")

  print(f"Test Accuracy : {test_acc*100:.2f}%")
  print(f"Macro F1 Score: {macro_f1:.4f}")
  print(f"Micro F1 Score: {micro_f1:.4f}")
  print("\nClassification Report:")
  print(
      classification_report(
          test_targets, test_preds, target_names=GENRE_NAMES, digits=4
      )
  )

  return (
      {
          "model_name": model_name,
          "test_accuracy": round(test_acc, 4),
          "macro_f1": round(macro_f1, 4),
          "micro_f1": round(micro_f1, 4),
      },
      test_preds,
      test_targets,
  )


def ensure_fusion_datasets(data_processed_dir: Path):
  required_files = [
      data_processed_dir / "fusion_train_dataset.pt",
      data_processed_dir / "fusion_val_dataset.pt",
      data_processed_dir / "fusion_test_dataset.pt",
  ]
  if all(path.exists() for path in required_files):
    return

  print("Fusion dataset files not found. Building them now...")
  create_fusion_dataset()

  missing = [path.name for path in required_files if not path.exists()]
  if missing:
    raise FileNotFoundError(
        f"Fusion dataset generation did not produce: {missing}. "
        "Check data/processed/chord_graphs and lyrics_transcriptions.csv."
    )


def main():
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  data_processed_dir = project_root / "data" / "processed"
  results_dir = project_root / "results"
  results_dir.mkdir(parents=True, exist_ok=True)

  ensure_fusion_datasets(data_processed_dir)

  device = torch.device("cpu")
  batch_size = 16

  # 1. Load splits
  train_data = torch.load(
      data_processed_dir / "fusion_train_dataset.pt", weights_only=False
  )
  val_data = torch.load(
      data_processed_dir / "fusion_val_dataset.pt", weights_only=False
  )
  test_data = torch.load(
      data_processed_dir / "fusion_test_dataset.pt", weights_only=False
  )

  models_to_test = [
      ("GNN-only", GNNOnlyModel(), "gnn"),
      ("BERT-only", BERTOnlyModel(), "bert"),
      ("Concat-fusion", ConcatFusionModel(), "fusion"),
      ("Cross-attention-fusion", CrossAttentionFusionModel(), "fusion"),
  ]

  ablation_results = []

  for model_name, model, model_type in models_to_test:
    # Explicit binding of model_type via default argument in lambda
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b, mt=model_type: custom_collate_fn(b, mt),
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b, mt=model_type: custom_collate_fn(b, mt),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b, mt=model_type: custom_collate_fn(b, mt),
    )

    metrics, test_preds, test_targets = train_and_evaluate_variant(
        model_name,
        model,
        train_loader,
        val_loader,
        test_loader,
        results_dir,
        device,
        model_type,
    )
    ablation_results.append(metrics)

    # Save predictions CSV for case study analysis
    pred_csv_path = results_dir / f"fusion_{model_name}_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["filename", "true_label", "predicted_label"])
      for idx, data_item in enumerate(test_data):
        writer.writerow([
            data_item.filename,
            GENRE_NAMES[test_targets[idx]],
            GENRE_NAMES[test_preds[idx]],
        ])

  # 4. Save Final Ablation Results Table
  csv_table_path = results_dir / "task3_ablation_results.csv"
  with open(csv_table_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model_name", "test_accuracy", "macro_f1", "micro_f1"]
    )
    writer.writeheader()
    writer.writerows(ablation_results)

  print("\n" + "=" * 60)
  print("TASK 3 ABLATION STUDY COMPARISON TABLE")
  print("=" * 60)
  print(
      f"{'Model Variant':<24} | {'Test Acc':<8} | {'Macro F1':<8} | {'Micro F1':<8}"
  )
  print("-" * 60)
  for r in ablation_results:
    print(
        f"{r['model_name']:<24} | {r['test_accuracy']*100:>6.2f}% |"
        f" {r['macro_f1']:>8.4f} | {r['micro_f1']:>8.4f}"
    )
  print("=" * 60)
  print(f"Results saved to: {csv_table_path.resolve()}")


if __name__ == "__main__":
  main()