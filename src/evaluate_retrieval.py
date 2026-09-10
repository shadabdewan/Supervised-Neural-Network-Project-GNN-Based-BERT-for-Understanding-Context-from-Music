import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch

from bert_encoder import tokenize_batch
from contrastive_model import ContrastiveGNNBERT


def compute_retrieval_metrics(similarity_matrix, direction="text2graph"):
  """Computes Recall@1, Recall@5, and Recall@10 metrics.

  - 'text2graph' (Caption -> Audio): For each caption (row), rank graphs (cols).
  - 'graph2text' (Audio -> Caption): For each graph (col), rank captions
  (rows).
  """
  # Matrix orientation: rows = queries, cols = targets
  if direction == "graph2text":
    sims = similarity_matrix.T  # Transpose so rows are graphs, cols are captions
  else:
    sims = similarity_matrix  # Rows are captions, cols are graphs

  num_queries, num_targets = sims.shape
  k10_limit = min(10, num_targets)

  ranks = []

  for i in range(num_queries):
    query_sims = sims[i]
    # Sort target indices in descending order of similarity
    sorted_indices = np.argsort(-query_sims)
    # Target i is the ground-truth match
    rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed rank
    ranks.append(rank)

  ranks = np.array(ranks)

  r1 = np.mean(ranks == 1) * 100.0
  r5 = np.mean(ranks <= 5) * 100.0
  r10 = np.mean(ranks <= k10_limit) * 100.0
  mrr = np.mean(1.0 / ranks)

  return {
      "R@1": r1,
      "R@5": r5,
      f"R@{k10_limit}": r10,
      "MRR": mrr,
      "k10_limit": k10_limit,
      "ranks": ranks,
      "sorted_indices": [
          np.argsort(-sims[i]) for i in range(num_queries)
      ],
  }


def main():
  # Setup paths
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent if script_dir.name == "src" else script_dir

  data_dir = project_root / "data" / "processed"
  results_dir = project_root / "results"
  results_dir.mkdir(parents=True, exist_ok=True)

  best_model_path = results_dir / "contrastive_best_model.pt"
  test_pt_path = data_dir / "contrastive_test_dataset.pt"

  csv_out_path = results_dir / "task4_retrieval_results.csv"
  txt_out_path = results_dir / "task4_qualitative_examples.txt"

  # 1. Load Test Dataset & Best Model
  if not test_pt_path.exists():
    print(f"Error: Test dataset not found at {test_pt_path}")
    return

  if not best_model_path.exists():
    print(f"Error: Best model checkpoint not found at {best_model_path}")
    return

  test_dataset = torch.load(test_pt_path)
  test_size = len(test_dataset)
  print(f"Loaded {test_size} test pairs.")

  device = torch.device("cpu")
  model = ContrastiveGNNBERT(
      embedding_dim=128, gnn_hidden_dim=64, freeze_bert_layers=True
  ).to(device)

  checkpoint = torch.load(best_model_path, map_location=device)
  model.load_state_dict(checkpoint["model_state"])
  model.eval()

  print(
      "Loaded best model checkpoint (Epoch"
      f" {checkpoint.get('epoch', 'Unknown')})."
  )

  # 2. Extract Embeddings for Full Test Set
  # Batch graphs
  graph_batch = Batch.from_data_list(test_dataset).to(device)

  # Extract captions and filenames
  captions = [data.caption for data in test_dataset]
  filenames = [
      getattr(data, "filename", f"sample_{i}.wav")
      for i, data in enumerate(test_dataset)
  ]

  # Tokenize text
  tokenized = tokenize_batch(captions, max_length=64)
  input_ids = tokenized["input_ids"].to(device)
  attention_mask = tokenized["attention_mask"].to(device)

  with torch.no_grad():
    graph_embeds = model.encode_graph(graph_batch)  # [test_size, 128]
    text_embeds = model.encode_text(
        input_ids, attention_mask
    )  # [test_size, 128]

    # Pairwise cosine similarity matrix (dot product of L2-normalized vectors)
    similarity_matrix = (
        (graph_embeds @ text_embeds.T).cpu().numpy()
    )  # [test_size, test_size]

  # 3. Compute Metrics
  # Note: similarity_matrix[i, j] is sim(graph_i, text_j)
  # For Text -> Graph: query text_j against graphs_i => similarity_matrix.T
  t2g_metrics = compute_retrieval_metrics(
      similarity_matrix.T, direction="text2graph"
  )
  g2t_metrics = compute_retrieval_metrics(
      similarity_matrix, direction="graph2text"
  )

  k_lim = t2g_metrics["k10_limit"]

  # 4. Print Results Table
  print("\n" + "=" * 65)
  print("TASK 4 CROSS-MODAL RETRIEVAL RESULTS")
  print("=" * 65)
  print(
      f"{'Direction':<22} | {'R@1':<8} | {'R@5':<8} |"
      f" {'R@' + str(k_lim):<8} | {'MRR':<8}"
  )
  print("-" * 65)
  print(
      f"{'Caption -> Audio':<22} | {t2g_metrics['R@1']:6.2f}% |"
      f" {t2g_metrics['R@5']:6.2f}% | {t2g_metrics['R@' + str(k_lim)]:6.2f}% |"
      f" {t2g_metrics['MRR']:.4f}"
  )
  print(
      f"{'Audio -> Caption':<22} | {g2t_metrics['R@1']:6.2f}% |"
      f" {g2t_metrics['R@5']:6.2f}% | {g2t_metrics['R@' + str(k_lim)]:6.2f}% |"
      f" {g2t_metrics['MRR']:.4f}"
  )
  print("=" * 65)

  # 5. Save Results Table to CSV
  with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Direction", "R@1 (%)", "R@5 (%)", f"R@{k_lim} (%)", "MRR"]
    )
    writer.writerow([
        "Caption -> Audio",
        f"{t2g_metrics['R@1']:.2f}",
        f"{t2g_metrics['R@5']:.2f}",
        f"{t2g_metrics['R@' + str(k_lim)]:.2f}",
        f"{t2g_metrics['MRR']:.4f}",
    ])
    writer.writerow([
        "Audio -> Caption",
        f"{g2t_metrics['R@1']:.2f}",
        f"{g2t_metrics['R@5']:.2f}",
        f"{g2t_metrics['R@' + str(k_lim)]:.2f}",
        f"{g2t_metrics['MRR']:.4f}",
    ])

  print(f"\nSaved retrieval table to: {csv_out_path.resolve()}")

  # 6. Qualitative Examples (Caption -> Audio Retrieval)
  random.seed(42)
  num_examples = min(10, test_size)
  sample_indices = random.sample(range(test_size), num_examples)

  qualitative_text_lines = []
  qualitative_text_lines.append(
      "=========================================================================\n"
  )
  qualitative_text_lines.append(
      "TASK 4: QUALITATIVE CAPTION-TO-AUDIO RETRIEVAL EXAMPLES\n"
  )
  qualitative_text_lines.append(
      "=========================================================================\n\n"
  )

  for ex_idx, q_idx in enumerate(sample_indices, start=1):
    caption_text = captions[q_idx]
    true_filename = filenames[q_idx]

    # For caption q_idx, similarity vector against all audio graphs is row q_idx of similarity_matrix.T
    sims_for_caption = similarity_matrix.T[q_idx]
    top_indices = np.argsort(-sims_for_caption)[:3]

    top_matches = []
    is_in_top3 = False

    for rank_i, target_idx in enumerate(top_indices, start=1):
      retrieved_fn = filenames[target_idx]
      score = sims_for_caption[target_idx]
      match_flag = retrieved_fn == true_filename
      if match_flag:
        is_in_top3 = True
      top_matches.append(
          (rank_i, retrieved_fn, score, " [MATCH]" if match_flag else "")
      )

    status_icon = "✓" if is_in_top3 else "✗"

    block = (
        f"Example {ex_idx:02d} [{status_icon} Top-3 Match]\n"
        f"Caption       : \"{caption_text}\"\n"
        f"Target Audio  : {true_filename}\n"
        f"Top-3 Retrieved Audio Clips:\n"
    )
    for r, fn, sc, flag in top_matches:
      block += f"  Rank {r}: {fn:<20} (Similarity: {sc:.4f}){flag}\n"
    block += "-" * 75 + "\n"

    qualitative_text_lines.append(block)

  # Write qualitative report file
  with open(txt_out_path, "w", encoding="utf-8") as f:
    f.writelines(qualitative_text_lines)

  print(f"Saved qualitative analysis to: {txt_out_path.resolve()}\n")

  # Print a snippet of qualitative examples to console
  print("Qualitative Sample Preview:")
  print("".join(qualitative_text_lines[3:15]))


if __name__ == "__main__":
  main()
  