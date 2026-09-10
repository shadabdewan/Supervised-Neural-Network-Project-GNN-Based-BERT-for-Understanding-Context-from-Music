import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data


def load_excluded_filenames(excluded_txt_path: Path) -> set:
  """Loads the set of excluded degenerate graph filenames."""
  if not excluded_txt_path.exists():
    return set()

  excluded = set()
  with open(excluded_txt_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line and not line.startswith("#"):
        excluded.add(line)
  return excluded


def json_to_pyg_data(graph_data: dict) -> Data:
  """Converts graph JSON dictionary (nodes, edges, caption) to a PyG Data object."""
  nodes = graph_data.get("nodes", [])
  edges = graph_data.get("edges", [])
  caption = graph_data.get("caption", "")

  # 1. Node features (x): Stacked chroma vectors [num_nodes, 12]
  if not nodes:
    raise ValueError("graph contains no nodes")

  # MusicCaps graphs use the same JSON schema as the GTZAN graphs:
  # each node stores its 12-dimensional vector under `features`.
  chroma_features = [node.get("features", node.get("chroma")) for node in nodes]
  if any(features is None for features in chroma_features):
    raise ValueError("graph node is missing `features`")
  if any(len(features) != 12 for features in chroma_features):
    raise ValueError("graph node features must contain 12 chroma values")
  x = torch.tensor(chroma_features, dtype=torch.float)

  # 2. Edges: Construct bidirectionally for undirected graph representation
  edge_index_list = []
  edge_weights_list = []

  for edge in edges:
    src, dst = edge["source"], edge["target"]
    weight = float(edge.get("weight", 1.0))

    # Add forward edge (src -> dst)
    edge_index_list.append([src, dst])
    edge_weights_list.append(weight)

    # Add reverse edge (dst -> src)
    edge_index_list.append([dst, src])
    edge_weights_list.append(weight)

  if edge_index_list:
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights_list, dtype=torch.float)
  else:
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.empty((0,), dtype=torch.float)

  # 3. Create PyG Data object with caption string attribute
  data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
  data.caption = str(caption).strip()
  if not data.caption:
    raise ValueError("graph contains an empty caption")
  data.filename = graph_data.get("filename", "")

  return data


def create_and_save_contrastive_datasets():
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent if script_dir.name == "src" else script_dir

  graphs_dir = project_root / "data" / "processed" / "musiccaps_graphs"
  excluded_txt_path = (
      project_root / "data" / "processed" / "musiccaps_excluded.txt"
  )
  output_dir = project_root / "data" / "processed"

  excluded_files = load_excluded_filenames(excluded_txt_path)
  print(f"Loaded {len(excluded_files)} excluded filenames.")

  # Load graph JSON files
  json_files = sorted(list(graphs_dir.glob("*.json")))
  dataset_list = []
  skipped_count = 0

  for filepath in json_files:
    if filepath.name in excluded_files or filepath.stem in excluded_files:
      skipped_count += 1
      continue

    try:
      with open(filepath, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

      pyg_data = json_to_pyg_data(graph_data)
      dataset_list.append(pyg_data)
    except Exception as e:
      print(f"Error loading {filepath.name}: {e}")

  total_pairs = len(dataset_list)
  print(f"Successfully processed {total_pairs} graph-caption pairs.")
  print(f"Skipped {skipped_count} degenerate files.")

  if total_pairs < 3:
    raise RuntimeError(
        f"Need at least 3 valid graph-caption pairs to create train/val/test splits; found {total_pairs}."
    )

  # Random Split: 80% train, 10% val, 10% test
  train_data, temp_data = train_test_split(
      dataset_list, test_size=0.20, random_state=42, shuffle=True
  )
  val_data, test_data = train_test_split(
      temp_data, test_size=0.50, random_state=42, shuffle=True
  )

  # Save split datasets
  train_path = output_dir / "contrastive_train_dataset.pt"
  val_path = output_dir / "contrastive_val_dataset.pt"
  test_path = output_dir / "contrastive_test_dataset.pt"

  torch.save(train_data, train_path)
  torch.save(val_data, val_path)
  torch.save(test_data, test_path)

  print("\n" + "=" * 50)
  print("CONTRASTIVE DATASET PREPARATION SUMMARY")
  print("=" * 50)
  print(f"Total Usable Pairs : {total_pairs}")
  print(f"Train Dataset Size : {len(train_data)} ({len(train_data)/total_pairs*100:.1f}%)")
  print(f"Val Dataset Size   : {len(val_data)} ({len(val_data)/total_pairs*100:.1f}%)")
  print(f"Test Dataset Size  : {len(test_data)} ({len(test_data)/total_pairs*100:.1f}%)")
  print("-" * 50)
  print(f"Saved Train Dataset: {train_path.resolve()}")
  print(f"Saved Val Dataset  : {val_path.resolve()}")
  print(f"Saved Test Dataset : {test_path.resolve()}")
  print("=" * 50)


if __name__ == "__main__":
  create_and_save_contrastive_datasets()