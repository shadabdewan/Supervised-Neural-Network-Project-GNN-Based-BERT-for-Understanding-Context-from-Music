import csv
import json
from pathlib import Path
import torch
from torch_geometric.data import Data


def load_excluded_tracks(excluded_file: Path) -> set:
  """Loads the set of excluded filenames from excluded_tracks.txt."""
  excluded = set()
  if excluded_file.exists():
    with open(excluded_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
          excluded.add(line)
          excluded.add(Path(line).stem)
  return excluded


def load_transcriptions(transcriptions_file: Path) -> dict:
  """Loads transcriptions from CSV and filters for tracks with 5+ words."""
  valid_lyrics = {}
  if transcriptions_file.exists():
    with open(transcriptions_file, "r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        filename = row["filename"]
        text = row["transcription"].strip()
        word_count = len(text.split())
        if word_count >= 5:
          valid_lyrics[filename] = text
          valid_lyrics[Path(filename).stem] = text
  return valid_lyrics


def create_fusion_dataset():
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  data_processed_dir = project_root / "data" / "processed"
  excluded_file = data_processed_dir / "excluded_tracks.txt"
  transcriptions_file = data_processed_dir / "lyrics_transcriptions.csv"
  chord_graphs_dir = data_processed_dir / "chord_graphs"
  genre2id_file = data_processed_dir / "genre2id.json"

  # 1. Load genre mapping
  with open(genre2id_file, "r", encoding="utf-8") as f:
    genre2id = json.load(f)

  # 2. Intersect valid tracks (635 tracks total)
  excluded_set = load_excluded_tracks(excluded_file)
  valid_lyrics = load_transcriptions(transcriptions_file)

  graph_files = sorted(list(chord_graphs_dir.glob("*.json")))
  fusion_data_list = []

  print("Processing tracks for Task 3 Fusion Dataset...")
  for graph_file in graph_files:
    track_stem = graph_file.stem

    # Filter conditions: Not excluded AND has >= 5 words in lyrics
    if track_stem in excluded_set or (track_stem + ".wav") in excluded_set:
      continue

    if track_stem not in valid_lyrics and (track_stem + ".wav") not in valid_lyrics:
      continue

    lyrics_text = valid_lyrics.get(
        track_stem, valid_lyrics.get(track_stem + ".wav", "")
    )

    # Read chord graph JSON
    with open(graph_file, "r", encoding="utf-8") as f:
      graph_json = json.load(f)

    raw_nodes = graph_json.get("nodes", [])
    raw_edges = graph_json.get("edges", [])
    genre_str = graph_json.get("genre", "").lower()

    if genre_str not in genre2id:
      continue

    if not raw_nodes:
      continue

    node_id_map = {}
    node_features = []
    for idx, node in enumerate(raw_nodes):
      if isinstance(node, dict):
        orig_id = node.get("node_id", node.get("id", node.get("name", idx)))
        features = node.get("features", node.get("chroma", []))
      else:
        orig_id = idx
        features = []

      node_id_map[orig_id] = len(node_id_map)
      node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float32)

    src_nodes = []
    dst_nodes = []
    weights = []
    for edge in raw_edges:
      if not isinstance(edge, dict):
        continue

      u_orig = edge.get("source", edge.get("src"))
      v_orig = edge.get("target", edge.get("dst"))
      w = float(edge.get("weight", 1.0))

      if u_orig not in node_id_map or v_orig not in node_id_map:
        continue

      u = node_id_map[u_orig]
      v = node_id_map[v_orig]

      src_nodes.extend([u, v])
      dst_nodes.extend([v, u])
      weights.extend([w, w])

    if not src_nodes:
      continue

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    y = torch.tensor([genre2id[genre_str]], dtype=torch.long)

    pyg_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    pyg_data.lyrics = lyrics_text
    pyg_data.filename = f"{track_stem}.wav"
    pyg_data.genre_str = genre_str

    fusion_data_list.append(pyg_data)

  print(f"Total fusion-ready samples loaded: {len(fusion_data_list)}")

  # 3. Stratified Split 80/10/10 by genre
  labels = [d.y.item() for d in fusion_data_list]
  from sklearn.model_selection import train_test_split

  train_data, temp_data, train_labels, temp_labels = train_test_split(
      fusion_data_list, labels, test_size=0.20, stratify=labels, random_state=42
  )

  val_data, test_data, _, _ = train_test_split(
      temp_data,
      temp_labels,
      test_size=0.50,
      stratify=temp_labels,
      random_state=42,
  )

  # Save datasets
  train_path = data_processed_dir / "fusion_train_dataset.pt"
  val_path = data_processed_dir / "fusion_val_dataset.pt"
  test_path = data_processed_dir / "fusion_test_dataset.pt"

  torch.save(train_data, train_path)
  torch.save(val_data, val_path)
  torch.save(test_data, test_path)

  # 4. Print Summary Breakdown
  print("\n" + "=" * 60)
  print("FUSION DATASET CREATION SUMMARY")
  print("=" * 60)
  print(f"Train split count : {len(train_data)}")
  print(f"Val split count   : {len(val_data)}")
  print(f"Test split count  : {len(test_data)}")

  id2genre = {v: k for k, v in genre2id.items()}

  print("\nPer-Genre Distribution across splits:")
  print(f"{'Genre':<12} | {'Train':<6} | {'Val':<5} | {'Test':<5}")
  print("-" * 35)
  for genre_id in sorted(id2genre.keys()):
    genre_name = id2genre[genre_id]
    tr_c = sum(1 for d in train_data if d.y.item() == genre_id)
    va_c = sum(1 for d in val_data if d.y.item() == genre_id)
    te_c = sum(1 for d in test_data if d.y.item() == genre_id)
    print(f"{genre_name:<12} | {tr_c:<6} | {va_c:<5} | {te_c:<5}")
  print("=" * 60)


if __name__ == "__main__":
  create_fusion_dataset()