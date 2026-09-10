import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# 0. Genre to Integer Mapping (Alphabetical)
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


def load_and_preprocess_dataset():
    # Setup base directory paths
    base_dir = Path(r"C:\studymatt\425\gnn-bert-music-context\data\processed")
    chord_graphs_dir = base_dir / "chord_graphs"
    excluded_tracks_path = base_dir / "excluded_tracks.txt"

    # 1. Load excluded filenames into a set
    excluded_files = set()
    if excluded_tracks_path.exists():
        with open(excluded_tracks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    excluded_files.add(line)
                    # Also exclude without extension if present
                    excluded_files.add(Path(line).stem)
        print(f"Loaded {len(excluded_files)} excluded track entries.")
    else:
        print(f"Warning: {excluded_tracks_path} not found. Processing all files.")

    # 2 & 3. Load valid JSON files and convert to PyG Data objects
    data_list = []

    for json_file in chord_graphs_dir.glob("*.json"):
        # Check against exclusion list
        if (
            json_file.name in excluded_files
            or json_file.stem in excluded_files
        ):
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        raw_nodes = raw_data.get("nodes", [])
        raw_edges = raw_data.get("edges", [])
        genre_str = raw_data.get("genre", "").lower()

        if genre_str not in GENRE_MAP:
            print(f"Skipping {json_file.name}: Unknown genre '{genre_str}'")
            continue

        if not raw_nodes:
            continue

        # Safely parse node items
        # Creates a node mapping to handle arbitrary string/non-contiguous IDs cleanly -> [0, 1, ..., N-1]
        node_id_map = {}
        node_features = []

        for idx, n in enumerate(raw_nodes):
            if isinstance(n, dict):
                # Retrieve original node ID, fallback to list index if missing
                orig_id = n.get("id", n.get("name", idx))
                features = n.get("features", n.get("chroma", []))
            else:
                orig_id = idx
                features = []

            node_id_map[orig_id] = len(node_id_map)
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Build edge index and edge weights using 0-indexed mapped IDs
        src_nodes = []
        dst_nodes = []
        weights = []

        for edge in raw_edges:
            if isinstance(edge, dict):
                u_orig = edge.get("source", edge.get("src"))
                v_orig = edge.get("target", edge.get("dst"))
                w = float(edge.get("weight", 1.0))
            else:
                continue

            # Skip edges pointing to invalid or unmapped nodes
            if u_orig not in node_id_map or v_orig not in node_id_map:
                continue

            u = node_id_map[u_orig]
            v = node_id_map[v_orig]

            # Undirected message passing: store both u->v and v->u
            src_nodes.append(u)
            dst_nodes.append(v)
            weights.append(w)

            src_nodes.append(v)
            dst_nodes.append(u)
            weights.append(w)

        # Skip if graph lost all valid edges during mapping
        if not src_nodes:
            continue

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        y = torch.tensor([GENRE_MAP[genre_str]], dtype=torch.long)

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            track_name=json_file.stem,
        )
        data_list.append(pyg_data)

    print(f"Successfully loaded and converted {len(data_list)} tracks.")

    # 4. Perform Stratified Train / Val / Test Split (80% Train, 10% Val, 10% Test)
    targets = [data.y.item() for data in data_list]

    train_data, temp_data, train_y, temp_y = train_test_split(
        data_list,
        targets,
        test_size=0.20,
        stratify=targets,
        random_state=42,
    )

    val_data, test_data, _, _ = train_test_split(
        temp_data,
        temp_y,
        test_size=0.50,
        stratify=temp_y,
        random_state=42,
    )

    # 5. Save dataset partitions to disk
    train_path = base_dir / "train_dataset.pt"
    val_path = base_dir / "val_dataset.pt"
    test_path = base_dir / "test_dataset.pt"

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)

    print(f"\nDatasets saved to {base_dir}:")
    print(f"  - {train_path.name}")
    print(f"  - {val_path.name}")
    print(f"  - {test_path.name}")

    # 6. Print Summary and Stratification Breakdown Report
    print("\n" + "=" * 65)
    print("DATASET SPLIT & STRATIFICATION SUMMARY")
    print("=" * 65)
    print(f"Total Usable Tracks: {len(data_list)}")
    print(
        f"Train Set: {len(train_data)} | Val Set: {len(val_data)} | Test Set: {len(test_data)}"
    )
    print("-" * 65)

    idx_to_genre = {v: k for k, v in GENRE_MAP.items()}
    train_counts = Counter([d.y.item() for d in train_data])
    val_counts = Counter([d.y.item() for d in val_data])
    test_counts = Counter([d.y.item() for d in test_data])

    print(
        f"{'Genre':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    )
    print("-" * 65)
    for idx in sorted(GENRE_MAP.values()):
        genre_name = idx_to_genre[idx]
        tr_c = train_counts.get(idx, 0)
        va_c = val_counts.get(idx, 0)
        te_c = test_counts.get(idx, 0)
        tot = tr_c + va_c + te_c
        print(
            f"{genre_name:<12} | {tr_c:<8} | {va_c:<8} | {te_c:<8} | {tot:<8}"
        )
    print("=" * 65)


if __name__ == "__main__":
    load_and_preprocess_dataset()