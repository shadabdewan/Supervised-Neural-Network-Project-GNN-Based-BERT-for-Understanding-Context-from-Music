import json
from pathlib import Path
import shutil


def check_and_isolate_degenerate_graphs():
    # 1. Setup path references
    dataset_dir = Path(
        r"C:\studymatt\425\gnn-bert-music-context\data\processed\chord_graphs"
    )
    parent_dir = dataset_dir.parent
    excluded_dir = parent_dir / "chord_graphs_excluded"
    excluded_tracks_path = parent_dir / "excluded_tracks.txt"

    if not dataset_dir.exists():
        print(f"Error: Directory not found: {dataset_dir}")
        return

    # Ensure output directory for isolated files exists
    excluded_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(dataset_dir.glob("*.json"))
    total_files = len(json_files)

    critical_files = []
    warning_files = []

    # 2. Scan and record file statistics
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            genre = data.get("genre", "Unknown")

            node_count = len(nodes)
            edge_count = len(edges)

            file_info = {
                "file_path": file_path,
                "filename": file_path.name,
                "genre": genre,
                "node_count": node_count,
                "edge_count": edge_count,
                "raw_data": data,
            }

            # Threshold: node_count < 3 OR edge_count < 3
            if node_count < 3 or edge_count < 3:
                critical_files.append(file_info)

            # Secondary warning flag: 2 nodes, 0 edges
            if node_count == 2 and edge_count == 0:
                warning_files.append(file_info)

        except Exception as e:
            print(f"Error reading file {file_path.name}: {e}")

    # 3. Print Summary Report
    print("=" * 65)
    print("DEGENERATE GRAPH INSPECTION & ISOLATION REPORT")
    print("=" * 65)
    print(f"Total active files scanned: {total_files}\n")

    # Critical report
    num_critical = len(critical_files)
    critical_pct = (num_critical / total_files * 100) if total_files > 0 else 0
    print(
        f"CRITICAL GRAPHS (node_count < 3 OR edge_count < 3): {num_critical} ({critical_pct:.2f}%)"
    )
    if num_critical > 0:
        for item in critical_files:
            print(
                f"  - File: {item['filename']} | Genre: {item['genre']} | Nodes: {item['node_count']} | Edges: {item['edge_count']}"
            )
    else:
        print("  None found in active directory.")
    print("-" * 65)

    # Warning report
    num_warning = len(warning_files)
    warning_pct = (num_warning / total_files * 100) if total_files > 0 else 0
    print(f"WARNING GRAPHS (2 nodes, 0 edges): {num_warning} ({warning_pct:.2f}%)")
    if num_warning > 0:
        for item in warning_files:
            print(
                f"  - File: {item['filename']} | Genre: {item['genre']} | Nodes: {item['node_count']} | Edges: {item['edge_count']}"
            )
    else:
        print("  None found.")
    print("-" * 65)

    # 4. Print raw contents for CRITICAL files
    if num_critical > 0:
        print("\nRAW CONTENTS OF CRITICAL FILES FOR INSPECTION:")
        print("=" * 65)
        for item in critical_files:
            print(f"\n--- {item['filename']} ---")
            print(json.dumps(item["raw_data"], indent=2))
        print("=" * 65)

    # 5. Safely MOVE CRITICAL files to chord_graphs_excluded/
    moved_count = 0
    for item in critical_files:
        src_path = item["file_path"]
        dest_path = excluded_dir / item["filename"]

        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dest_path))
                moved_count += 1
            except Exception as e:
                print(f"Error moving {item['filename']}: {e}")

    # 6. Synchronize excluded_tracks.txt with all files currently in chord_graphs_excluded/
    all_excluded_in_folder = sorted(
        [f.name for f in excluded_dir.glob("*.json")]
    )
    try:
        with open(excluded_tracks_path, "w", encoding="utf-8") as out_f:
            out_f.write(
                "# The following tracks were excluded due to insufficient graph structure\n"
            )
            out_f.write(
                "# (node_count < 3 or edge_count < 3) making them unusable for GNN message passing.\n"
            )
            for filename in all_excluded_in_folder:
                out_f.write(f"{filename}\n")
    except Exception as e:
        print(f"Error updating {excluded_tracks_path}: {e}")

    # 7. Final Status Confirmation
    remaining_files = len(list(dataset_dir.glob("*.json")))

    print("\n" + "=" * 65)
    print("FINAL ISOLATION SUMMARY")
    print("=" * 65)
    print(f"Newly moved files this run : {moved_count}")
    print(f"Total files in excluded dir: {len(all_excluded_in_folder)}")
    print(f"Remaining active files      : {remaining_files}")
    print(f"Excluded files location     : {excluded_dir.resolve()}")
    print(f"Sync file updated          : {excluded_tracks_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    check_and_isolate_degenerate_graphs()