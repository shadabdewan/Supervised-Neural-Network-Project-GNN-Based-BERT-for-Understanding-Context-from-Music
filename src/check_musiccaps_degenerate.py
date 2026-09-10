import json
from pathlib import Path


def main():
  # Define paths using pathlib
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent if script_dir.name == "src" else script_dir

  graphs_dir = project_root / "data" / "processed" / "musiccaps_graphs"
  excluded_txt_path = (
      project_root / "data" / "processed" / "musiccaps_excluded.txt"
  )

  # Check input directory
  if not graphs_dir.exists():
    print(f"Error: Directory not found at {graphs_dir}")
    return

  json_files = sorted(list(graphs_dir.glob("*.json")))
  total_files = len(json_files)

  if total_files == 0:
    print(f"No JSON files found in {graphs_dir}")
    return

  print(f"Scanning {total_files} MusicCaps graph files in {graphs_dir}...\n")

  degenerate_files = []
  usable_count = 0

  # Scan each graph file
  for filepath in json_files:
    try:
      with open(filepath, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

      filename = graph_data.get("filename", filepath.name)
      nodes = graph_data.get("nodes", [])
      edges = graph_data.get("edges", [])

      node_count = len(nodes)
      edge_count = len(edges)

      # Flag as degenerate if fewer than 2 nodes OR 0 edges
      if node_count < 2 or edge_count < 1:
        degenerate_files.append({
            "filepath_name": filepath.name,
            "filename": filename,
            "node_count": node_count,
            "edge_count": edge_count,
        })
      else:
        usable_count += 1

    except Exception as e:
      print(f"Warning: Failed to parse {filepath.name}: {e}")
      degenerate_files.append({
          "filepath_name": filepath.name,
          "filename": filepath.name,
          "node_count": 0,
          "edge_count": 0,
      })

  degenerate_count = len(degenerate_files)
  degenerate_pct = (
      (degenerate_count / total_files) * 100 if total_files > 0 else 0.0
  )

  # Save degenerate list to text file
  with open(excluded_txt_path, "w", encoding="utf-8") as f:
    f.write(
        "# MusicCaps Degenerate Graphs Exclusion List\n"
        "# Criteria: node_count < 2 OR edge_count < 1\n"
        f"# Total Scanned: {total_files} | Excluded: {degenerate_count}\n"
    )
    for item in degenerate_files:
      f.write(f"{item['filepath_name']}\n")

  # Print detailed report
  print("=" * 60)
  print("MUSICCAPS DEGENERATE GRAPH ANALYSIS REPORT")
  print("=" * 60)
  print(f"Total Graphs Scanned : {total_files}")
  print(f"Degenerate Graphs    : {degenerate_count} ({degenerate_pct:.2f}%)")
  print(f"Usable Task 4 Pairs  : {usable_count}")
  print("-" * 60)

  if degenerate_count > 0:
    print("Degenerate Files List:")
    for d in degenerate_files:
      print(
          f"  - {d['filepath_name']:<25} | Nodes: {d['node_count']} | Edges:"
          f" {d['edge_count']}"
      )
  else:
    print("No degenerate graphs found!")

  print("-" * 60)
  print(f"Excluded list saved to: {excluded_txt_path.resolve()}")
  print(
      f"Final Task 4 Dataset Size: {usable_count} (graph, caption)"
      " pairs ready for training."
  )
  print("=" * 60)


if __name__ == "__main__":
  main()