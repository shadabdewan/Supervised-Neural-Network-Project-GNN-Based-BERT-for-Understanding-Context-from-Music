import json
import logging
from pathlib import Path
import pandas as pd

# Import reusable functions from existing pipeline modules
from audio_features import build_chord_templates, extract_chroma, label_chords
from graph_builder import build_chord_graph, segment_chords, smooth_chord_labels


def main():
  # Define directory and file paths
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent if script_dir.name == "src" else script_dir

  raw_audio_dir = project_root / "data" / "raw" / "musiccaps_audio"
  metadata_csv_path = project_root / "data" / "raw" / "musiccaps_metadata.csv"
  output_dir = project_root / "data" / "processed" / "musiccaps_graphs"
  output_dir.mkdir(parents=True, exist_ok=True)

  log_path = output_dir / "processing_errors.log"
  logger = logging.getLogger("build_musiccaps_graphs")
  logger.setLevel(logging.ERROR)
  logger.handlers.clear()
  file_handler = logging.FileHandler(log_path, encoding="utf-8")
  file_handler.setFormatter(logging.Formatter("%(asctime)s | Filename: %(message)s"))
  logger.addHandler(file_handler)

  # Check metadata availability
  if not metadata_csv_path.exists():
    print(
        f"Error: Metadata CSV not found at {metadata_csv_path}. Please run"
        " download_musiccaps.py first."
    )
    return

  df_metadata = pd.read_csv(metadata_csv_path)
  print(f"Loaded metadata with {len(df_metadata)} entries.")

  # Build reference chord templates
  chord_templates = build_chord_templates()

  # PARAMETER ADJUSTMENT FOR 10-SECOND CLIPS:
  # In GTZAN (30s clips), a median filter window of 9 (~4.5s) and a minimum
  # segment duration of 1.0s were used. For 10-second clips, these settings
  # would over-smooth the sequence, collapsing distinct chord transitions and
  # reducing entire clips to 1 node. Reducing the filter window to 5 (~2.5s)
  # and min segment duration to 0.5s preserves fine-grained transitions while
  # suppressing frame noise.
  MEDIAN_WINDOW = 5
  MIN_SEGMENT_DURATION = 0.5

  processed_count = 0
  skipped_count = 0
  failed_count = 0

  all_node_counts = []
  all_edge_counts = []

  print("\nStarting MusicCaps graph construction...\n" + "-" * 60)

  for idx, row in df_metadata.iterrows():
    filename = str(row["filename"]).strip()
    caption = str(row["caption"]).strip()
    ytid = str(row["ytid"]).strip()

    audio_path = raw_audio_dir / filename
    json_out_path = output_dir / f"{ytid}.json"

    # 1. Resumability Check
    if json_out_path.exists() and json_out_path.stat().st_size > 0:
      skipped_count += 1
      # Read existing graph metadata to include in statistics
      try:
        with open(json_out_path, "r", encoding="utf-8") as f:
          existing_graph = json.load(f)
          all_node_counts.append(len(existing_graph.get("nodes", [])))
          all_edge_counts.append(len(existing_graph.get("edges", [])))
      except Exception:
        pass
      continue

    # Verify input audio exists
    if not audio_path.exists():
      failed_count += 1
      err_msg = f"Audio file not found at {audio_path}"
      logger.error(f"{filename} | {err_msg}")
      print(f"[{idx+1}/{len(df_metadata)}] Skipped (File Missing): {filename}")
      continue

    # 2. Extract Chroma & Build Graph
    try:
      # Step A: Chroma extraction (0.5s windows)
      chroma, times = extract_chroma(
          str(audio_path), window_sec=0.5
      )

      # Step B: Template-based chord classification with tuned smoothing
      raw_chords = label_chords(chroma, chord_templates)
      chords = smooth_chord_labels(raw_chords, window_size=MEDIAN_WINDOW)

      # Step C: Build node and edge structure using tuned minimum duration
      segments = segment_chords(
          chords,
          times,
          chroma,
          min_duration=MIN_SEGMENT_DURATION,
      )
      nodes, edges = build_chord_graph(segments)

      # Step D: Construct JSON with MusicCaps caption instead of genre
      graph_data = {
          "filename": filename,
          "caption": caption,
          "nodes": nodes,
          "edges": edges,
      }

      # Save to JSON
      with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

      num_nodes = len(nodes)
      num_edges = len(edges)

      all_node_counts.append(num_nodes)
      all_edge_counts.append(num_edges)
      processed_count += 1

      print(
          f"[{idx+1}/{len(df_metadata)}] Processed {filename} ->"
          f" Nodes: {num_nodes}, Edges: {num_edges}"
      )

    except Exception as e:
      failed_count += 1
      err_msg = str(e).replace("\n", " ")
      logger.error(f"{filename} | Error: {err_msg}")
      print(f"[{idx+1}/{len(df_metadata)}] Failed {filename}: {err_msg}")

  # 3. Final Summary Statistics
  total_graphs = len(all_node_counts)

  print("\n" + "=" * 60)
  print("MUSICCAPS GRAPH BUILDING SUMMARY")
  print("=" * 60)
  print(f"New Processed  : {processed_count}")
  print(f"Already Existed: {skipped_count}")
  print(f"Failed/Missing : {failed_count}")
  print(f"Total Graphs   : {total_graphs}")

  if total_graphs > 0:
    avg_nodes = sum(all_node_counts) / total_graphs
    min_nodes = min(all_node_counts)
    max_nodes = max(all_node_counts)

    avg_edges = sum(all_edge_counts) / total_graphs
    min_edges = min(all_edge_counts)
    max_edges = max(all_edge_counts)

    print("-" * 60)
    print("Graph Complexity Statistics (10s clips):")
    print(
        f"  Nodes -> Avg: {avg_nodes:.2f} | Min: {min_nodes} | Max: {max_nodes}"
    )
    print(
        f"  Edges -> Avg: {avg_edges:.2f} | Min: {min_edges} | Max: {max_edges}"
    )

  print(f"\nGraphs output saved to: {output_dir.resolve()}")
  print("=" * 60)


if __name__ == "__main__":
  main()