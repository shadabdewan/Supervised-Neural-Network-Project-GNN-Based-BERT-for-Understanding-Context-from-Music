"""
Debug script to test chord detection on a sample of ~20 files across genres.

This script tests the updated smoothing parameters (window=9, min_duration=1.0)
on a representative sample of tracks across all 8 genres to evaluate whether
the parameters are reasonable before running the full 400-file batch.

Process:
1. Select 2-3 files from each of the 8 genres (~20 files total)
2. For each file: extract chroma, label, smooth (with new params), segment, build graph
3. Collect statistics per file
4. Print results table and distribution analysis
5. Provide verdict on parameter reasonableness
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import traceback
import random

from audio_features import extract_chroma, build_chord_templates, label_chords


# ============================================================================
# Configuration
# ============================================================================

DATASET_ROOT = Path(
    r"C:\studymatt\425\gnn-bert-music-context\data\raw\Data\genres_original"
)
GENRES = ["blues", "country", "disco", "hiphop", "metal", "pop", "reggae", "rock"]

# Updated parameters (loosened from previous 15 and 1.5)
MEDIAN_FILTER_WINDOW = 9      # frames (~4.5 seconds at 0.5s per frame)
MIN_SEGMENT_DURATION = 1.0    # seconds

FILES_PER_GENRE = 3  # Sample 3 files per genre for ~24 files total


# ============================================================================
# Smoothing Functions (reused from debug_single_track.py)
# ============================================================================

def smooth_chord_labels_improved(
    chord_labels: List[str],
    window_size: int = MEDIAN_FILTER_WINDOW,
) -> List[str]:
    """Apply median filtering to smooth chord labels."""
    if len(chord_labels) == 0:
        return chord_labels
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(chord_labels)):
        start = max(0, i - half_window)
        end = min(len(chord_labels), i + half_window + 1)
        window_labels = chord_labels[start:end]
        
        label_counts = {}
        for label in window_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        most_common = max(label_counts.keys(), key=lambda k: label_counts[k])
        smoothed.append(most_common)
    
    return smoothed


def segment_chords_improved(
    chord_labels: List[str],
    frame_times: np.ndarray,
    chroma_matrix: np.ndarray,
    min_duration: float = MIN_SEGMENT_DURATION,
) -> List[Dict]:
    """Collapse consecutive identical chords into segments with minimum duration merging."""
    segments = []
    
    if len(chord_labels) == 0:
        return segments
    
    # Step 1: Run-length encoding into initial segments
    current_chord = chord_labels[0]
    segment_start_idx = 0
    
    for i in range(1, len(chord_labels)):
        if chord_labels[i] != current_chord:
            start_time = frame_times[segment_start_idx]
            end_time = frame_times[i - 1]
            duration = end_time - start_time
            avg_chroma = chroma_matrix[:, segment_start_idx:i].mean(axis=1)
            
            segments.append({
                "chord": current_chord,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(duration),
                "avg_chroma": avg_chroma,
            })
            
            current_chord = chord_labels[i]
            segment_start_idx = i
    
    # End final segment
    start_time = frame_times[segment_start_idx]
    end_time = frame_times[-1]
    duration = end_time - start_time
    avg_chroma = chroma_matrix[:, segment_start_idx:].mean(axis=1)
    
    segments.append({
        "chord": current_chord,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration": float(duration),
        "avg_chroma": avg_chroma,
    })
    
    # Step 2: Merge segments shorter than min_duration
    merged = True
    while merged:
        merged = False
        new_segments = []
        i = 0
        
        while i < len(segments):
            seg = segments[i]
            
            if seg["duration"] < min_duration and i < len(segments) - 1:
                prev_duration = segments[i - 1]["duration"] if i > 0 else 0
                next_duration = segments[i + 1]["duration"] if i + 1 < len(segments) else 0
                
                if prev_duration >= next_duration and i > 0:
                    # Merge with previous
                    prev_seg = new_segments.pop()
                    prev_frames = int((prev_seg["end_time"] - prev_seg["start_time"]) / 0.5)
                    curr_frames = int((seg["end_time"] - seg["start_time"]) / 0.5)
                    total_frames = prev_frames + curr_frames
                    
                    if total_frames > 0:
                        combined_chroma = (
                            prev_seg["avg_chroma"] * prev_frames +
                            seg["avg_chroma"] * curr_frames
                        ) / total_frames
                    else:
                        combined_chroma = (prev_seg["avg_chroma"] + seg["avg_chroma"]) / 2
                    
                    merged_seg = {
                        "chord": prev_seg["chord"],
                        "start_time": prev_seg["start_time"],
                        "end_time": seg["end_time"],
                        "duration": seg["end_time"] - prev_seg["start_time"],
                        "avg_chroma": combined_chroma,
                    }
                    new_segments.append(merged_seg)
                    merged = True
                    i += 1
                    
                else:
                    # Merge with next
                    if i + 1 < len(segments):
                        next_seg = segments[i + 1]
                        curr_frames = int((seg["end_time"] - seg["start_time"]) / 0.5)
                        next_frames = int((next_seg["end_time"] - next_seg["start_time"]) / 0.5)
                        total_frames = curr_frames + next_frames
                        
                        if total_frames > 0:
                            combined_chroma = (
                                seg["avg_chroma"] * curr_frames +
                                next_seg["avg_chroma"] * next_frames
                            ) / total_frames
                        else:
                            combined_chroma = (seg["avg_chroma"] + next_seg["avg_chroma"]) / 2
                        
                        merged_seg = {
                            "chord": next_seg["chord"],
                            "start_time": seg["start_time"],
                            "end_time": next_seg["end_time"],
                            "duration": next_seg["end_time"] - seg["start_time"],
                            "avg_chroma": combined_chroma,
                        }
                        new_segments.append(merged_seg)
                        merged = True
                        i += 2
                        
                    else:
                        new_segments.append(seg)
                        i += 1
            else:
                new_segments.append(seg)
                i += 1
        
        segments = new_segments
    
    return segments


def build_chord_graph_improved(segments: List[Dict]) -> Tuple[List[Dict], List[Dict], List[int]]:
    """Build graph and return nodes, edges, and edge weights list."""
    if len(segments) == 0:
        return [], [], []
    
    # Collect unique chords and average chroma
    chord_to_chromas = {}
    for segment in segments:
        chord = segment["chord"]
        if chord not in chord_to_chromas:
            chord_to_chromas[chord] = []
        chord_to_chromas[chord].append(segment["avg_chroma"])
    
    # Create nodes
    chord_to_node_id = {}
    nodes = []
    for node_id, (chord, chromas) in enumerate(sorted(chord_to_chromas.items())):
        avg_chroma = np.mean(chromas, axis=0)
        nodes.append({
            "node_id": node_id,
            "chord": chord,
            "features": avg_chroma.tolist(),
        })
        chord_to_node_id[chord] = node_id
    
    # Create edges
    edge_dict = {}
    for i in range(len(segments) - 1):
        current_chord = segments[i]["chord"]
        next_chord = segments[i + 1]["chord"]
        
        source_id = chord_to_node_id[current_chord]
        target_id = chord_to_node_id[next_chord]
        
        if source_id == target_id:
            continue
        
        edge_key = (source_id, target_id)
        edge_dict[edge_key] = edge_dict.get(edge_key, 0) + 1
    
    edges = [
        {"source": src, "target": tgt, "weight": weight}
        for (src, tgt), weight in sorted(edge_dict.items())
    ]
    
    edge_weights = [weight for (_, _, weight) in sorted(
        [(src, tgt, weight) for (src, tgt), weight in edge_dict.items()]
    )]
    
    return nodes, edges, edge_weights


# ============================================================================
# Sample Selection and Processing
# ============================================================================

def get_sample_files() -> List[Tuple[str, str]]:
    """
    Select a random sample of files across genres.
    
    Returns:
        List of (genre, filepath) tuples
    """
    sample_files = []
    
    for genre in GENRES:
        genre_dir = DATASET_ROOT / genre
        if not genre_dir.exists():
            continue
        
        # Get all wav files and randomly select FILES_PER_GENRE
        wav_files = list(genre_dir.glob("*.wav"))
        
        # Randomly select up to FILES_PER_GENRE files
        selected_files = random.sample(wav_files, min(FILES_PER_GENRE, len(wav_files)))
        
        for wav_file in sorted(selected_files):
            sample_files.append((genre, wav_file))
    
    return sample_files


def process_file(
    audio_path: Path,
    genre: str,
    templates: Dict,
) -> Dict:
    """
    Process a single file through the full pipeline.
    
    Returns:
        Dict with filename, genre, node_count, edge_count, edge_weights
    """
    chroma_matrix, frame_times = extract_chroma(str(audio_path), sr=22050, window_sec=0.5)
    raw_labels = label_chords(chroma_matrix, templates)
    smoothed_labels = smooth_chord_labels_improved(raw_labels, window_size=MEDIAN_FILTER_WINDOW)
    segments = segment_chords_improved(smoothed_labels, frame_times, chroma_matrix, min_duration=MIN_SEGMENT_DURATION)
    nodes, edges, edge_weights = build_chord_graph_improved(segments)
    
    return {
        "filename": audio_path.name,
        "genre": genre,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "edge_weights": edge_weights,
        "max_edge_weight": max(edge_weights) if edge_weights else 0,
    }


# ============================================================================
# Main Script
# ============================================================================

def main():
    print("="*90)
    print("DEBUG: Sample Track Chord Detection Test")
    print("="*90)
    print(f"\nParameters:")
    print(f"  Median filter window: {MEDIAN_FILTER_WINDOW} frames (~{MEDIAN_FILTER_WINDOW * 0.5:.1f} seconds)")
    print(f"  Minimum segment duration: {MIN_SEGMENT_DURATION} seconds")
    print(f"  Sample size: {FILES_PER_GENRE} files per genre × {len(GENRES)} genres\n")
    
    # Get sample files
    sample_files = get_sample_files()
    print(f"Selected {len(sample_files)} files for testing\n")
    
    # Build templates once
    templates = build_chord_templates()
    
    # Process each file
    results = []
    failures = []
    
    print("="*90)
    print("Processing files...")
    print("="*90)
    
    for genre, audio_path in sample_files:
        try:
            result = process_file(audio_path, genre, templates)
            results.append(result)
            print(f"✓ {audio_path.name:30} ({genre:10}) -> {result['node_count']} nodes, {result['edge_count']} edges")
        except Exception as e:
            failures.append((audio_path.name, str(e)))
            print(f"✗ {audio_path.name:30} - ERROR: {str(e)[:60]}")
    
    # Print results table
    print("\n" + "="*90)
    print("RESULTS TABLE")
    print("="*90)
    print(f"{'Filename':<30} {'Genre':<12} {'Nodes':<8} {'Edges':<8} {'Max Weight':<12}")
    print("-"*90)
    for res in results:
        print(
            f"{res['filename']:<30} {res['genre']:<12} {res['node_count']:<8} "
            f"{res['edge_count']:<8} {res['max_edge_weight']:<12}"
        )
    
    # Distribution statistics
    print("\n" + "="*90)
    print("DISTRIBUTION STATISTICS")
    print("="*90)
    
    if not results:
        print("No files processed successfully!")
        return
    
    node_counts = [r["node_count"] for r in results]
    edge_counts = [r["edge_count"] for r in results]
    all_weights = []
    for r in results:
        all_weights.extend(r["edge_weights"])
    
    # Node count statistics
    print(f"\nNode count (unique chords per track):")
    print(f"  Average: {np.mean(node_counts):.2f}")
    print(f"  Median: {np.median(node_counts):.0f}")
    print(f"  Min: {np.min(node_counts)}")
    print(f"  Max: {np.max(node_counts)}")
    
    # Edge count statistics
    print(f"\nEdge count (transitions per track):")
    print(f"  Average: {np.mean(edge_counts):.2f}")
    print(f"  Min: {np.min(edge_counts)}")
    print(f"  Max: {np.max(edge_counts)}")
    
    # Edge weight distribution
    print(f"\nEdge weight distribution (across all {len(all_weights)} edges):")
    if all_weights:
        weight_counts = {}
        for w in all_weights:
            weight_counts[w] = weight_counts.get(w, 0) + 1
        for weight in sorted(weight_counts.keys()):
            count = weight_counts[weight]
            pct = 100 * count / len(all_weights)
            print(f"  Weight {weight}: {count} edges ({pct:.1f}%)")
    
    # Target range analysis
    print(f"\nTarget range analysis (expected 3-8 nodes):")
    in_range = sum(1 for c in node_counts if 3 <= c <= 8)
    below_range = sum(1 for c in node_counts if c < 3)
    above_range = sum(1 for c in node_counts if c > 8)
    
    print(f"  In target range (3-8): {in_range}/{len(node_counts)} ({100*in_range/len(node_counts):.1f}%)")
    print(f"  Below range (<3): {below_range} files")
    print(f"  Above range (>8): {above_range} files")
    
    # Final verdict
    print("\n" + "="*90)
    print("VERDICT")
    print("="*90)
    
    success_rate = in_range / len(node_counts) if node_counts else 0
    
    if success_rate >= 0.70:
        print(f"\n✓ PARAMETERS LOOK REASONABLE")
        print(f"  {in_range} out of {len(node_counts)} files ({100*success_rate:.0f}%) fall within the target 3-8 node range.")
        print(f"  A few outliers are expected and acceptable.")
        print(f"\n  RECOMMENDATION: Proceed to full batch processing with these parameters.")
    else:
        print(f"\n✗ NEEDS FURTHER TUNING")
        print(f"  Only {in_range} out of {len(node_counts)} files ({100*success_rate:.0f}%) fall within the target 3-8 node range.")
        print(f"  More than 30% of files are outside the target range.\n")
        
        if above_range > below_range:
            print(f"  Analysis: {above_range} files have TOO MANY nodes (>8) - chord flickering still too high")
            print(f"  RECOMMENDATION: Increase median filter window (e.g., 11 or 13) to smooth more aggressively")
        elif below_range > above_range:
            print(f"  Analysis: {below_range} files have TOO FEW nodes (<3) - over-smoothing/merging")
            print(f"  RECOMMENDATION: Decrease median filter window (e.g., 7 or 5) or increase min segment duration")
        else:
            print(f"  Analysis: Mixed failures - both over-smoothing and under-smoothing observed")
            print(f"  RECOMMENDATION: Check individual outlier files to understand the pattern")
    
    # Summary
    print(f"\n{'='*90}")
    print(f"Summary: Processed {len(results)}/{len(sample_files)} files successfully")
    if failures:
        print(f"Failed files ({len(failures)}):")
        for fname, err in failures:
            print(f"  - {fname}: {err[:50]}")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
