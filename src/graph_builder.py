"""
Chord-transition graph builder for GTZAN dataset.

This script:
1. Extracts chroma features from audio files (via audio_features.py)
2. Labels frames with chord names
3. Smooths and segments chord sequences (Stage C)
4. Builds chord-transition graphs (Stage D)
5. Saves graph JSON files (resumable, error-tolerant)
6. Prints batch processing statistics

Run with: python graph_builder.py
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

from audio_features import extract_chroma, build_chord_templates, label_chords


# ============================================================================
# Configuration
# ============================================================================

DATASET_ROOT = Path(r"C:\studymatt\425\gnn-bert-music-context\data\raw\Data\genres_original")
OUTPUT_DIR = Path(r"C:\studymatt\425\gnn-bert-music-context\data\processed\chord_graphs")
FAILED_LOG = OUTPUT_DIR / "failed_files.log"

GENRES = ["blues", "country", "disco", "hiphop", "metal", "pop", "reggae", "rock"]
FILES_PER_GENRE = 70  # Process first 70 files per genre (files 00000-00069)


# ============================================================================
# Stage C: Smoothing and Run-Length Encoding
# ============================================================================

def smooth_chord_labels(chord_labels: List[str], window_size: int = 9) -> List[str]:
    """
    Apply median filtering to smooth chord label sequence.
    
    Removes single-frame flickering by applying majority voting
    in a sliding window. For non-string data we'd use scipy.signal.medfilt,
    but for chord labels we use manual majority voting.
    
    Parameters
    ----------
    chord_labels : list of str
        Original chord labels, one per frame.
    window_size : int
        Sliding window size for majority voting (default: 9).
    
    Returns
    -------
    smoothed : list of str
        Chord labels after median filtering.
    """
    if len(chord_labels) == 0:
        return chord_labels
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(chord_labels)):
        # Window bounds
        start = max(0, i - half_window)
        end = min(len(chord_labels), i + half_window + 1)
        
        # Get labels in window
        window_labels = chord_labels[start:end]
        
        # Find most common label
        label_counts = {}
        for label in window_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        most_common = max(label_counts.keys(), key=lambda k: label_counts[k])
        smoothed.append(most_common)
    
    return smoothed


def segment_chords(
    chord_labels: List[str],
    frame_times: np.ndarray,
    chroma_matrix: np.ndarray,
    min_duration: float = 1.0,
) -> List[Dict]:
    """
    Collapse consecutive identical chord labels into segments, with minimum duration merging.
    
    Each segment records: chord name, start_time, end_time, duration,
    and average chroma vector across all frames in that segment.
    After initial run-length encoding, merges any segment shorter than min_duration
    into whichever neighboring segment has longer duration.
    
    Parameters
    ----------
    chord_labels : list of str
        Smoothed chord labels, one per frame.
    frame_times : np.ndarray
        Time in seconds for each frame (shape: n_frames).
    chroma_matrix : np.ndarray
        Chroma features (shape: 12, n_frames).
    min_duration : float
        Minimum allowed segment duration in seconds (default: 1.0).
    
    Returns
    -------
    segments : list of dict
        Each dict: {"chord": str, "start_time": float, "end_time": float,
                     "duration": float, "avg_chroma": np.ndarray (shape: 12)}
    """
    segments = []
    
    if len(chord_labels) == 0:
        return segments
    
    # Start first segment
    current_chord = chord_labels[0]
    segment_start_idx = 0
    
    for i in range(1, len(chord_labels)):
        # Check if chord changed
        if chord_labels[i] != current_chord:
            # End current segment
            start_time = frame_times[segment_start_idx]
            end_time = frame_times[i - 1]
            duration = end_time - start_time
            
            # Average chroma across segment
            avg_chroma = chroma_matrix[:, segment_start_idx:i].mean(axis=1)
            
            segments.append({
                "chord": current_chord,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(duration),
                "avg_chroma": avg_chroma,
            })
            
            # Start new segment
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
    
    # Merge segments shorter than min_duration
    merged = True
    while merged:
        merged = False
        new_segments = []
        i = 0
        
        while i < len(segments):
            seg = segments[i]
            
            if seg["duration"] < min_duration and i < len(segments) - 1:
                # Merge with neighbor - pick the one with longer duration
                prev_duration = segments[i - 1]["duration"] if i > 0 else 0
                next_duration = segments[i + 1]["duration"] if i + 1 < len(segments) else 0
                
                if prev_duration >= next_duration and i > 0:
                    # Merge with previous segment
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
                    # Merge with next segment
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


# ============================================================================
# Stage D: Graph Construction
# ============================================================================

def build_chord_graph(segments: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Build a graph from chord segments.
    
    Nodes: one per unique chord in the track. Each node stores its chord name
    and average chroma features computed across ALL occurrences of that chord.
    
    Edges: between consecutive segments. If the same chord pair (u->v) appears
    multiple times non-consecutively, increment the weight instead of creating
    duplicate edges. Skip self-loops (shouldn't happen after smoothing).
    
    Parameters
    ----------
    segments : list of dict
        From segment_chords(), each with "chord" and "avg_chroma" keys.
    
    Returns
    -------
    nodes : list of dict
        Each: {"node_id": int, "chord": str, "features": list (12 values)}
    edges : list of dict
        Each: {"source": int, "target": int, "weight": int}
    """
    if len(segments) == 0:
        return [], []
    
    # Collect all unique chords and their average chroma features
    chord_to_chromas = {}  # chord_name -> list of avg_chroma vectors
    for segment in segments:
        chord = segment["chord"]
        if chord not in chord_to_chromas:
            chord_to_chromas[chord] = []
        chord_to_chromas[chord].append(segment["avg_chroma"])
    
    # Create nodes with overall average chroma per chord
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
    
    # Create edges from consecutive segments
    edge_dict = {}  # (source_id, target_id) -> weight
    for i in range(len(segments) - 1):
        current_chord = segments[i]["chord"]
        next_chord = segments[i + 1]["chord"]
        
        source_id = chord_to_node_id[current_chord]
        target_id = chord_to_node_id[next_chord]
        
        # Skip self-loops
        if source_id == target_id:
            continue
        
        # Increment or create edge
        edge_key = (source_id, target_id)
        edge_dict[edge_key] = edge_dict.get(edge_key, 0) + 1
    
    # Convert to list of dicts
    edges = [
        {"source": src, "target": tgt, "weight": weight}
        for (src, tgt), weight in sorted(edge_dict.items())
    ]
    
    return nodes, edges


# ============================================================================
# Batch Processing
# ============================================================================

def setup_output_directory():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_audio_file(audio_path: Path, genre: str, track_number: int) -> Dict:
    """
    Process a single audio file through Stages C and D.
    
    Parameters
    ----------
    audio_path : Path
        Path to the .wav file
    genre : str
        Genre name (for output JSON)
    track_number : int
        Track number (for output JSON)
    
    Returns
    -------
    result : dict
        Graph data ready for JSON serialization.
    
    Raises
    ------
    Exception
        If any step fails (audio loading, feature extraction, etc.)
    """
    # Stage A: Extract chroma features
    chroma_matrix, frame_times = extract_chroma(str(audio_path))
    
    # Stage B: Label chords
    templates = build_chord_templates()
    chord_labels = label_chords(chroma_matrix, templates)
    
    # Stage C: Smooth and segment
    smoothed_labels = smooth_chord_labels(chord_labels, window_size=9)
    segments = segment_chords(smoothed_labels, frame_times, chroma_matrix, min_duration=1.0)
    
    # Stage D: Build graph
    nodes, edges = build_chord_graph(segments)
    
    # Format output
    result = {
        "filename": audio_path.name,
        "genre": genre,
        "nodes": nodes,
        "edges": edges,
    }
    
    return result


def run_batch_processing():
    """
    Main batch processing loop.
    
    Processes first 50 files per genre (400 total), with resumability and
    error handling. Prints per-file summaries and overall statistics.
    """
    setup_output_directory()
    
    # Initialize logging
    logging.basicConfig(
        filename=str(FAILED_LOG),
        level=logging.ERROR,
        format="%(message)s",
        filemode="w",  # Overwrite log at start
    )
    
    # Track statistics
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    all_node_counts = []
    all_edge_counts = []
    failed_files = []
    
    print("="*80)
    print("Chord-Transition Graph Builder - Batch Processing")
    print("="*80)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing first {FILES_PER_GENRE} files per genre ({len(GENRES)} genres)\n")
    
    # Loop through genres and files
    for genre in GENRES:
        genre_dir = DATASET_ROOT / genre
        
        if not genre_dir.exists():
            print(f"[ERROR] Genre directory not found: {genre_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing genre: {genre}")
        print(f"{'='*80}")
        
        # Get all .wav files, sorted
        wav_files = sorted(genre_dir.glob("*.wav"))
        
        # Process only first FILES_PER_GENRE files
        wav_files = wav_files[:FILES_PER_GENRE]
        
        genre_processed = 0
        genre_skipped = 0
        genre_failed = 0
        
        for track_idx, audio_path in enumerate(wav_files):
            # Extract track number from filename (e.g., "blues.00042.wav" -> 42)
            track_number = int(audio_path.stem.split(".")[-1])
            
            # Output filename
            output_filename = f"{genre}.{track_number:05d}.json"
            output_path = OUTPUT_DIR / output_filename
            
            # Check if already processed (resumability)
            if output_path.exists():
                print(f"[SKIP] {audio_path.name} - already processed")
                total_skipped += 1
                genre_skipped += 1
                continue
            
            # Try to process
            try:
                result = process_audio_file(audio_path, genre, track_number)
                
                # Save to JSON (only after full success)
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                
                # Print summary
                n_nodes = len(result["nodes"])
                n_edges = len(result["edges"])
                all_node_counts.append(n_nodes)
                all_edge_counts.append(n_edges)
                
                print(
                    f"[OK] {audio_path.name:30} | "
                    f"nodes: {n_nodes:2d} | edges: {n_edges:2d}"
                )
                
                total_processed += 1
                genre_processed += 1
                
            except Exception as e:
                error_msg = f"{audio_path.name}: {str(e)[:100]}"
                print(f"[FAIL] {error_msg}")
                
                # Log full traceback
                logging.error(f"{audio_path.name}")
                logging.error(f"  {str(e)}")
                logging.error(traceback.format_exc())
                logging.error("-" * 80)
                
                failed_files.append(audio_path.name)
                total_failed += 1
                genre_failed += 1
        
        # Genre summary
        print(f"\nGenre summary ('{genre}'):")
        print(f"  Processed: {genre_processed}")
        print(f"  Skipped: {genre_skipped}")
        print(f"  Failed: {genre_failed}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total files processed: {total_processed}")
    print(f"Total files skipped (already done): {total_skipped}")
    print(f"Total files failed: {total_failed}")
    print(f"Total files: {total_processed + total_skipped + total_failed}")
    
    if all_node_counts:
        print(f"\nGraph statistics across {len(all_node_counts)} tracks:")
        print(f"  Average nodes per track: {np.mean(all_node_counts):.1f}")
        print(f"  Min nodes: {np.min(all_node_counts)}")
        print(f"  Max nodes: {np.max(all_node_counts)}")
        print(f"  Average edges per track: {np.mean(all_edge_counts):.1f}")
        print(f"  Min edges: {np.min(all_edge_counts)}")
        print(f"  Max edges: {np.max(all_edge_counts)}")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for fname in failed_files:
            print(f"  - {fname}")
        print(f"\nDetailed error log: {FAILED_LOG}")
    
    print("="*80)


if __name__ == "__main__":
    run_batch_processing()
