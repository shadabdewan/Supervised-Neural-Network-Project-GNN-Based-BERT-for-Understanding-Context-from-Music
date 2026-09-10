"""
Debug script to test chord detection and improved smoothing on a single track.

This script tests the chord extraction pipeline on one file to diagnose
and fix excessive chord flickering before running the full batch.

Process:
1. Extract raw chroma and label chords (before any smoothing)
2. Apply improved smoothing (larger median filter window + segment duration merge)
3. Print detailed diagnostics and compare against expected range
"""

from pathlib import Path
from typing import List, Dict
import numpy as np

from audio_features import extract_chroma, build_chord_templates, label_chords


# ============================================================================
# Configuration
# ============================================================================

TEST_FILE = Path(
    r"C:\studymatt\425\gnn-bert-music-context\data\raw\Data\genres_original\blues\blues.00034.wav"
)
MIN_SEGMENT_DURATION = 1.5  # seconds
MEDIAN_FILTER_WINDOW = 15   # frames (roughly 7.5 seconds at 0.5s per frame)


# ============================================================================
# Improved Smoothing Functions
# ============================================================================

def smooth_chord_labels_improved(
    chord_labels: List[str],
    window_size: int = MEDIAN_FILTER_WINDOW,
) -> List[str]:
    """
    Apply median filtering with larger window to smooth chord labels.
    
    Parameters
    ----------
    chord_labels : list of str
        Original chord labels, one per frame.
    window_size : int
        Sliding window size for majority voting (default: 15).
    
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


def segment_chords_improved(
    chord_labels: List[str],
    frame_times: np.ndarray,
    chroma_matrix: np.ndarray,
    min_duration: float = MIN_SEGMENT_DURATION,
) -> List[Dict]:
    """
    Collapse consecutive identical chords into segments, with minimum duration merging.
    
    After initial run-length encoding, merges any segment shorter than min_duration
    into whichever neighboring segment has longer duration. Repeats until all
    segments meet minimum duration requirement.
    
    Parameters
    ----------
    chord_labels : list of str
        Smoothed chord labels, one per frame.
    frame_times : np.ndarray
        Time in seconds for each frame (shape: n_frames).
    chroma_matrix : np.ndarray
        Chroma features (shape: 12, n_frames).
    min_duration : float
        Minimum allowed segment duration in seconds (default: 1.5).
    
    Returns
    -------
    segments : list of dict
        Each dict: {"chord": str, "start_time": float, "end_time": float,
                     "duration": float, "avg_chroma": np.ndarray (shape: 12)}
    """
    segments = []
    
    if len(chord_labels) == 0:
        return segments
    
    # Step 1: Run-length encoding into initial segments
    current_chord = chord_labels[0]
    segment_start_idx = 0
    
    for i in range(1, len(chord_labels)):
        if chord_labels[i] != current_chord:
            # End current segment
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
    
    # Step 2: Merge segments shorter than min_duration
    merged = True
    while merged:
        merged = False
        new_segments = []
        i = 0
        
        while i < len(segments):
            seg = segments[i]
            
            # Check if segment is too short
            if seg["duration"] < min_duration and i < len(segments) - 1:
                # Merge with neighbor - pick the one with longer duration
                prev_duration = segments[i - 1]["duration"] if i > 0 else 0
                next_duration = segments[i + 1]["duration"] if i + 1 < len(segments) else 0
                
                if prev_duration >= next_duration and i > 0:
                    # Merge with previous segment
                    prev_seg = new_segments.pop()  # Remove the previous segment we just added
                    
                    # Combine chroma vectors (weighted by original frame counts)
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
                    # Merge with next segment (or with previous if no next)
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
                        i += 2  # Skip next segment since we merged it
                        
                    else:
                        # This is the last segment, can't merge forward, keep as-is
                        new_segments.append(seg)
                        i += 1
            else:
                # Segment meets minimum duration or is the last one
                new_segments.append(seg)
                i += 1
        
        segments = new_segments
    
    return segments


def build_chord_graph_improved(segments: List[Dict]):
    """
    Build a graph from chord segments.
    
    Returns: (nodes, edges, edge_weights_list)
    """
    if len(segments) == 0:
        return [], [], []
    
    # Collect all unique chords and their average chroma features
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
    
    # Extract edge weights for statistics
    edge_weights = [w for (_, _, w) in sorted(
        [(src, tgt, weight) for (src, tgt), weight in edge_dict.items()]
    )]
    
    return nodes, edges, edge_weights


# ============================================================================
# Main Debug Script
# ============================================================================

def main():
    print("="*80)
    print("DEBUG: Single-Track Chord Detection Test")
    print("="*80)
    print(f"Test file: {TEST_FILE}\n")
    
    # Check file exists
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        return
    
    # Stage 1: Extract chroma and label chords (RAW)
    print("Stage 1: Extracting chroma and labeling chords (raw)...\n")
    chroma_matrix, frame_times = extract_chroma(str(TEST_FILE), sr=22050, window_sec=0.5)
    templates = build_chord_templates()
    raw_chord_labels = label_chords(chroma_matrix, templates)
    
    print(f"Total frames: {len(raw_chord_labels)}")
    print(f"Time per frame: 0.5 seconds")
    print(f"Total duration: {frame_times[-1]:.1f} seconds")
    print(f"\nRaw chord label sequence (first 100 frames):")
    print(f"{raw_chord_labels[:100]}\n")
    
    # Stage 2: Apply improved smoothing
    print("="*80)
    print("Stage 2: Applying improved smoothing...")
    print(f"  - Median filter window: {MEDIAN_FILTER_WINDOW} frames (~{MEDIAN_FILTER_WINDOW * 0.5:.1f} seconds)")
    print(f"  - Minimum segment duration: {MIN_SEGMENT_DURATION} seconds")
    print()
    
    smoothed_labels = smooth_chord_labels_improved(raw_chord_labels, window_size=MEDIAN_FILTER_WINDOW)
    segments = segment_chords_improved(smoothed_labels, frame_times, chroma_matrix, min_duration=MIN_SEGMENT_DURATION)
    
    print(f"Smoothed segments (total: {len(segments)}):\n")
    print(f"{'Chord':<8} {'Start (s)':<12} {'End (s)':<12} {'Duration (s)':<15}")
    print("-"*60)
    for seg in segments:
        print(f"{seg['chord']:<8} {seg['start_time']:<12.2f} {seg['end_time']:<12.2f} {seg['duration']:<15.2f}")
    
    # Stage 3: Build graph and analyze
    print("\n" + "="*80)
    print("Stage 3: Building chord-transition graph...")
    print()
    
    nodes, edges, edge_weights = build_chord_graph_improved(segments)
    
    unique_chords = len(nodes)
    num_edges = len(edges)
    
    print(f"Graph statistics:")
    print(f"  Unique chord nodes: {unique_chords}")
    print(f"  Edges (transitions): {num_edges}")
    print(f"  Edge weights: {edge_weights}")
    
    if edge_weights:
        print(f"  Weight distribution:")
        for weight in sorted(set(edge_weights)):
            count = edge_weights.count(weight)
            print(f"    Weight {weight}: {count} edge(s)")
    
    # Stage 4: Comparison against expected range
    print("\n" + "="*80)
    print("Stage 4: Validation against target specification")
    print("="*80)
    print(f"\nTarget from assignment spec: 3-8 unique chord nodes per 30-second track")
    print(f"Actual result: {unique_chords} unique chord nodes\n")
    
    if 3 <= unique_chords <= 8:
        print("✓ PASS - node count in expected range")
    else:
        if unique_chords > 8:
            print(f"✗ FAIL - node count still too high ({unique_chords} > 8)")
            print("  Recommendation: further increase smoothing window or adjust min segment duration")
        else:
            print(f"✗ FAIL - node count too low ({unique_chords} < 3)")
            print("  Recommendation: reduce smoothing window or lower min segment duration")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
