"""
Audio feature extraction for chord-transition graph building.

This module provides reusable functions for:
- Extracting chroma features from audio files
- Building chord templates (major and minor triads)
- Labeling chroma frames with chord names based on similarity

No file I/O or batch processing here - these are pure functions
imported and used by graph_builder.py.
"""

import numpy as np
import librosa


def extract_chroma(audio_path, sr=22050, window_sec=0.5):
    """
    Extract chroma features from an audio file with specified time resolution.
    
    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file (e.g., "blues.00000.wav")
    sr : int, optional
        Sample rate (default: 22050 Hz)
    window_sec : float, optional
        Desired time resolution per chroma frame in seconds (default: 0.5 sec)
    
    Returns
    -------
    chroma_matrix : np.ndarray
        Shape (12, n_frames). Chroma features with 12 pitch classes.
    frame_times : np.ndarray
        Shape (n_frames,). Time in seconds for each frame.
    
    Notes
    -----
    For a 30-second clip at window_sec=0.5, expect ~60 frames.
    Hop length is computed to achieve the desired window_sec resolution.
    """
    # Load audio file
    audio, _ = librosa.load(str(audio_path), sr=sr)
    
    # Compute hop_length to achieve desired window_sec resolution
    # hop_length [samples] = sr [samples/sec] * window_sec [sec]
    hop_length = int(sr * window_sec)
    
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    
    # Convert frame indices to time (in seconds)
    n_frames = chroma.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    
    return chroma, frame_times


def build_chord_templates():
    """
    Build 24 chord templates (12 major + 12 minor triads) as binary vectors.
    
    Each template is a 12-dimensional binary vector over pitch classes:
    [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
    
    Major triad: root, root+4 semitones, root+7 semitones
    Minor triad: root, root+3 semitones, root+7 semitones
    
    Returns
    -------
    templates : dict
        Mapping from chord name (e.g., "C", "C#m") to 12-dim binary numpy array.
    
    Examples
    --------
    >>> templates = build_chord_templates()
    >>> templates["C"]  # C major
    array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    >>> templates["Am"]  # A minor
    array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    """
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    templates = {}
    
    for root_idx, root_name in enumerate(pitch_classes):
        # Major triad: root, root+4, root+7 semitones
        major_template = np.zeros(12, dtype=int)
        major_template[root_idx] = 1
        major_template[(root_idx + 4) % 12] = 1
        major_template[(root_idx + 7) % 12] = 1
        templates[root_name] = major_template
        
        # Minor triad: root, root+3, root+7 semitones
        minor_template = np.zeros(12, dtype=int)
        minor_template[root_idx] = 1
        minor_template[(root_idx + 3) % 12] = 1
        minor_template[(root_idx + 7) % 12] = 1
        templates[f"{root_name}m"] = minor_template
    
    return templates


def label_chords(chroma_matrix, templates):
    """
    Label each chroma frame with the chord name of highest similarity.
    
    For each column (time frame) in chroma_matrix, computes cosine similarity
    to all 24 chord templates and assigns the chord name with highest similarity.
    
    Parameters
    ----------
    chroma_matrix : np.ndarray
        Shape (12, n_frames). Chroma features from extract_chroma().
    templates : dict
        Chord name -> 12-dim binary array, from build_chord_templates().
    
    Returns
    -------
    chord_labels : list of str
        Length n_frames. Chord label for each frame.
    
    Notes
    -----
    Uses cosine similarity (dot product / norms) for matching.
    """
    n_frames = chroma_matrix.shape[1]
    chord_labels = []
    
    for frame_idx in range(n_frames):
        frame_chroma = chroma_matrix[:, frame_idx]
        
        # Normalize frame chroma
        frame_norm = np.linalg.norm(frame_chroma)
        if frame_norm < 1e-6:
            # Degenerate frame, assign to first chord
            chord_labels.append(list(templates.keys())[0])
            continue
        
        # Compute cosine similarity to each template
        best_chord = None
        best_similarity = -1.0
        
        for chord_name, template in templates.items():
            template_norm = np.linalg.norm(template)
            if template_norm < 1e-6:
                continue
            
            # Cosine similarity
            similarity = np.dot(frame_chroma, template) / (frame_norm * template_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_chord = chord_name
        
        if best_chord is None:
            best_chord = list(templates.keys())[0]
        
        chord_labels.append(best_chord)
    
    return chord_labels
