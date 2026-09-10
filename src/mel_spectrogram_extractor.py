import csv
from pathlib import Path
import librosa
import numpy as np

# Fixed audio parameter constants
SAMPLE_RATE = 22050
N_MELS = 128
TARGET_FRAMES = 1300


def extract_and_save_spectrograms():
  # Robust path setup relative to script execution context
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent

  raw_audio_dir = (
      Path(
          r"C:\studymatt\425\gnn-bert-music-context\data\raw\Data\genres_original"
      )
      if Path(
          r"C:\studymatt\425\gnn-bert-music-context\data\raw\Data\genres_original"
      ).exists()
      else project_root / "data" / "raw" / "Data" / "genres_original"
  )

  data_processed_dir = project_root / "data" / "processed"
  excluded_tracks_path = data_processed_dir / "excluded_tracks.txt"
  output_dir = data_processed_dir / "mel_spectrograms"
  labels_csv_path = output_dir / "labels.csv"

  output_dir.mkdir(parents=True, exist_ok=True)

  # 1. Load set of excluded track filenames
  excluded_files = set()
  if excluded_tracks_path.exists():
    with open(excluded_tracks_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
          # Store stem and raw name for matching
          excluded_files.add(line)
          excluded_files.add(Path(line).stem)
    print(f"Loaded {len(excluded_files)} excluded track references.")

  # 2. Collect all wav files
  wav_files = sorted(list(raw_audio_dir.rglob("*.wav")))
  print(f"Found {len(wav_files)} total audio files in {raw_audio_dir}")

  processed_count = 0
  skipped_excluded_count = 0
  skipped_existing_count = 0

  manifest_entries = []
  sample_shape = None

  for wav_path in wav_files:
    track_stem = wav_path.stem
    genre = wav_path.parent.name

    # Skip files flagged in excluded_tracks.txt
    if track_stem in excluded_files or wav_path.name in excluded_files:
      skipped_excluded_count += 1
      continue

    npy_filename = f"{track_stem}.npy"
    out_npy_path = output_dir / npy_filename

    manifest_entries.append({"filename": npy_filename, "genre": genre})

    # Resumable execution check: skip if .npy already exists
    if out_npy_path.exists():
      skipped_existing_count += 1
      if sample_shape is None:
        sample_shape = np.load(out_npy_path).shape
      continue

    try:
      # 3. Load audio and compute Log-Mel Spectrogram
      y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
      mel_spec = librosa.feature.melspectrogram(
          y=y, sr=sr, n_mels=N_MELS, hop_length=512
      )
      log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

      # 4. Pad or trim to fixed shape (128, 1300)
      current_frames = log_mel_spec.shape[1]
      min_db = log_mel_spec.min()

      if current_frames < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - current_frames
        log_mel_spec = np.pad(
            log_mel_spec,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=min_db,
        )
      elif current_frames > TARGET_FRAMES:
        log_mel_spec = log_mel_spec[:, :TARGET_FRAMES]

      # 5. Save array
      np.save(out_npy_path, log_mel_spec.astype(np.float32))
      processed_count += 1
      if sample_shape is None:
        sample_shape = log_mel_spec.shape

      print(f"Processed: {wav_path.name} -> {npy_filename}")

    except Exception as e:
      print(f"Error processing {wav_path.name}: {e}")

  # 6. Save labels.csv manifest file
  with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "genre"])
    writer.writeheader()
    writer.writerows(manifest_entries)

  # 7. Print summary
  print("\n" + "=" * 60)
  print("MEL SPECTROGRAM EXTRACTION SUMMARY")
  print("=" * 60)
  print(f"New files processed & saved : {processed_count}")
  print(f"Files previously extracted  : {skipped_existing_count}")
  print(f"Excluded tracks skipped     : {skipped_excluded_count}")
  print(f"Total active dataset size   : {len(manifest_entries)}")
  print(f"Spectrogram shape           : {sample_shape}")
  print(f"Manifest written to         : {labels_csv_path.resolve()}")
  print("=" * 60)


if __name__ == "__main__":
  extract_and_save_spectrograms()