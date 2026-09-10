import csv
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import whisper

# Navigate up from src/ to the project root (gnn-bert-music-context)
src_directory = Path(__file__).resolve().parent
project_root = src_directory.parent

dataset_directory = project_root / "data"/ "raw" / "Data" / "genres_original"
output_csv_path = project_root / "data" / "processed" / "lyrics_transcriptions.csv"
excluded_genres = {"classical", "jazz"}


def load_audio_with_ffmpeg(path: Path) -> np.ndarray:
    """Decode audio with imageio-ffmpeg's bundled executable for Whisper."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0


try:
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_directory}")

    audio_files = sorted(
        audio_path
        for audio_path in dataset_directory.glob("*/*.wav")
        if audio_path.parent.name not in excluded_genres
    )
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in: {dataset_directory}")

    existing_rows = {}
    if output_csv_path.is_file():
        with output_csv_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                key = (row.get("genre", ""), row.get("filename", ""))
                if key[0] not in excluded_genres and key[1]:
                    existing_rows[key] = row

    rows_to_process = []
    for audio_path in audio_files:
        key = (audio_path.parent.name, audio_path.name)
        existing_row = existing_rows.get(key)
        if existing_row and not existing_row.get("error", ""):
            continue
        rows_to_process.append(audio_path)

    print(
        f"Found {len(audio_files)} audio files. "
        f"Already completed: {len(audio_files) - len(rows_to_process)}. "
        f"Remaining: {len(rows_to_process)}."
    )

    model = None
    if rows_to_process:
        print("Loading Whisper model on CPU...")
        model = whisper.load_model("base", device="cpu")

    successful = 0
    failed = 0

    # Ensure output directory exists before writing
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "genre",
                "transcription",
                "lyrics_flag",
                "error",
            ],
        )
        writer.writeheader()

        for audio_path in audio_files:
            key = (audio_path.parent.name, audio_path.name)
            existing_row = existing_rows.get(key)
            if existing_row and not existing_row.get("error", ""):
                text = existing_row.get("transcription", "").strip()
                writer.writerow(
                    {
                        "filename": audio_path.name,
                        "genre": audio_path.parent.name,
                        "transcription": text,
                        "lyrics_flag": not bool(text),
                        "error": "",
                    }
                )
                continue

            print(f"[{successful + failed + 1}/{len(audio_files)}] Transcribing {audio_path.name}...")

            try:
                audio = load_audio_with_ffmpeg(audio_path)
                result = model.transcribe(audio, fp16=False)
                text = result["text"].strip()
                writer.writerow(
                    {
                        "filename": audio_path.name,
                        "genre": audio_path.parent.name,
                        "transcription": text,
                        "lyrics_flag": not bool(text),
                        "error": "",
                    }
                )
                print(text)
                successful += 1
            except Exception as error:
                failed += 1
                writer.writerow(
                    {
                        "filename": audio_path.name,
                        "genre": audio_path.parent.name,
                        "transcription": "",
                        "lyrics_flag": True,
                        "error": str(error),
                    }
                )
                print(f"Failed to transcribe {audio_path}: {error}")

    print(
        f"\nFinished: {successful} transcribed, {failed} failed. "
        f"Results saved in: {output_csv_path}"
    )
except FileNotFoundError as error:
    print(f"Error: {error}")
except Exception as error:
    print(f"Whisper transcription failed: {error}") 