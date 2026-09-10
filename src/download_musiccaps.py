# Note: YouTube availability changes over time due to video deletions, privacy setting changes,
# and copyright actions. Success rate is expected to be less than 100% — anywhere from 70-95% is normal.

import csv
import subprocess
import sys
import urllib.request
import logging
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def load_musiccaps_dataset():
    """Load the MusicCaps dataset via HuggingFace datasets or fallback to raw CSV URL."""
    print("Loading MusicCaps dataset...")
    try:
        from datasets import load_dataset

        ds = load_dataset("google/MusicCaps", split="train")
        rows = [row for row in ds]
        print("Successfully loaded dataset via HuggingFace `datasets` library.")
        return rows
    except Exception as e:
        print(f"Hugging Face `datasets` load failed ({e}). Falling back to direct CSV download...")
        csv_urls = [
            "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv?download=true",
            "https://raw.githubusercontent.com/google-research/google-research/master/musiccaps/musiccaps-public.csv",
        ]
        last_error = e
        for csv_url in csv_urls:
            try:
                with urllib.request.urlopen(csv_url, timeout=30) as response:
                    csv_content = response.read().decode("utf-8")
                    if csv_content.startswith("version https://git-lfs.github.com/spec/v1"):
                        raise ValueError("received a Git LFS pointer instead of CSV content")

                    csv_text = csv_content.splitlines()
                    reader = csv.DictReader(csv_text)
                    required_columns = {"ytid", "caption", "start_s", "end_s"}
                    actual_columns = set(reader.fieldnames or [])
                    missing_columns = required_columns - actual_columns
                    if missing_columns:
                        raise ValueError(
                            f"CSV is missing required columns: {sorted(missing_columns)}"
                        )
                    rows = list(reader)
                if not rows:
                    raise ValueError("CSV contained no rows")
                print(f"Successfully loaded MusicCaps CSV from {csv_url}.")
                return rows
            except Exception as fallback_error:
                last_error = fallback_error
        raise RuntimeError(
            f"Unable to load MusicCaps metadata from HuggingFace or GitHub: {last_error}"
        ) from last_error


def download_and_trim_audio(ytid, start_s, end_s, output_path):
    """Downloads audio from YouTube using yt-dlp and trims it to start_s -> end_s using ffmpeg."""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Install it with: pip install yt-dlp")

    youtube_url = f"https://www.youtube.com/watch?v={ytid}"
    temp_download_template = output_path.parent / f"temp_{ytid}.%(ext)s"
    start_sec = float(start_s)
    end_sec = float(end_s)
    duration = end_sec - start_sec
    if start_sec < 0 or duration <= 0:
        raise ValueError(f"Invalid clip window: start_s={start_s}, end_s={end_s}")

    for stale_file in output_path.parent.glob(f"temp_{ytid}.*"):
        stale_file.unlink()

    # 1. Download best audio with yt-dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(temp_download_template),
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Locate downloaded temp audio file (extension might be webm, m4a, etc.)
    downloaded_files = list(output_path.parent.glob(f"temp_{ytid}.*"))
    if not downloaded_files:
        raise FileNotFoundError("yt-dlp completed but temp audio file was not found.")

    temp_audio_file = downloaded_files[0]

    try:
        # 2. Trim and convert to 22050 Hz mono WAV using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss",
            str(start_sec),  # Start time
            "-i",
            str(temp_audio_file),  # Input temp file
            "-t",
            str(duration),  # Duration
            "-ar",
            "22050",  # Sample rate 22050 Hz
            "-ac",
            "1",  # Mono audio
            "-c:a",
            "pcm_s16le",  # WAV encoding
            str(output_path),
        ]

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed: {detail}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg completed but output WAV is missing or empty")
    finally:
        # Clean up temporary download file
        if temp_audio_file.exists():
            temp_audio_file.unlink()


def main():
    if yt_dlp is None:
        print("Error: yt-dlp is not installed. Install it with: pip install yt-dlp")
        return

    # Setup directories
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir

    raw_dir = project_root / "data" / "raw"
    audio_dir = raw_dir / "musiccaps_audio"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    failed_log_path = raw_dir / "musiccaps_failed.log"
    metadata_csv_path = raw_dir / "musiccaps_metadata.csv"

    # Configure failure logger
    logger = logging.getLogger("download_musiccaps")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    file_handler = logging.FileHandler(failed_log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(file_handler)

    # 1. Load dataset
    full_dataset = load_musiccaps_dataset()

    # 2. Select FIRST 300 rows and confirm non-empty captions
    selected_rows = full_dataset[:300]
    print(f"\nSelected first {len(selected_rows)} rows for download.")

    non_empty_caption_count = 0
    for r in selected_rows:
        caption = r.get("caption", "").strip()
        if caption:
            non_empty_caption_count += 1

    print(
        f"Confirmed non-empty captions for {non_empty_caption_count}/{len(selected_rows)} selected rows."
    )
    if non_empty_caption_count != len(selected_rows):
        raise ValueError("MusicCaps selection contains empty captions; refusing to create an incomplete dataset.")

    # 3. Download Clips
    total_attempted = len(selected_rows)
    succeeded_rows = []
    failed_count = 0

    print("\nStarting audio downloads...\n" + "-" * 50)

    for idx, row in enumerate(selected_rows, start=1):
        ytid = str(row["ytid"]).strip()
        caption = str(row["caption"]).strip()
        start_s = row["start_s"]
        end_s = row["end_s"]

        filename = f"{ytid}.wav"
        output_wav_path = audio_dir / filename

        # Resumability check
        if output_wav_path.exists() and output_wav_path.stat().st_size > 0:
            succeeded_rows.append({"ytid": ytid, "caption": caption, "filename": filename})
        else:
            try:
                download_and_trim_audio(ytid, start_s, end_s, output_wav_path)
                succeeded_rows.append({"ytid": ytid, "caption": caption, "filename": filename})
            except Exception as e:
                failed_count += 1
                err_msg = str(e).replace("\n", " ")
                logger.error(f"{ytid} | Error: {err_msg}")

        # Progress update every 10 clips
        if idx % 10 == 0 or idx == total_attempted:
            current_succeeded = len(succeeded_rows)
            print(
                f"Progress: [{idx}/{total_attempted}] | "
                f"Succeeded: {current_succeeded} | Failed: {failed_count}"
            )

    # 4. Save metadata CSV for successfully downloaded clips
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ytid", "caption", "filename"])
        writer.writeheader()
        writer.writerows(succeeded_rows)

    # 5. Final Summary
    total_succeeded = len(succeeded_rows)
    success_rate = (total_succeeded / total_attempted * 100) if total_attempted > 0 else 0.0

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total Attempted : {total_attempted}")
    print(f"Total Succeeded : {total_succeeded}")
    print(f"Total Failed    : {failed_count}")
    print(f"Success Rate    : {success_rate:.2f}%")
    print(f"Metadata Saved  : {metadata_csv_path.resolve()}")
    print(f"Audio Saved To  : {audio_dir.resolve()}")
    if failed_count > 0:
        print(f"Failure Log     : {failed_log_path.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()