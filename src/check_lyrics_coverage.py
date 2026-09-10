"""Check lyrics coverage and GNN/BERT fusion usability.

This script reads the processed lyrics CSV, measures coverage quality for the
BERT pipeline, and reports how many tracks are available for the Task 3 fusion
set after excluding the GNN-degenerate tracks.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LYRICS_CSV = PROJECT_ROOT / "data" / "processed" / "lyrics_transcriptions.csv"
EXCLUDED_TRACKS_TXT = PROJECT_ROOT / "data" / "processed" / "excluded_tracks.txt"
VERY_SHORT_WORD_THRESHOLD = 5


def word_count(text) -> int:
    """Return the number of words in a lyrics string; NaN/empty -> 0."""
    if pd.isna(text):
        return 0
    text = str(text).strip()
    return len(text.split()) if text else 0


def char_count(text) -> int:
    """Return the number of characters in a lyrics string; NaN/empty -> 0."""
    if pd.isna(text):
        return 0
    return len(str(text).strip())


def normalize_track_id(raw_value) -> str:
    """Normalize filenames like 'blues.00002.wav' or 'blues.00002.json' to 'blues.00002'."""
    if pd.isna(raw_value):
        return ""
    return Path(str(raw_value).strip()).stem


def load_excluded_tracks(path: Path) -> set:
    """Load the track IDs from the GNN excluded list."""
    if not path.exists():
        print(f"WARNING: {path} not found - treating excluded set as empty.\n")
        return set()

    with path.open("r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    excluded = set()
    for line in raw_lines:
        if line.startswith("#"):
            continue
        excluded.add(normalize_track_id(line))
    return excluded


def main():
    if not LYRICS_CSV.exists():
        raise FileNotFoundError(f"Could not find {LYRICS_CSV}")

    df = pd.read_csv(LYRICS_CSV)

    # The processed CSV uses 'transcription' in this project, while the task description
    # says the column may be named 'lyrics'. Support both names.
    lyrics_col = "lyrics" if "lyrics" in df.columns else "transcription"
    df = df.rename(columns={lyrics_col: "lyrics"})

    # Compute counts.
    df["char_count"] = df["lyrics"].apply(char_count)
    df["word_count"] = df["lyrics"].apply(word_count)

    # Flag rows.
    df["is_empty"] = df["word_count"] == 0
    df["is_very_short"] = (df["word_count"] > 0) & (df["word_count"] < VERY_SHORT_WORD_THRESHOLD)
    df["is_usable_bert"] = df["word_count"] >= VERY_SHORT_WORD_THRESHOLD

    # Per-genre summary.
    summary_df = (
        df.groupby("genre", dropna=False)
        .agg(
            total_tracks=("filename", "count"),
            empty=("is_empty", "sum"),
            very_short=("is_very_short", "sum"),
            usable_5plus=("is_usable_bert", "sum"),
            avg_word_count=("word_count", "mean"),
        )
        .reset_index()
        .sort_values("genre")
    )

    print("=" * 100)
    print("PER-GENRE LYRICS COVERAGE SUMMARY")
    print("=" * 100)
    print(
        summary_df.to_string(
            index=False,
            formatters={"avg_word_count": lambda x: f"{x:.2f}"},
        )
    )

    # Overall totals.
    total_tracks = len(df)
    total_empty = int(df["is_empty"].sum())
    total_very_short = int(df["is_very_short"].sum())
    total_usable_bert = int(df["is_usable_bert"].sum())
    overall_avg_words = df["word_count"].mean()

    print("\n" + "=" * 100)
    print("OVERALL TOTALS (ALL GENRES)")
    print("=" * 100)
    print(f"Total tracks:                             {total_tracks}")
    print(f"Empty (0 words):                          {total_empty}")
    print(f"Very short (<{VERY_SHORT_WORD_THRESHOLD} words):    {total_very_short}")
    print(f"Usable for BERT (5+ words):               {total_usable_bert}")
    print(f"Average word count:                       {overall_avg_words:.2f}")

    # Cross-reference against the GNN excluded list.
    excluded_ids = load_excluded_tracks(EXCLUDED_TRACKS_TXT)
    df["track_id"] = df["filename"].apply(normalize_track_id)
    df["is_gnn_usable"] = ~df["track_id"].isin(excluded_ids)
    df["is_fusion_usable"] = df["is_gnn_usable"] & df["is_usable_bert"]

    n_gnn_usable = int(df["is_gnn_usable"].sum())
    n_fusion_usable = int(df["is_fusion_usable"].sum())

    print("\n" + "=" * 100)
    print("TASK 3 FUSION USABILITY")
    print("=" * 100)
    print(f"Tracks in GNN excluded list:              {len(excluded_ids)}")
    print(f"Tracks usable for GNN (not excluded):     {n_gnn_usable}")
    print(f"Tracks usable for BERT (5+ words):        {total_usable_bert}")
    print(f"Tracks usable for BOTH (fusion-ready):    {n_fusion_usable}")
    if total_tracks:
        print(f"Fusion-ready fraction of all tracks:      {n_fusion_usable / total_tracks:.2%}")

    # Per-genre fusion-ready counts.
    fusion_df = (
        df.groupby("genre", dropna=False)
        .agg(fusion_ready=("is_fusion_usable", "sum"))
        .reset_index()
        .sort_values("genre")
    )

    print("\nPer-genre fusion-ready counts:")
    print(fusion_df.to_string(index=False))


if __name__ == "__main__":
    main()
