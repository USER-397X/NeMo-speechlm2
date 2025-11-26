#!/usr/bin/env python3
"""
Convert CSV metadata to NeMo manifest JSON format.

Converts: metadata.csv (audio_path, transcription)
      To: manifest.json (audio_filepath, text, duration)

Usage:
    python convert_csv_to_manifest.py \
        --input metadata.csv \
        --output manifest.json \
        --audio-dir /path/to/audio  # optional: prepend to audio paths
"""
import argparse
import csv
import json
import os
import sys
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf


def get_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"Warning: Could not read {audio_path}: {e}")
        return 0.0


def convert_csv_to_manifest(
    input_csv: str,
    output_json: str,
    audio_dir: str = None,
    audio_col: str = "audio_path",
    text_col: str = "transcription",
    delimiter: str = ",",
):
    """
    Convert CSV to NeMo manifest JSON format.

    Args:
        input_csv: Path to input CSV file
        output_json: Path to output JSON manifest
        audio_dir: Optional directory to prepend to audio paths
        audio_col: Column name for audio path (default: audio_path)
        text_col: Column name for transcription (default: transcription)
        delimiter: CSV delimiter (default: ,)
    """
    entries = []
    skipped = 0

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Validate columns
        if audio_col not in reader.fieldnames:
            raise ValueError(f"Column '{audio_col}' not found. Available: {reader.fieldnames}")
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found. Available: {reader.fieldnames}")

        rows = list(reader)

    print(f"Processing {len(rows)} entries...")

    for row in tqdm(rows, desc="Converting"):
        audio_path = row[audio_col].strip()
        text = row[text_col].strip()

        # Prepend audio directory if specified
        if audio_dir:
            audio_path = os.path.join(audio_dir, audio_path)

        # Get absolute path
        audio_path = os.path.abspath(audio_path)

        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            skipped += 1
            continue

        # Get duration
        duration = get_duration(audio_path)
        if duration == 0.0:
            skipped += 1
            continue

        entries.append({
            "audio_filepath": audio_path,
            "text": text,
            "duration": round(duration, 4)
        })

    # Write output
    with open(output_json, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nDone!")
    print(f"  Converted: {len(entries)} entries")
    print(f"  Skipped: {skipped} entries")
    print(f"  Output: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV metadata to NeMo manifest JSON format"
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON manifest")
    parser.add_argument("--audio-dir", help="Directory to prepend to audio paths")
    parser.add_argument("--audio-col", default="audio_path", help="Audio path column name (default: audio_path)")
    parser.add_argument("--text-col", default="transcription", help="Transcription column name (default: transcription)")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")

    args = parser.parse_args()

    convert_csv_to_manifest(
        input_csv=args.input,
        output_json=args.output,
        audio_dir=args.audio_dir,
        audio_col=args.audio_col,
        text_col=args.text_col,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()
