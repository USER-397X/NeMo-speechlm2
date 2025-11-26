#!/usr/bin/env python3
"""
Enhanced Lhotse Shar Converter - Convert ASR datasets to Lhotse shar format with multi-shard support.

This script supports:
1. Triple input modes: JSON manifest, TXT file pairs, or existing Lhotse shar
2. Configurable sharding (default: 50 entries per shard)
3. Dynamic digit numbering (6+ digits based on total shards)
4. Custom fields preservation including raw_transcript backup
5. FLAC audio format in recording.tar archives

Output format:
- cuts.000000.jsonl.gz, cuts.000001.jsonl.gz, ...
- recording.000000.tar, recording.000001.tar, ...
"""
import argparse
import gzip
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import tarfile
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Check and install dependencies
try:
    import lhotse
except ImportError:
    print("Lhotse not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lhotse"])
    import lhotse

try:
    import soundfile as sf
except ImportError:
    print("Soundfile not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.shar import SharWriter


def setup_logging(log_filename: Optional[str] = None, level: str = 'INFO'):
    """Set up logging configuration.

    Args:
        log_filename: Custom log filename. If None, uses timestamp-based name
        level: Logging level (INFO, DEBUG, etc.)
    """
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'lhotse_shar_conversion_{timestamp}.log'

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_filename}")
    return logger


# Initialize logger
logger = logging.getLogger(__name__)


def read_json_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Read JSON manifest file line by line.

    Args:
        manifest_path: Path to the JSON manifest file

    Returns:
        List of dictionaries containing manifest entries with audio_filepath, text, duration
    """
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                # Validate required fields
                if 'audio_filepath' not in entry:
                    logger.error(f"Line {line_num}: Missing 'audio_filepath' field")
                    continue
                if 'text' not in entry:
                    logger.error(f"Line {line_num}: Missing 'text' field")
                    continue
                if 'duration' not in entry:
                    logger.warning(f"Line {line_num}: Missing 'duration' field, will measure from audio")

                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line {line_num}: {e}")
                logger.error(f"Line content: {line.strip()}")
                continue

    logger.info(f"Successfully read {len(entries)} entries from {manifest_path}")
    return entries


def read_txt_manifest(audio_txt: str, transcript_txt: str) -> List[Dict[str, Any]]:
    """Read TXT manifest files (audio paths and transcripts).

    Args:
        audio_txt: Path to text file containing audio file paths (one per line)
        transcript_txt: Path to text file containing transcripts (one per line, aligned with audio)

    Returns:
        List of dictionaries containing manifest entries with audio_filepath, text, duration
    """
    logger.info(f"Reading audio paths from: {audio_txt}")
    logger.info(f"Reading transcripts from: {transcript_txt}")

    # Read audio file paths
    with open(audio_txt, 'r', encoding='utf-8') as f:
        audio_paths = [line.strip() for line in f if line.strip()]

    # Read transcripts
    with open(transcript_txt, 'r', encoding='utf-8') as f:
        transcripts = [line.strip() for line in f if line.strip()]

    # Validate line count matching
    if len(audio_paths) != len(transcripts):
        logger.error(
            f"Line count mismatch: {len(audio_paths)} audio paths vs {len(transcripts)} transcripts"
        )
        raise ValueError(
            f"Audio filepath and transcript files must have the same number of lines. "
            f"Got {len(audio_paths)} audio paths and {len(transcripts)} transcripts."
        )

    logger.info(f"Processing {len(audio_paths)} audio-transcript pairs...")

    # Create manifest entries (duration will be measured during processing)
    entries = []
    for audio_path, text in zip(audio_paths, transcripts):
        entry = {
            'audio_filepath': audio_path,
            'text': text,
            # Duration will be measured in process_manifest_entry
        }
        entries.append(entry)

    logger.info(f"Successfully created {len(entries)} entries from TXT manifest")
    return entries


def read_shar_manifest(shar_input_dir: str, temp_dir: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Read existing Lhotse shar format files and extract to manifest entries.

    Args:
        shar_input_dir: Directory containing cuts.XXXXXX.jsonl.gz and recording.XXXXXX.tar files
        temp_dir: Temporary directory for extracted audio files (auto-created if None)

    Returns:
        Tuple of (manifest entries list, temp_dir_path)
    """
    logger.info(f"Reading Lhotse shar format from: {shar_input_dir}")

    if not os.path.isdir(shar_input_dir):
        raise ValueError(f"Shar input directory does not exist: {shar_input_dir}")

    # Create temporary directory for extracted audio files
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='lhotse_shar_extract_')
        logger.info(f"Created temporary directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    # Find all cuts and recording files
    cuts_files = []
    recording_files = []

    # Support both numbered (cuts.000000.jsonl.gz) and non-numbered (cuts.jsonl.gz) formats
    cuts_pattern = re.compile(r'cuts(?:\.(\d+))?\.jsonl\.gz')
    recording_pattern = re.compile(r'recording(?:\.(\d+))?\.tar')

    for filename in sorted(os.listdir(shar_input_dir)):
        cuts_match = cuts_pattern.match(filename)
        if cuts_match:
            # Use empty string for non-numbered files, otherwise use the number
            shard_id = cuts_match.group(1) if cuts_match.group(1) else ""
            cuts_files.append((shard_id, os.path.join(shar_input_dir, filename)))

        recording_match = recording_pattern.match(filename)
        if recording_match:
            # Use empty string for non-numbered files, otherwise use the number
            shard_id = recording_match.group(1) if recording_match.group(1) else ""
            recording_files.append((shard_id, os.path.join(shar_input_dir, filename)))

    logger.info(f"Found {len(cuts_files)} cuts files and {len(recording_files)} recording files")

    if not cuts_files or not recording_files:
        raise ValueError(
            f"No shar files found in {shar_input_dir}. "
            f"Expected cuts[.XXXXXX].jsonl.gz and recording[.XXXXXX].tar files "
            f"(both numbered and non-numbered formats supported)."
        )

    # Create shard pairs
    cuts_dict = {shard_id: path for shard_id, path in cuts_files}
    recording_dict = {shard_id: path for shard_id, path in recording_files}

    # Find common shard IDs
    common_shards = sorted(set(cuts_dict.keys()) & set(recording_dict.keys()))

    if not common_shards:
        raise ValueError("No matching cuts/recording pairs found")

    # Log appropriate message for single vs. multi-shard
    if len(common_shards) == 1 and common_shards[0] == "":
        logger.info(f"Processing single shard (non-numbered format)")
    else:
        logger.info(f"Processing {len(common_shards)} shard pairs")

    # Process each shard pair
    all_entries = []

    for shard_id in tqdm(common_shards, desc="Extracting shards"):
        cuts_path = cuts_dict[shard_id]
        recording_path = recording_dict[shard_id]

        # Format shard identifier for logging
        shard_label = "single shard" if shard_id == "" else f"shard {shard_id}"
        logger.debug(f"Processing {shard_label}: {os.path.basename(cuts_path)} + {os.path.basename(recording_path)}")

        # Extract audio files from tar
        shard_temp_dir = os.path.join(temp_dir, f"shard_{shard_id if shard_id else 'single'}")
        os.makedirs(shard_temp_dir, exist_ok=True)

        audio_files = {}  # id -> filepath mapping

        # Supported audio extensions
        audio_extensions = {'.flac', '.wav', '.opus', '.mp3', '.ogg', '.m4a', '.wma'}

        try:
            with tarfile.open(recording_path, 'r') as tar:
                members = tar.getmembers()
                total_files = len(members)
                logger.debug(f"  Found {total_files} files in tar...")

                skipped_count = 0
                for member in members:
                    if member.isfile():
                        # Check if file is an audio file
                        file_ext = os.path.splitext(member.name)[1].lower()

                        if file_ext not in audio_extensions:
                            logger.debug(f"  Skipping non-audio file: {member.name}")
                            skipped_count += 1
                            continue

                        # Extract to temp directory
                        tar.extract(member, path=shard_temp_dir)
                        extracted_path = os.path.join(shard_temp_dir, member.name)

                        # Get ID from filename (remove extension)
                        file_id = os.path.splitext(os.path.basename(member.name))[0]
                        audio_files[file_id] = extracted_path

                if skipped_count > 0:
                    logger.debug(f"  Skipped {skipped_count} non-audio files")
                logger.debug(f"  Extracted {len(audio_files)} audio files")

        except Exception as e:
            logger.error(f"Failed to extract audio from {recording_path}: {e}")
            continue

        # Read metadata from cuts file
        try:
            with gzip.open(cuts_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        cut_data = json.loads(line.strip())

                        # Extract ID
                        cut_id = cut_data.get('id')
                        if not cut_id:
                            logger.warning(f"  Line {line_num}: No ID found, skipping")
                            continue

                        # Find corresponding audio file
                        if cut_id not in audio_files:
                            logger.warning(f"  Cut ID {cut_id}: No matching audio file found")
                            continue

                        audio_filepath = audio_files[cut_id]

                        # Extract text from supervisions
                        text = ""
                        if 'supervisions' in cut_data and cut_data['supervisions']:
                            text = cut_data['supervisions'][0].get('text', '')

                        # Extract duration
                        duration = cut_data.get('duration', 0.0)

                        # Create manifest entry
                        entry = {
                            'id': cut_id,
                            'audio_filepath': audio_filepath,
                            'text': text,
                            'duration': duration,
                        }

                        # Preserve custom fields if they exist
                        if 'supervisions' in cut_data and cut_data['supervisions']:
                            if 'custom' in cut_data['supervisions'][0]:
                                entry['custom'] = cut_data['supervisions'][0]['custom']

                        all_entries.append(entry)

                    except json.JSONDecodeError as e:
                        logger.error(f"  Failed to parse line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to read cuts file {cuts_path}: {e}")
            continue

    logger.info(f"Successfully extracted {len(all_entries)} entries from {len(common_shards)} shards")
    logger.info(f"Temporary audio files stored in: {temp_dir}")

    return all_entries, temp_dir


def calculate_digit_width(total_shards: int) -> int:
    """Calculate the number of digits needed for shard numbering.

    Args:
        total_shards: Total number of shards

    Returns:
        Number of digits (minimum 6, expandable based on total_shards)
    """
    return max(6, len(str(total_shards)))


def process_manifest_entry(args: Tuple[Dict[str, Any], int]) -> Optional[Tuple[MonoCut, Dict[str, Any]]]:
    """Process a single manifest entry and create Lhotse MonoCut.

    Args:
        args: Tuple containing (manifest_entry, worker_id)

    Returns:
        Tuple of (MonoCut object, metadata_dict) or None if processing failed
    """
    entry, worker_id = args

    try:
        # Extract audio filepath
        audio_filepath = entry.get('audio_filepath')
        if not audio_filepath:
            logger.error(f"[Worker {worker_id}] No audio_filepath in entry")
            return None

        # Verify the audio file exists
        if not os.path.exists(audio_filepath):
            logger.error(f"[Worker {worker_id}] Audio file not found: {audio_filepath}")
            return None

        # Get audio file info
        info = sf.info(audio_filepath)

        # Log audio file info in debug mode
        logger.debug(f"[Worker {worker_id}] Audio: {os.path.basename(audio_filepath)} | "
                     f"Duration: {info.duration:.4f}s | SR: {info.samplerate}Hz | "
                     f"Samples: {info.frames} | Channels: {info.channels}")

        # Use existing ID or generate new one
        cut_id = entry.get('id', uuid.uuid4().hex)

        # Extract text
        text = entry.get('text', '')

        # Create Recording
        recording = Recording(
            id=cut_id,
            sampling_rate=info.samplerate,
            num_samples=info.frames,
            duration=info.duration,
            sources=[
                AudioSource(
                    type="file",
                    source=os.path.abspath(audio_filepath),
                    channels=list(range(info.channels))
                )
            ],
        )

        # Build custom fields for supervision
        custom_fields = {}

        # Add raw_transcript as backup of original text
        custom_fields['raw_transcript'] = text

        # Preserve any existing custom fields from input
        if 'supervisions' in entry and entry['supervisions'] and 'custom' in entry['supervisions'][0]:
            input_custom = entry['supervisions'][0]['custom']
            # Merge existing custom fields (raw_transcript takes precedence)
            for key, value in input_custom.items():
                if key not in custom_fields:
                    custom_fields[key] = value

        # Create SupervisionSegment
        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut_id,
            start=0.0,
            duration=info.duration,
            channel=0,
            text=text,
            custom=custom_fields,
        )

        # Create MonoCut
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=info.duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
        )

        # Create metadata for verification
        metadata = {
            'id': cut_id,
            'audio_filepath': audio_filepath,
            'text': text,
            'duration': info.duration,
            'sampling_rate': info.samplerate,
            'custom_fields': list(custom_fields.keys()),
        }

        return cut, metadata

    except Exception as e:
        logger.error(f"[Worker {worker_id}] Failed to process entry: {e}")
        logger.debug(f"[Worker {worker_id}] Entry: {entry}")
        return None


def create_lhotse_shar_dataset(
    entries: List[Dict[str, Any]],
    output_dir: str,
    shard_size: int = 50,
    audio_format: str = 'flac',
    num_workers: int = None,
) -> None:
    """Create Lhotse shar dataset with multi-shard support.

    Args:
        entries: List of manifest entries
        output_dir: Directory to save output files
        shard_size: Number of entries per shard (default: 50)
        audio_format: Format to save audio files in recording.tar (default: flac)
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if not entries:
        logger.error("No entries provided")
        return

    # Calculate total shards and digit width
    total_entries = len(entries)
    total_shards = (total_entries + shard_size - 1) // shard_size
    digit_width = calculate_digit_width(total_shards)

    logger.info(f"Processing {total_entries} entries")
    logger.info(f"Shard size: {shard_size} entries per shard")
    logger.info(f"Total shards: {total_shards}")
    logger.info(f"Digit width: {digit_width} (e.g., cuts.{'0' * digit_width}.jsonl.gz)")

    # Prepare worker arguments
    worker_args = [(entry, i % num_workers) for i, entry in enumerate(entries)]

    # Process entries in parallel
    logger.info(f"Processing entries with {num_workers} workers...")
    cuts = []
    metadata_list = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap(process_manifest_entry, worker_args),
            total=len(worker_args),
            desc="Processing entries"
        ):
            if result is not None:
                cut, metadata = result
                cuts.append(cut)
                metadata_list.append(metadata)

    logger.info(f"Successfully processed {len(cuts)} out of {total_entries} entries")

    if not cuts:
        logger.error("No cuts created, cannot proceed")
        return

    # Create CutSet
    cut_set = CutSet.from_cuts(cuts)

    # Export to shar format with sharding
    logger.info(f"Exporting to shar format with audio format: {audio_format}")
    logger.info(f"Output directory: {output_dir}")

    try:
        with SharWriter(
            output_dir=output_dir,
            fields={"recording": audio_format},
            shard_size=shard_size,
            include_cuts=True,
        ) as writer:
            for cut in tqdm(cut_set, desc="Writing shards"):
                writer.write(cut)

        logger.info(f"Successfully exported {len(cuts)} cuts to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to write shar: {e}")
        raise

    # Rename files with consistent digit width
    rename_shar_files(output_dir, digit_width)

    # Verify output
    verify_shar_output(output_dir, total_shards, digit_width, metadata_list)


def rename_shar_files(output_dir: str, digit_width: int) -> None:
    """Rename shar output files to use consistent digit width.

    Args:
        output_dir: Directory containing shar files
        digit_width: Number of digits for shard numbering
    """
    logger.info(f"Renaming shar files to use {digit_width}-digit numbering...")

    # List all files
    files = os.listdir(output_dir)

    # Pattern matching for cuts and recording files
    import re
    cuts_pattern = re.compile(r'cuts\.(\d+)\.jsonl\.gz')
    recording_pattern = re.compile(r'recording\.(\d+)\.tar')

    renamed_count = 0

    for filename in files:
        new_filename = None

        # Check for cuts files
        match = cuts_pattern.match(filename)
        if match:
            shard_num = int(match.group(1))
            new_filename = f"cuts.{shard_num:0{digit_width}d}.jsonl.gz"

        # Check for recording files
        match = recording_pattern.match(filename)
        if match:
            shard_num = int(match.group(1))
            new_filename = f"recording.{shard_num:0{digit_width}d}.tar"

        # Rename if needed
        if new_filename and new_filename != filename:
            old_path = os.path.join(output_dir, filename)
            new_path = os.path.join(output_dir, new_filename)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                logger.debug(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1

    logger.info(f"Renamed {renamed_count} files")


def verify_shar_output(
    output_dir: str,
    expected_shards: int,
    digit_width: int,
    metadata_list: List[Dict[str, Any]]
) -> None:
    """Verify the shar output files.

    Args:
        output_dir: Directory containing shar files
        expected_shards: Expected number of shards
        digit_width: Number of digits in shard numbering
        metadata_list: List of metadata dictionaries for verification
    """
    logger.info("Verifying shar output...")

    # Check for cuts and recording files
    cuts_files = []
    recording_files = []

    for i in range(expected_shards):
        shard_id = f"{i:0{digit_width}d}"

        cuts_file = os.path.join(output_dir, f"cuts.{shard_id}.jsonl.gz")
        recording_file = os.path.join(output_dir, f"recording.{shard_id}.tar")

        if os.path.exists(cuts_file):
            cuts_files.append(cuts_file)
            size_kb = os.path.getsize(cuts_file) / 1024
            logger.debug(f"✓ cuts.{shard_id}.jsonl.gz ({size_kb:.1f} KB)")
        else:
            logger.warning(f"✗ Missing: cuts.{shard_id}.jsonl.gz")

        if os.path.exists(recording_file):
            recording_files.append(recording_file)
            size_mb = os.path.getsize(recording_file) / (1024 * 1024)
            logger.debug(f"✓ recording.{shard_id}.tar ({size_mb:.1f} MB)")
        else:
            logger.warning(f"✗ Missing: recording.{shard_id}.tar")

    logger.info(f"Found {len(cuts_files)}/{expected_shards} cuts files")
    logger.info(f"Found {len(recording_files)}/{expected_shards} recording files")

    # Verify first cuts file content
    if cuts_files:
        first_cuts = cuts_files[0]
        logger.info(f"Verifying first cuts file: {os.path.basename(first_cuts)}")

        try:
            with gzip.open(first_cuts, 'rt', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    cut_data = json.loads(first_line)

                    # Check for required fields
                    logger.info(f"Sample cut ID: {cut_data.get('id', 'N/A')}")

                    if 'supervisions' in cut_data and cut_data['supervisions']:
                        sup = cut_data['supervisions'][0]
                        logger.info(f"Sample text: {sup.get('text', 'N/A')[:50]}...")

                        if 'custom' in sup:
                            custom = sup['custom']
                            logger.info(f"Custom fields: {', '.join(custom.keys())}")

                            if 'raw_transcript' in custom:
                                logger.info("✓ raw_transcript field present")
                            else:
                                logger.warning("✗ raw_transcript field missing")

                    # Count entries in file
                    entry_count = 1
                    for _ in f:
                        entry_count += 1
                    logger.info(f"Entries in first shard: {entry_count}")

        except Exception as e:
            logger.error(f"Failed to verify cuts file: {e}")

    # Summary
    total_size_mb = sum(os.path.getsize(f) for f in recording_files) / (1024 * 1024)
    logger.info(f"Total recording archive size: {total_size_mb:.2f} MB")
    logger.info("Verification complete")


def main():
    parser = argparse.ArgumentParser(
        description="""
        Convert ASR datasets to Lhotse shar format with multi-shard support.

        Supports three input modes:
        1. JSON manifest (default): --manifest path/to/manifest.json
        2. TXT file pairs: --audio-txt path/to/audio.txt --transcript-txt path/to/transcript.txt
        3. Existing Lhotse shar: --shar-input-dir path/to/shar/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input mode selection
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--input-mode',
        type=str,
        choices=['json', 'txt', 'shar'],
        default='json',
        help='Input format: json (manifest file), txt (audio+transcript pair), or shar (existing lhotse shar). Default: json'
    )

    # JSON mode arguments
    input_group.add_argument(
        '--manifest',
        type=str,
        help='Path to JSON manifest file (required for json mode)'
    )

    # TXT mode arguments
    input_group.add_argument(
        '--audio-txt',
        type=str,
        help='Path to text file with audio filepaths, one per line (required for txt mode)'
    )
    input_group.add_argument(
        '--transcript-txt',
        type=str,
        help='Path to text file with transcripts, one per line (required for txt mode)'
    )

    # SHAR mode arguments
    input_group.add_argument(
        '--shar-input-dir',
        type=str,
        help='Path to directory with existing lhotse shar files (cuts.XXXXXX.jsonl.gz + recording.XXXXXX.tar) (required for shar mode)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save output shar files'
    )
    output_group.add_argument(
        '--shard-size',
        type=int,
        default=50,
        help='Number of entries per shard. Default: 50'
    )
    output_group.add_argument(
        '--audio-format',
        type=str,
        default='flac',
        choices=['flac', 'wav', 'opus'],
        help='Audio format for recording.tar archives. Default: flac'
    )

    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers. Default: CPU count - 1'
    )

    # Logging options
    log_group = parser.add_argument_group('Logging Options')
    log_group.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Custom log filename. Default: lhotse_shar_conversion_TIMESTAMP.log'
    )
    log_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Set up logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    global logger
    logger = setup_logging(args.log_file, log_level)

    # Validate input arguments based on mode
    if args.input_mode == 'json':
        if not args.manifest:
            parser.error("--manifest is required when --input-mode is 'json'")
        if args.audio_txt or args.transcript_txt:
            logger.warning("--audio-txt and --transcript-txt are ignored in json mode")
        if args.shar_input_dir:
            logger.warning("--shar-input-dir is ignored in json mode")

    elif args.input_mode == 'txt':
        if not args.audio_txt or not args.transcript_txt:
            parser.error("Both --audio-txt and --transcript-txt are required when --input-mode is 'txt'")
        if args.manifest:
            logger.warning("--manifest is ignored in txt mode")
        if args.shar_input_dir:
            logger.warning("--shar-input-dir is ignored in txt mode")

    elif args.input_mode == 'shar':
        if not args.shar_input_dir:
            parser.error("--shar-input-dir is required when --input-mode is 'shar'")
        if args.manifest:
            logger.warning("--manifest is ignored in shar mode")
        if args.audio_txt or args.transcript_txt:
            logger.warning("--audio-txt and --transcript-txt are ignored in shar mode")

    logger.info(f"Starting Lhotse shar conversion")
    logger.info(f"Input mode: {args.input_mode}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Shard size: {args.shard_size}")
    logger.info(f"Audio format: {args.audio_format}")
    logger.info(f"Lhotse version: {lhotse.__version__}")

    # Track temporary directory for cleanup
    temp_dir_to_cleanup = None

    try:
        # Read input based on mode
        if args.input_mode == 'json':
            logger.info(f"Reading JSON manifest: {args.manifest}")
            entries = read_json_manifest(args.manifest)

        elif args.input_mode == 'txt':
            logger.info(f"Reading TXT manifest files")
            entries = read_txt_manifest(args.audio_txt, args.transcript_txt)

        elif args.input_mode == 'shar':
            logger.info(f"Reading existing Lhotse shar: {args.shar_input_dir}")
            entries, temp_dir_to_cleanup = read_shar_manifest(args.shar_input_dir)
            logger.info(f"Extracted audio files will be cleaned up after conversion")

        if not entries:
            logger.error("No entries to process. Exiting.")
            return

        # Create shar dataset
        create_lhotse_shar_dataset(
            entries=entries,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            audio_format=args.audio_format,
            num_workers=args.num_workers,
        )

        logger.info("Conversion completed successfully")

    finally:
        # Cleanup temporary directory if it was created
        if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
            logger.info(f"Cleaning up temporary directory: {temp_dir_to_cleanup}")
            try:
                shutil.rmtree(temp_dir_to_cleanup)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory: {e}")


if __name__ == "__main__":
    main()
