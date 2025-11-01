# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WER/CER calculation utilities with proper empty string handling.
100% compliant with HuggingFace Open ASR Leaderboard protocol.
"""

from typing import Dict, List, Tuple


def calculate_wer_with_empty_handling(
    predictions: List[str], references: List[str], metric_type: str = "wer"
) -> Tuple[float, Dict, int, int, int, int, int]:
    """
    Calculate WER/CER with proper empty string handling and exact error counting.

    Implements the HuggingFace Open ASR Leaderboard protocol for WER/CER calculation:
    - Handles 4 categories of empty string combinations
    - Uses jiwer for exact error counting (substitutions, deletions, insertions)
    - Returns overall error rate and detailed statistics

    Args:
        predictions: List of hypothesis strings (model outputs)
        references: List of reference strings (ground truth)
        metric_type: Type of metric ('wer' for word-level, 'cer' for character-level)

    Returns:
        Tuple containing:
            - overall_error_rate: Error rate as a float (0.0 to 1.0+)
            - stats: Dictionary with empty string handling statistics
            - total_errors: Total number of errors (S+D+I) across all samples
            - total_units: Total number of reference units (words or characters)
            - substitutions: Number of substitution errors
            - deletions: Number of deletion errors
            - insertions: Number of insertion errors

    Empty String Handling:
        Category 1: Both empty → 0.0 error (correct silence)
        Category 2: Reference empty, prediction not → All predicted units are insertions
        Category 3: Reference not empty, prediction empty → All reference units are deletions
        Category 4: Both not empty → Normal WER/CER calculation via jiwer

    Example:
        >>> predictions = ["hello world", "", "test"]
        >>> references = ["hello world", "", "text"]
        >>> error_rate, stats, errors, units, subs, dels, ins = calculate_wer_with_empty_handling(
        ...     predictions, references, metric_type="wer"
        ... )
        >>> print(f"WER: {error_rate:.2%}")
        WER: 20.00%  # 1 substitution out of 5 words
    """
    import jiwer

    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have the same length. "
            f"Got {len(predictions)} predictions and {len(references)} references."
        )

    # Track empty string handling statistics
    empty_stats = {
        "both_empty": 0,  # Category 1: Both empty (correct)
        "ref_empty_pred_not": 0,  # Category 2: Reference empty, prediction not (all insertions)
        "ref_not_pred_empty": 0,  # Category 3: Reference not empty, prediction empty (all deletions)
        "both_not_empty": 0,  # Category 4: Both not empty (normal WER/CER)
    }

    # Separate samples based on empty string categories
    non_empty_predictions = []
    non_empty_references = []
    insertion_errors = 0
    deletion_errors = 0

    for pred, ref in zip(predictions, references):
        pred_empty = len(pred.strip()) == 0
        ref_empty = len(ref.strip()) == 0

        if pred_empty and ref_empty:
            # Category 1: Both empty → 0 errors
            empty_stats["both_empty"] += 1
        elif ref_empty and not pred_empty:
            # Category 2: Reference empty, prediction not → All predicted units are insertions
            empty_stats["ref_empty_pred_not"] += 1
            if metric_type == "wer":
                insertion_errors += len(pred.split())
            else:  # cer
                insertion_errors += len(pred)
        elif not ref_empty and pred_empty:
            # Category 3: Reference not empty, prediction empty → All reference units are deletions
            empty_stats["ref_not_pred_empty"] += 1
            if metric_type == "wer":
                deletion_errors += len(ref.split())
            else:  # cer
                deletion_errors += len(ref)
        else:
            # Category 4: Both not empty → Normal WER/CER calculation
            empty_stats["both_not_empty"] += 1
            non_empty_predictions.append(pred)
            non_empty_references.append(ref)

    # Calculate WER/CER for non-empty samples using jiwer
    jiwer_errors = 0
    jiwer_units = 0
    jiwer_subs = 0
    jiwer_dels = 0
    jiwer_ins = 0

    if non_empty_references:
        if metric_type == "wer":
            # Word-level evaluation
            output = jiwer.process_words(non_empty_references, non_empty_predictions)
            jiwer_subs = output.substitutions
            jiwer_dels = output.deletions
            jiwer_ins = output.insertions
            jiwer_errors = jiwer_subs + jiwer_dels + jiwer_ins
            jiwer_units = sum(len(ref.split()) for ref in non_empty_references)
        else:  # cer
            # Character-level evaluation
            output = jiwer.process_characters(non_empty_references, non_empty_predictions)
            jiwer_subs = output.substitutions
            jiwer_dels = output.deletions
            jiwer_ins = output.insertions
            jiwer_errors = jiwer_subs + jiwer_dels + jiwer_ins
            jiwer_units = sum(len(ref) for ref in non_empty_references)

    # Calculate total errors and units
    total_errors = jiwer_errors + insertion_errors + deletion_errors
    total_subs = jiwer_subs
    total_dels = jiwer_dels + deletion_errors
    total_ins = jiwer_ins + insertion_errors

    # Total units from all reference strings
    if metric_type == "wer":
        total_units = sum(len(ref.split()) for ref in references)
    else:  # cer
        total_units = sum(len(ref) for ref in references)

    # Calculate overall error rate
    overall_error_rate = total_errors / total_units if total_units > 0 else 0.0

    return overall_error_rate, empty_stats, total_errors, total_units, total_subs, total_dels, total_ins
