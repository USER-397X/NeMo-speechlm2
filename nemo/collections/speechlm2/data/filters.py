#!/usr/bin/env python3
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
Data filtering utilities for NeMo SpeechLM2.

This module provides filtering classes for lhotse Cut objects to enable
quality control and data curation in speech recognition training pipelines.

Features:
- SimilarityFilter: Filter based on audio-text similarity scores from custom metadata
- KeywordFilter: Filter based on exact keywords and regex patterns with probabilistic filtering

All filters follow the callable protocol: filter(cut) -> bool
  Returns True if cut should be KEPT, False if it should be FILTERED OUT.
"""

import logging
import os
import random
import re
from collections import Counter
from typing import List, Optional, Tuple, Union

from lhotse.cut import Cut
from omegaconf import ListConfig, OmegaConf


class SimilarityFilter:
    """
    Filter lhotse Cut objects based on audio-text similarity scores in custom metadata.

    This filter reads the 'similarity' field from cut.supervisions[0].custom and
    filters out cuts with similarity scores below the specified threshold.

    Typical use case: Remove samples where automatic transcription quality is low,
    as indicated by low similarity scores between audio and reference text.

    Args:
        similarity_threshold (float): Minimum similarity score required to keep a cut.
                                     Cuts with similarity < threshold are filtered out.
                                     Default: 0.0 (no filtering)

    Example:
        >>> # Filter out cuts with similarity < 0.2
        >>> similarity_filter = SimilarityFilter(similarity_threshold=0.2)
        >>> filtered_cuts = cuts.filter(similarity_filter)

    Implementation Notes:
    - If 'similarity' field is missing, cut is kept (no filtering)
    - If 'similarity' cannot be converted to float, cut is kept
    - Threshold of 0.0 effectively disables filtering (all cuts kept)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.0,
        log_interval: int = 100000,
        enable_periodic_stats: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.enable_periodic_stats = enable_periodic_stats

        # Statistics tracking
        self.total_cuts_processed = 0
        self.filtered_cuts = 0
        self._log_interval = log_interval

        # Simplified initialization logging (detailed config logged by dataloader)
        if self.similarity_threshold > 0:
            logging.info(f"SimilarityFilter: Ready (threshold < {self.similarity_threshold:.2f} will be filtered)")
        else:
            logging.info("SimilarityFilter: Disabled (threshold=0.0)")

    def __call__(self, cut: Cut) -> bool:
        """
        Evaluate whether a cut should be kept based on similarity score.

        Args:
            cut (Cut): The lhotse Cut object to evaluate

        Returns:
            bool: True if cut should be KEPT, False if it should be FILTERED OUT
        """
        self.total_cuts_processed += 1

        # No filtering if threshold is 0
        if self.similarity_threshold <= 0:
            return True

        # Extract similarity from custom metadata
        try:
            # Check if this is a standard Cut object with supervisions
            if hasattr(cut, 'supervisions') and len(cut.supervisions) > 0:
                meta_dict = cut.supervisions[0].custom
                similarity = meta_dict.get('similarity')

                if similarity is not None:
                    similarity_value = float(similarity)

                    # Filter out if below threshold
                    if similarity_value < self.similarity_threshold:
                        self.filtered_cuts += 1
                        return False
            # If it's a transformed object (NeMoMultimodalConversation), check custom field
            elif hasattr(cut, 'custom') and cut.custom is not None:
                similarity = cut.custom.get('similarity')

                if similarity is not None:
                    similarity_value = float(similarity)

                    # Filter out if below threshold
                    if similarity_value < self.similarity_threshold:
                        self.filtered_cuts += 1
                        return False

        except (AttributeError, ValueError, TypeError, IndexError, KeyError):
            # If we can't get similarity, keep the cut (safe default)
            pass

        # Periodic statistics logging (only if enabled)
        if self.enable_periodic_stats and self.total_cuts_processed % self._log_interval == 0:
            self._log_statistics()

        return True

    def _log_statistics(self):
        """Log filtering statistics periodically (rank-0 only in distributed training)."""
        if self.total_cuts_processed == 0:
            return

        # Only log from rank 0 in distributed training to avoid duplicate logs
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except (ImportError, RuntimeError):
            pass  # Single GPU or no distributed training

        filtered_pct = (self.filtered_cuts / self.total_cuts_processed) * 100
        kept = self.total_cuts_processed - self.filtered_cuts

        logging.info(
            f"[SimilarityFilter] Processed: {self.total_cuts_processed:,} | "
            f"Kept: {kept:,} | Filtered: {self.filtered_cuts:,} ({filtered_pct:.1f}%)"
        )

    def get_statistics(self) -> dict:
        """
        Get filtering statistics.

        Returns:
            dict: Statistics about filtering performance
        """
        filtered_pct = (
            (self.filtered_cuts / self.total_cuts_processed * 100) if self.total_cuts_processed > 0 else 0.0
        )

        return {
            'total_processed': self.total_cuts_processed,
            'filtered': self.filtered_cuts,
            'kept': self.total_cuts_processed - self.filtered_cuts,
            'filtered_percentage': filtered_pct,
        }


class KeywordFilter:
    """
    Filter lhotse Cut objects based on keywords and regex patterns in text.

    This filter supports:
    - Exact keyword matching (e.g., filter out "MBC 뉴스")
    - Regex pattern matching (e.g., filter out "MBC .+ 기자")
    - Probabilistic filtering (e.g., filter "Galaxy" only 25% of the time)

    Each keyword/pattern can have an associated probability (0.0-1.0) that determines
    how likely it will be used for filtering. This enables fine-grained control over
    data curation, allowing gradual reduction of certain patterns without complete removal.

    Args:
        keywords (list, optional): List of exact keywords to filter, where each item is:
                                  - A string: always filters (100% probability)
                                  - A [string, float] pair: filters with given probability
                                  Example: ["빅스비", ["Galaxy", 0.25]]
        patterns (list, optional): List of regex patterns to filter, same format as keywords
                                  Example: [["MBC .+ 기자", 1.0]]
        filter_brackets_except_tokens (bool): If True, filter samples with invalid angle brackets
                                             (experimental feature, requires tokenizer_dir)
        tokenizer_dir (str, optional): Path to tokenizer directory with vocab.txt (for bracket filtering)
        debug_mode (bool): If True, logs detailed filtering information

    Example:
        >>> # Filter "빅스비" always, "Galaxy" 25% of the time
        >>> keyword_filter = KeywordFilter(
        ...     keywords=["빅스비", ["Galaxy", 0.25]],
        ...     patterns=[["MBC .+ 기자", 1.0]]
        ... )
        >>> filtered_cuts = cuts.filter(keyword_filter)

    Probability Behavior:
    - 1.0: Always filter (100% of matches)
    - 0.5: Filter half of matches (50% probability)
    - 0.0: Never filter (0% probability, effectively disabled)

    Implementation Notes:
    - Checks supervision.text field from Cut objects
    - OmegaConf compatible (works with Hydra YAML configs)
    - Thread-safe for filtering, but statistics may have race conditions
    """

    def __init__(
        self,
        keywords: Optional[List[Union[str, List]]] = None,
        patterns: Optional[List[Union[str, List]]] = None,
        filter_brackets_except_tokens: bool = False,
        tokenizer_dir: Optional[str] = None,
        debug_mode: bool = False,
        log_interval: int = 100000,
        enable_periodic_stats: bool = False,
    ):
        self.keywords = []
        self.keyword_probs = {}
        self.patterns = []
        self.pattern_probs = {}
        self.compiled_patterns = []
        self.debug_mode = debug_mode
        self.filter_brackets_except_tokens = filter_brackets_except_tokens
        self.enable_periodic_stats = enable_periodic_stats

        # Handle bracket filtering (experimental)
        if self.filter_brackets_except_tokens:
            if tokenizer_dir is None:
                raise ValueError("tokenizer_dir must be provided when filter_brackets_except_tokens=True")

            vocab_path = os.path.join(tokenizer_dir, 'vocab.txt')
            if not os.path.exists(vocab_path):
                raise ValueError(f"vocab.txt not found at {vocab_path}")

            self.allowed_tokens = self._extract_angle_brackets_with_content(vocab_path)
            logging.info(f"KeywordFilter: loaded {len(self.allowed_tokens)} allowed angle bracket tokens")

        # Convert OmegaConf objects to Python native types
        if isinstance(keywords, ListConfig):
            keywords = OmegaConf.to_container(keywords)
        if isinstance(patterns, ListConfig):
            patterns = OmegaConf.to_container(patterns)

        # Process keywords
        if keywords:
            for item in keywords:
                # Handle OmegaConf nested structures
                if isinstance(item, ListConfig):
                    item = OmegaConf.to_container(item)

                # If item is a list/tuple with 2 elements (keyword, probability)
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        kw, prob = item
                        kw_str = str(kw)

                        # Validate probability
                        if isinstance(prob, (int, float)) and 0.0 <= float(prob) <= 1.0:
                            prob_val = float(prob)
                        else:
                            logging.warning(f"Invalid probability for keyword '{kw_str}': {prob}. Using 1.0")
                            prob_val = 1.0

                        self.keywords.append(kw_str)
                        self.keyword_probs[kw_str] = prob_val
                    except Exception as e:
                        # Fall back to treating the whole item as a keyword
                        item_str = str(item)
                        logging.warning(f"Error processing keyword: {e}. Using '{item_str}' with prob=1.0")
                        self.keywords.append(item_str)
                        self.keyword_probs[item_str] = 1.0
                else:
                    # Simple keyword (string or other value converted to string)
                    item_str = str(item)
                    self.keywords.append(item_str)
                    self.keyword_probs[item_str] = 1.0

        # Process patterns (similar logic as keywords)
        if patterns:
            for item in patterns:
                # Handle OmegaConf nested structures
                if isinstance(item, ListConfig):
                    item = OmegaConf.to_container(item)

                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        pattern, prob = item
                        pattern_str = str(pattern)

                        if isinstance(prob, (int, float)) and 0.0 <= float(prob) <= 1.0:
                            prob_val = float(prob)
                        else:
                            logging.warning(f"Invalid probability for pattern '{pattern_str}': {prob}. Using 1.0")
                            prob_val = 1.0

                        self.patterns.append(pattern_str)
                        self.pattern_probs[pattern_str] = prob_val
                    except Exception as e:
                        item_str = str(item)
                        logging.warning(f"Error processing pattern: {e}. Using '{item_str}' with prob=1.0")
                        self.patterns.append(item_str)
                        self.pattern_probs[item_str] = 1.0
                else:
                    item_str = str(item)
                    self.patterns.append(item_str)
                    self.pattern_probs[item_str] = 1.0

        # Compile regex patterns
        for pattern in self.patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern))
            except (re.error, TypeError) as e:
                logging.warning(f"Invalid regex pattern '{pattern}': {str(e)}")
                self.compiled_patterns.append(None)

        # Statistics tracking
        self.total_cuts_processed = 0
        self.filtered_cuts = 0
        self.skipped_cuts = 0  # Cuts that matched but were kept due to probability
        self.match_counts = Counter()
        self.prob_skipped_counts = Counter()
        self._log_interval = log_interval

        # Simplified configuration logging (detailed config logged by dataloader)
        total = len(self.keywords) + len(self.patterns)
        if total == 0:
            logging.info("KeywordFilter: Disabled (no keywords or patterns)")
            return

        logging.info(f"KeywordFilter: Ready ({len(self.keywords)} keywords, {len(self.patterns)} patterns)")

        # Show sample keywords/patterns in debug mode only
        if self.debug_mode:
            logging.info("  [DEBUG] Sample keywords:")
            for i, keyword in enumerate(self.keywords[:3]):
                prob = self.keyword_probs[keyword]
                logging.info(f"    - '{keyword}' (prob={prob:.2f})")
            if len(self.keywords) > 3:
                logging.info(f"    ... and {len(self.keywords)-3} more")

            if self.patterns:
                logging.info("  [DEBUG] Sample patterns:")
                for i, pattern in enumerate(self.patterns[:3]):
                    prob = self.pattern_probs[pattern]
                    logging.info(f"    - '{pattern}' (prob={prob:.2f})")
                if len(self.patterns) > 3:
                    logging.info(f"    ... and {len(self.patterns)-3} more")

    def _extract_angle_brackets_with_content(self, file_path: str) -> List[str]:
        """
        Extract all angle bracket tokens from vocabulary file.

        Args:
            file_path (str): Path to vocab.txt file

        Returns:
            List[str]: List of allowed angle bracket tokens (e.g., ['<|startoftranscript|>', ...])
        """
        result = []
        pattern = re.compile(r'<.*?>')

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                matches = pattern.findall(line)
                result.extend(matches)

        return result

    def _has_valid_angle_brackets(self, text: str, allowed: List[str]) -> bool:
        """
        Check if text contains only valid angle bracket tokens.

        Args:
            text (str): Text to check
            allowed (List[str]): List of allowed angle bracket tokens

        Returns:
            bool: True if all angle brackets in text are valid, False otherwise
        """
        # Check for mismatched brackets
        if ('<' in text and '>' not in text) or ('>' in text and '<' not in text):
            return False

        # Check for balanced brackets
        if text.count('<') != text.count('>'):
            return False

        # Extract all angle bracket tags
        pattern = r'<[^>]*>'
        tags = re.findall(pattern, text)

        # Check if all tags are in allowed list
        for tag in tags:
            if tag not in allowed:
                return False

        return True

    def __call__(self, cut: Cut) -> bool:
        """
        Evaluate whether a cut should be kept based on keyword/pattern matching.

        Implements probabilistic filtering where each keyword/pattern has its own
        probability of being applied.

        Args:
            cut (Cut): The lhotse Cut object to evaluate

        Returns:
            bool: True if cut should be KEPT, False if it should be FILTERED OUT
        """
        self.total_cuts_processed += 1

        # No filtering if no keywords or patterns configured
        if not self.keywords and not self.patterns:
            return True

        # Extract text based on object type
        texts = []
        if hasattr(cut, 'supervisions'):
            # Standard Cut object
            texts = [sup.text for sup in cut.supervisions if sup.text is not None]
        elif hasattr(cut, 'turns'):
            # Transformed NeMoMultimodalConversation object
            for turn in cut.turns:
                # Import here to avoid circular dependency
                from nemo.collections.common.data.lhotse.text_adapters import TextTurn, AudioTurn
                if isinstance(turn, TextTurn):
                    texts.append(turn.value)
                elif isinstance(turn, AudioTurn) and turn.text:
                    texts.append(turn.text)

        # If no text found, keep the cut
        if not texts:
            return True

        # Check all texts from the cut
        for text in texts:
            if text is None:
                continue

            # Check bracket filtering if enabled
            if self.filter_brackets_except_tokens:
                if self._has_valid_angle_brackets(text, self.allowed_tokens):
                    continue  # Valid brackets, check next text
                else:
                    # Invalid brackets, filter out this cut
                    self.filtered_cuts += 1
                    if self.debug_mode:
                        logging.debug(f"Filtered (invalid angle brackets): '{text[:100]}'")
                    return False

            # Check exact keywords
            for keyword in self.keywords:
                if keyword in text:
                    # Check probability for this keyword
                    prob = self.keyword_probs[keyword]
                    if prob < 1.0 and random.random() > prob:
                        # Probabilistically skip this match
                        if self.debug_mode:
                            logging.debug(
                                f"Probabilistically skipped filter for keyword '{keyword}' (prob={prob:.2f})"
                            )
                        self.prob_skipped_counts[f"keyword:{keyword}"] += 1
                        self.skipped_cuts += 1
                        continue

                    # Apply filter (keyword matched and probability check passed)
                    self._log_filtered_sample(text, keyword, "keyword", prob)
                    return False

            # Check regex patterns
            for i, pattern_re in enumerate(self.compiled_patterns):
                if pattern_re is None:
                    continue

                if pattern_re.search(text):
                    pattern = self.patterns[i]
                    # Check probability for this pattern
                    prob = self.pattern_probs[pattern]
                    if prob < 1.0 and random.random() > prob:
                        # Probabilistically skip this match
                        if self.debug_mode:
                            logging.debug(
                                f"Probabilistically skipped filter for pattern '{pattern}' (prob={prob:.2f})"
                            )
                        self.prob_skipped_counts[f"pattern:{pattern}"] += 1
                        self.skipped_cuts += 1
                        continue

                    # Apply filter (pattern matched and probability check passed)
                    self._log_filtered_sample(text, pattern, "pattern", prob)
                    return False

        # Log statistics periodically (only if enabled)
        if self.enable_periodic_stats and self.total_cuts_processed % self._log_interval == 0:
            self._log_statistics()

        return True

    def _log_filtered_sample(self, text: str, matched_pattern: str, match_type: str, prob: float):
        """
        Log information about a filtered sample.

        Args:
            text (str): The text that was filtered
            matched_pattern (str): The keyword or regex pattern that caused filtering
            match_type (str): Type of match ('keyword' or 'pattern')
            prob (float): The probability value that was used for filtering
        """
        self.filtered_cuts += 1
        self.match_counts[matched_pattern] += 1

        if self.debug_mode:
            # Truncate text if too long
            display_text = text[:100] + "..." if len(text) > 100 else text
            logging.debug(f"Filtered ({match_type} '{matched_pattern}', prob={prob:.2f}): '{display_text}'")

    def _log_statistics(self):
        """Log filtering statistics periodically (rank-0 only in distributed training)."""
        if self.total_cuts_processed == 0:
            return

        # Only log from rank 0 in distributed training to avoid duplicate logs
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except (ImportError, RuntimeError):
            pass  # Single GPU or no distributed training

        filtered_pct = (self.filtered_cuts / self.total_cuts_processed) * 100
        kept = self.total_cuts_processed - self.filtered_cuts
        total_matches = self.filtered_cuts + self.skipped_cuts
        skipped_pct = (self.skipped_cuts / max(1, total_matches)) * 100 if total_matches > 0 else 0

        logging.info(
            f"[KeywordFilter] Processed: {self.total_cuts_processed:,} | "
            f"Kept: {kept:,} | Filtered: {self.filtered_cuts:,} ({filtered_pct:.1f}%) | "
            f"Prob-skipped: {self.skipped_cuts:,} ({skipped_pct:.1f}%)"
        )

        # Show top matches in debug mode only
        if self.debug_mode and self.filtered_cuts > 0:
            top_matches = self.match_counts.most_common(3)
            if top_matches:
                matches_str = ', '.join(f'{k}({v})' for k, v in top_matches)
                logging.info(f"  [DEBUG] Top matches: {matches_str}")

    def get_statistics(self) -> dict:
        """
        Get filtering statistics.

        Returns:
            dict: Detailed statistics about filtering performance
        """
        filtered_pct = (
            (self.filtered_cuts / self.total_cuts_processed * 100) if self.total_cuts_processed > 0 else 0.0
        )

        total_matches = self.filtered_cuts + self.skipped_cuts
        skipped_pct = (self.skipped_cuts / max(1, total_matches)) * 100 if total_matches > 0 else 0.0

        return {
            'total_processed': self.total_cuts_processed,
            'filtered': self.filtered_cuts,
            'kept': self.total_cuts_processed - self.filtered_cuts,
            'filtered_percentage': filtered_pct,
            'probabilistically_skipped': self.skipped_cuts,
            'skipped_percentage': skipped_pct,
            'top_matches': dict(self.match_counts.most_common(10)),
        }
