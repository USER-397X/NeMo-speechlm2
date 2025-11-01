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
WER/CER Calculator abstraction for SALM models.

This module provides a unified interface for calculating Word Error Rate (WER) and
Character Error Rate (CER) with support for different normalization strategies:

1. LegacyWERCalculator: Simple editdistance.eval() without normalization (backward compatible)
2. OpenASRLeaderboardWERCalculator: HuggingFace Open ASR Leaderboard compliant with:
   - Text normalization (contractions, numbers, punctuation)
   - Empty string handling (4 categories)
   - Multilingual support (English vs non-English)
   - Exact error counting via jiwer

Usage:
    # Legacy mode (default)
    calculator = create_wer_calculator(normalizer="legacy")
    error_rate, stats, norm_preds, norm_refs = calculator.calculate(predictions, references)

    # Open ASR Leaderboard mode
    calculator = create_wer_calculator(normalizer="open_asr_leaderboard")
    error_rate, stats, norm_preds, norm_refs = calculator.calculate(
        predictions, references, language="en", metric_type="wer"
    )
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class WERCalculatorBase(ABC):
    """
    Base class for WER/CER calculators.

    All calculator implementations must inherit from this class and implement
    the calculate() method.
    """

    @abstractmethod
    def calculate(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en",
        metric_type: str = "wer",
    ) -> Tuple[float, Dict, List[str], List[str]]:
        """
        Calculate error rate with optional normalization.

        Args:
            predictions: List of hypothesis strings (model outputs)
            references: List of reference strings (ground truth)
            language: Language code for normalization
                - 'en', 'en-US', 'en-GB' → English normalization
                - Other codes → Multilingual normalization
            metric_type: Type of metric to compute
                - 'wer': Word Error Rate (word-level)
                - 'cer': Character Error Rate (character-level)

        Returns:
            Tuple of (error_rate, stats_dict, normalized_predictions, normalized_references) where:
                - error_rate: Float between 0.0 and 1.0+ (can exceed 1.0 with many insertions)
                - stats_dict: Dictionary with calculation details
                    - 'total_errors': Total number of errors (S+D+I)
                    - 'total_units': Total number of reference units (words or characters)
                    - 'substitutions': Number of substitution errors
                    - 'deletions': Number of deletion errors
                    - 'insertions': Number of insertion errors
                    - Additional calculator-specific statistics
                - normalized_predictions: List of normalized prediction strings (after text cleaning)
                - normalized_references: List of normalized reference strings (after text cleaning)
        """
        pass


class LegacyWERCalculator(WERCalculatorBase):
    """
    Legacy WER/CER calculator using simple editdistance.eval().

    This implementation provides backward compatibility with the original SALM WER calculation.
    It does NOT perform text normalization and has limited empty string handling.

    Features:
    - Simple word/character splitting
    - Direct editdistance calculation
    - Basic empty reference handling
    - Chat template role marker removal (for accurate ASR metrics)

    Limitations:
    - No normalization (punctuation, contractions, numbers preserved as-is)
    - Limited empty string handling
    - Less accurate than Open ASR Leaderboard protocol
    """

    def calculate(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en",
        metric_type: str = "wer",
    ) -> Tuple[float, Dict, List[str], List[str]]:
        import editdistance
        from .normalization import remove_chat_template_roles

        total_errors = 0
        total_units = 0

        # Clean predictions and references by removing chat template role markers
        # This ensures WER/CER only measures ASR transcription accuracy, not prompt tokens
        cleaned_predictions = [remove_chat_template_roles(p) for p in predictions]
        cleaned_references = [remove_chat_template_roles(r) for r in references]

        for pred, ref in zip(cleaned_predictions, cleaned_references):
            # Split into units (words for WER, characters for CER)
            if metric_type == "wer":
                pred_units = pred.split()
                ref_units = ref.split()
            else:  # cer
                pred_units = list(pred)
                ref_units = list(ref)

            # Calculate edit distance
            if len(ref_units) > 0:
                total_errors += editdistance.eval(pred_units, ref_units)
                total_units += len(ref_units)

        # Compute error rate
        error_rate = total_errors / total_units if total_units > 0 else 0.0

        stats = {
            "total_errors": total_errors,
            "total_units": total_units,
            "substitutions": 0,  # Legacy mode doesn't track detailed breakdown
            "deletions": 0,
            "insertions": 0,
            "normalizer": "legacy",
            "metric_type": metric_type,
        }

        # Return cleaned text (with chat template markers removed)
        return error_rate, stats, cleaned_predictions, cleaned_references


class OpenASRLeaderboardWERCalculator(WERCalculatorBase):
    """
    HuggingFace Open ASR Leaderboard compliant WER/CER calculator.

    This implementation follows the official Open ASR Leaderboard protocol with:
    - Text normalization (English and multilingual)
    - Proper empty string handling (4 categories)
    - Exact error counting via jiwer
    - Support for both WER and CER

    Language Support:
    - English ('en', 'en-US', 'en-GB', etc.):
        - Contraction expansion (won't → will not)
        - Number normalization (twenty one → 21)
        - British-American spelling (colour → color)
        - Punctuation removal
    - Multilingual (all other languages):
        - Diacritic removal
        - Punctuation removal
        - Unicode normalization

    Empty String Handling:
    1. Both empty → 0.0 error (correct silence)
    2. Ref empty, pred not → All prediction units are insertions
    3. Ref not, pred empty → All reference units are deletions
    4. Both not empty → Normal WER/CER calculation
    """

    def __init__(self):
        # Import normalizers and utilities from local metrics modules
        try:
            from .normalization import BasicMultilingualTextNormalizer, EnglishTextNormalizer
            from .wer_utils import calculate_wer_with_empty_handling

            self.en_normalizer = EnglishTextNormalizer()
            self.ml_normalizer = BasicMultilingualTextNormalizer()
            self.calculate_wer_fn = calculate_wer_with_empty_handling

            logger.info("OpenASRLeaderboardWERCalculator initialized successfully")

        except ImportError as e:
            logger.error(
                f"Failed to import Open ASR Leaderboard modules: {e}\n"
                f"Please ensure nemo/collections/speechlm2/metrics/ contains:\n"
                f"  - normalization.py\n"
                f"  - wer_utils.py"
            )
            raise

    def calculate(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en",
        metric_type: str = "wer",
    ) -> Tuple[float, Dict, List[str], List[str]]:
        # Map language code to normalizer selection
        lang_code = self._map_language(language)

        # Select appropriate normalizer
        normalizer = self.en_normalizer if lang_code == "en" else self.ml_normalizer

        # Apply normalization (preserve empty strings)
        norm_preds = [normalizer(p) if p.strip() else "" for p in predictions]
        norm_refs = [normalizer(r) if r.strip() else "" for r in references]

        # Calculate WER/CER with proper empty string handling (now returns S/D/I breakdown)
        error_rate, empty_stats, total_errors, total_units, substitutions, deletions, insertions = self.calculate_wer_fn(
            norm_preds, norm_refs, metric_type
        )

        # Prepare statistics with detailed error breakdown
        stats = {
            "total_errors": total_errors,
            "total_units": total_units,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "normalizer": "open_asr_leaderboard",
            "language": lang_code,
            "metric_type": metric_type,
            "empty_handling": empty_stats,
        }

        # Return normalized text for logging transparency
        return error_rate, stats, norm_preds, norm_refs

    def _map_language(self, lang: str) -> str:
        """
        Map language code to normalizer selection (en vs multilingual).

        Args:
            lang: Language code (e.g., 'en', 'en-US', 'ko-KR', 'ja-JP', 'zh-CN')

        Returns:
            'en' for English variants, 'multilingual' for all other languages
        """
        # English language codes
        en_langs = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-NZ", "en-IN"]

        return "en" if lang in en_langs else "multilingual"


def create_wer_calculator(normalizer: str = "legacy") -> WERCalculatorBase:
    """
    Factory function to create WER calculator instance.

    Args:
        normalizer: Calculator type to create
            - "legacy": Simple editdistance.eval() (default, backward compatible)
            - "open_asr_leaderboard": HuggingFace Open ASR Leaderboard protocol

    Returns:
        WERCalculatorBase instance

    Raises:
        ValueError: If normalizer type is not recognized

    Examples:
        # Legacy mode (default)
        >>> calculator = create_wer_calculator()
        >>> error_rate, stats = calculator.calculate(predictions, references)

        # Open ASR Leaderboard mode
        >>> calculator = create_wer_calculator("open_asr_leaderboard")
        >>> error_rate, stats = calculator.calculate(
        ...     predictions, references, language="ko-KR", metric_type="wer"
        ... )
    """
    if normalizer == "open_asr_leaderboard":
        logger.info("Creating OpenASRLeaderboardWERCalculator")
        return OpenASRLeaderboardWERCalculator()
    elif normalizer == "legacy":
        logger.info("Creating LegacyWERCalculator (backward compatible mode)")
        return LegacyWERCalculator()
    else:
        raise ValueError(
            f"Unknown normalizer: {normalizer}. "
            f"Must be 'legacy' or 'open_asr_leaderboard'"
        )
