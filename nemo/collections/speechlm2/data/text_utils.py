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
Text utility functions for SALM dataset processing.

This module provides helper functions for priority-based reference text selection
from lhotse shar custom metadata fields, adapted from se-trainer implementation.
"""

import re
from typing import Dict, Any, Union


def get_whisper_result(custom_dict: Dict[str, Any]) -> str:
    """
    Safely extract whisper_result from custom metadata field.

    Handles both dictionary format and string format for backward compatibility.

    Args:
        custom_dict: Custom metadata dictionary from lhotse Cut

    Returns:
        str: Whisper result text, or empty string if not found

    Examples:
        >>> get_whisper_result({"whisper_result": "Hello world"})
        'Hello world'
        >>> get_whisper_result({"whisper_result": {"text": "Hello"}})
        'Hello'
        >>> get_whisper_result({})
        ''
    """
    if not custom_dict:
        return ""

    whisper_result = custom_dict.get("whisper_result", "")

    # Handle dict format: {"text": "..."}
    if isinstance(whisper_result, dict):
        return whisper_result.get("text", "")

    # Handle string format directly
    return whisper_result if isinstance(whisper_result, str) else ""


def count_alphanumeric(text: str) -> int:
    """
    Count alphanumeric characters in text.

    Used to compare information richness between different text sources
    (e.g., whisper_result vs itn) for conditional selection.

    Args:
        text: Input text string

    Returns:
        int: Number of alphanumeric characters

    Examples:
        >>> count_alphanumeric("Hello, world!")
        10
        >>> count_alphanumeric("안녕하세요 123")
        3
        >>> count_alphanumeric("")
        0
    """
    if not text:
        return 0
    return sum(c.isalnum() for c in text)


def add_punctuation(text: str, whisper_result: str, lang: str) -> str:
    """
    Add punctuation marks to the input text based on the whisper result.

    Adapted from se-trainer implementation. This function adds appropriate
    punctuation based on language and whisper_result endings.

    Args:
        text: Input text to add punctuation to
        whisper_result: Whisper ASR result used as reference for punctuation
        lang: Language code (e.g., 'en-US', 'ko-KR', 'zh-CN')

    Returns:
        str: Text with added punctuation

    Examples:
        >>> add_punctuation("Hello world", "Hello world.", "en-US")
        'Hello world.'
        >>> add_punctuation("안녕하세요", "안녕하세요。", "ko-KR")
        '안녕하세요.'

    Notes:
        - Respects existing punctuation (won't add if already present)
        - Language-specific punctuation rules:
          * ko, en, de, fr, es, it, pt, pl, vi: Use '.' or '?'
          * zh, ja: Use '。' or '？'
          * hi: Use '।'
          * th: No punctuation added
    """
    if not text or not whisper_result:
        return text

    # Extract language code (first 2 characters)
    lang_code = lang[:2] if len(lang) >= 2 else lang

    # Remove invalid Unicode surrogate characters
    text = re.sub(r"[\ud800-\udfff]", "", text)

    # Skip if text is empty or Thai (Thai doesn't typically use punctuation)
    if len(text) == 0 or lang_code == "th":
        return text

    # European languages + Korean + Vietnamese
    elif lang_code in ["ko", "en", "de", "fr", "es", "it", "pt", "pl", "vi"]:
        # Check if text already has ending punctuation
        if text[-1] in [".", "?", "!", ","]:
            return text

        # Add question mark if whisper_result ends with '?'
        if whisper_result.endswith("?"):
            # Spanish special case: add inverted question mark at the beginning
            if lang_code == "es":
                if whisper_result.startswith("¿"):
                    return "¿" + text + "?"
            else:
                return text + "?"
        else:
            # Add period by default
            return text + "."

    # Chinese and Japanese
    elif lang_code in ["zh", "ja"]:
        # Check if text already has CJK ending punctuation
        if text[-1] in ["。", "？", "！", "?", "!"]:
            return text

        # Add CJK punctuation if whisper_result ends with it
        if whisper_result[-1] in ["。", "？", "?"]:
            return text + whisper_result[-1]

    # Hindi
    elif lang_code == "hi":
        # Check if text already has Hindi ending punctuation
        if text[-1] in ["।", "?", "!"]:
            return text

        # Add Hindi punctuation if whisper_result ends with it
        if whisper_result[-1] in ["।", "?"]:
            return text + whisper_result[-1]

    return text


def get_reference_text_with_priority(
    cut,
    use_itn: bool = True,
    use_whisper_result: bool = False,
    convert_all_uppercase_to_lowercase: bool = True,
    capitalize_first_letter: bool = True
) -> str:
    """
    Get reference text with configurable priority-based selection from custom metadata fields.

    This function implements the text selection logic from se-trainer, which uses
    a priority hierarchy to select the best available reference text from multiple
    sources in the lhotse shar custom metadata.

    **KEY BEHAVIOR (se-trainer exact match):**
    - When use_itn=True: Uses custom metadata (ground_truth_transcript → itn → whisper_result)
    - When use_itn=False: Starts from supervisions.text, only checks whisper_result conditionally

    Priority hierarchy when use_itn=True:
        1. custom.ground_truth_transcript - Manually verified text (if present)
        2. custom.itn - Inverse Text Normalization result (overwrites priority 1)
        3. custom.whisper_result - Whisper ASR output (if alphanumeric count >= current)
        4. supervisions[0].text - Fallback

    Priority hierarchy when use_itn=False:
        1. supervisions[0].text - Start from original manifest text
        2. custom.whisper_result - Only if richer than supervisions.text (when use_whisper_result=True)

    Args:
        cut: Lhotse Cut object with optional custom metadata fields
        use_itn: Enable ITN text usage (default: True)
                 When False, starts from supervisions.text instead of custom metadata
        use_whisper_result: Enable whisper_result usage (default: False)
                           When False, skips whisper_result entirely
        convert_all_uppercase_to_lowercase: Convert all-uppercase text to lowercase (default: False)
                                           Example: "HELLO WORLD" → "hello world"
        capitalize_first_letter: Capitalize first letter if all lowercase (default: False)
                                Example: "hello world" → "Hello world"

    Returns:
        str: Selected reference text based on priority logic and flags

    Examples:
        >>> # Case 1: use_itn=True, use_whisper_result=True
        >>> # Uses full priority: ground_truth_transcript → itn → whisper_result
        >>> cut.custom = {
        ...     "ground_truth_transcript": "Hello world",
        ...     "itn": "hello world",
        ...     "whisper_result": "hello world example"
        ... }
        >>> get_reference_text_with_priority(cut, use_itn=True, use_whisper_result=True)
        'hello world example'  # Whisper result is richer than ITN

        >>> # Case 2: use_itn=False, use_whisper_result=True
        >>> # Starts from supervisions.text, checks whisper_result
        >>> cut.supervisions[0].text = "fallback text"
        >>> cut.custom = {
        ...     "ground_truth_transcript": "Hello world",
        ...     "itn": "hello world",
        ...     "whisper_result": "fallback text with more content"
        ... }
        >>> get_reference_text_with_priority(cut, use_itn=False, use_whisper_result=True)
        'fallback text with more content'  # Whisper result richer than supervisions.text

        >>> # Case 3: use_itn=True, use_whisper_result=False
        >>> # Uses ground_truth_transcript → itn only
        >>> cut.custom = {
        ...     "ground_truth_transcript": "Hello world",
        ...     "itn": "hello world",
        ...     "whisper_result": "hello world example"
        ... }
        >>> get_reference_text_with_priority(cut, use_itn=True, use_whisper_result=False)
        'hello world'  # Whisper result skipped, ITN selected

        >>> # Case 4: use_itn=False, use_whisper_result=False
        >>> # Only uses supervisions.text
        >>> cut.supervisions[0].text = "fallback text"
        >>> cut.custom = {...}  # All custom fields ignored
        >>> get_reference_text_with_priority(cut, use_itn=False, use_whisper_result=False)
        'fallback text'  # Only supervisions.text used

        >>> # Case 5: No custom fields, use supervisions fallback
        >>> cut.custom = {}
        >>> cut.supervisions = [type('obj', (object,), {'text': 'fallback text'})()]
        >>> get_reference_text_with_priority(cut)
        'fallback text'

    Notes:
        - **EXACT se-trainer match**: use_itn controls starting point, not just itn field
        - Defaults: use_itn=True (enable ITN priority), use_whisper_result=False (disable Whisper by default)
        - If custom fields don't exist, falls back to supervisions[0].text
        - Empty strings are treated as missing values
        - The alphanumeric count comparison ensures we prefer more informative text
    """
    # Get fallback text from supervisions
    fallback_text = cut.supervisions[0].text if cut.supervisions else ""

    # Check if custom field exists
    if not hasattr(cut, 'custom') or not cut.custom:
        return fallback_text

    custom = cut.custom

    # === KEY LOGIC: use_itn determines the starting point ===
    if use_itn:
        # use_itn=True: Priority-based selection from custom metadata
        selected_text = fallback_text

        # Priority 1: ground_truth_transcript (if present)
        if 'ground_truth_transcript' in custom and custom['ground_truth_transcript']:
            selected_text = custom['ground_truth_transcript']

        # Priority 2: itn (overwrites priority 1 if present)
        if 'itn' in custom and custom['itn']:
            selected_text = custom['itn']

        # Priority 3: whisper_result (conditional on alphanumeric count and flag)
        if use_whisper_result:
            whisper_result = get_whisper_result(custom)
            if whisper_result and selected_text:
                # Use whisper_result if it's richer in content
                if count_alphanumeric(whisper_result) >= count_alphanumeric(selected_text):
                    selected_text = whisper_result
            elif whisper_result and not selected_text:
                # If no other text available, use whisper_result
                selected_text = whisper_result

    else:
        # use_itn=False: Start from supervisions.text (ignore custom metadata for ITN)
        selected_text = fallback_text

        # Only check whisper_result if use_whisper_result=True
        if use_whisper_result:
            whisper_result = get_whisper_result(custom)
            if whisper_result:
                # Use whisper_result if it's richer than supervisions.text
                if count_alphanumeric(whisper_result) >= count_alphanumeric(selected_text):
                    selected_text = whisper_result

    # Apply text case normalization before returning
    final_text = selected_text if selected_text else fallback_text
    final_text = apply_text_case_normalization(
        final_text,
        convert_all_uppercase_to_lowercase=convert_all_uppercase_to_lowercase,
        capitalize_first_letter=capitalize_first_letter
    )

    return final_text


def apply_text_case_normalization(
    text: str,
    convert_all_uppercase_to_lowercase: bool = True,
    capitalize_first_letter: bool = True
) -> str:
    """
    Apply text case normalization rules to training data.

    This function implements two optional text case transformations that can improve
    model training by normalizing inconsistent text capitalization patterns.

    Args:
        text: Input text string to normalize
        convert_all_uppercase_to_lowercase: If True, converts all-uppercase text to lowercase
                                            Example: "HELLO WORLD" → "hello world"
        capitalize_first_letter: If True, capitalizes first letter if text is all lowercase
                                Example: "hello world" → "Hello world"

    Returns:
        str: Normalized text string

    Examples:
        >>> # Option 1: Convert all uppercase to lowercase
        >>> apply_text_case_normalization("HELLO WORLD", convert_all_uppercase_to_lowercase=True)
        'hello world'

        >>> # Option 2: Capitalize first letter
        >>> apply_text_case_normalization("hello world", capitalize_first_letter=True)
        'Hello world'

        >>> # Both options enabled (order: uppercase→lowercase, then capitalize)
        >>> apply_text_case_normalization("HELLO WORLD",
        ...     convert_all_uppercase_to_lowercase=True,
        ...     capitalize_first_letter=True)
        'Hello world'

        >>> # No transformation if text is mixed case
        >>> apply_text_case_normalization("Hello World", capitalize_first_letter=True)
        'Hello World'

        >>> # Empty strings are handled safely
        >>> apply_text_case_normalization("", capitalize_first_letter=True)
        ''

    Notes:
        - Only processes non-empty strings
        - Checks are case-sensitive (isupper(), islower())
        - Mixed-case text is not transformed
        - Both options can be enabled simultaneously
        - Order: uppercase→lowercase first, then capitalize
        - Matches se-trainer implementation exactly
    """
    if not text:  # Handle empty strings safely
        return text

    # Option 1: Convert all uppercase to lowercase
    if convert_all_uppercase_to_lowercase and text.isupper():
        text = text.lower()

    # Option 2: Capitalize first letter if all lowercase
    if capitalize_first_letter and text.islower():
        text = text[0].upper() + text[1:] if len(text) > 0 else text

    return text
