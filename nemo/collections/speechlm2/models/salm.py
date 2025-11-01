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
import warnings
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any, Optional, Union

import editdistance
import torch
from lhotse import CutSet
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
from torch import Tensor
# from torch.distributed.fsdp import fully_shard
try:
    from torch.distributed.fsdp import fully_shard
except Exception:
    from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import GenerationConfig

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_speech_encoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging


class SALM(LightningModule, HFHubMixin):
    def __init__(self, cfg, skip_perception_setup: bool = False) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALM as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights)
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        # Skip during restore_from() since weights will be loaded via load_state_dict()
        if not skip_perception_setup:
            setup_speech_encoder(self, pretrained_weights=self.cfg.pretrained_weights)

        self._use_fsdp = False
        self._use_tp = False

        # Validation prediction logging configuration
        self.log_prediction_valid = cfg.get('log_prediction_valid', False)
        self.log_prediction_valid_samples = cfg.get('log_prediction_valid_samples', 1)

        # Training prediction logging configuration (separate from validation)
        self.log_prediction_train = cfg.get('log_prediction_train', False)
        self.log_prediction_train_interval = cfg.get('log_prediction_train_interval', 100)
        self.log_prediction_train_samples = cfg.get('log_prediction_train_samples', 1)

        # WER Calculator configuration
        wer_calc_cfg = cfg.get('wer_calculator', {})
        normalizer = wer_calc_cfg.get('normalizer', 'legacy')

        from nemo.collections.speechlm2.metrics.wer_calculator import create_wer_calculator
        self.wer_calculator = create_wer_calculator(normalizer)

        logging.info(f"WER Calculator initialized: {normalizer}")

    def _format_prediction_log(
        self,
        idx,
        prompt,
        reference,
        prediction,
        reference_normalized,
        prediction_normalized,
        wer,
        wer_stats,
        language="en",
        step_type="Validation",
        dataset_name="",
        audio_len_samples=None,
        metric_type="wer"
    ):
        """
        Format prediction log with enhanced information including normalized text and detailed error breakdown.

        Args:
            idx: Sample index
            prompt: Input prompt (may contain audio locator tags and formatting)
            reference: Ground truth text (raw)
            prediction: Model prediction (raw)
            reference_normalized: Normalized ground truth text
            prediction_normalized: Normalized prediction text
            wer: Word error rate (or CER if metric_type='cer')
            wer_stats: Statistics dict from WER calculator with error breakdown
            language: Language code for this sample
            step_type: "Training" or "Validation"
            dataset_name: Name of dataset (for validation)
            audio_len_samples: Audio length in samples (optional, for audio token count estimation)
            metric_type: Type of metric ('wer' or 'cer')

        Returns:
            List of strings, each a clean single-line log entry
        """
        lines = []

        # Helper function to clean text for single-line display
        def clean_text_for_display(text, max_length=120):
            """
            Clean text for display by removing newlines and control characters.

            Args:
                text: Input text to clean
                max_length: Maximum display length before truncation

            Returns:
                Cleaned, single-line text suitable for logging
            """
            if not text:
                return ""

            # Replace all newline characters with space
            cleaned = text.replace('\n', ' ').replace('\r', ' ')

            # Replace tab characters with space
            cleaned = cleaned.replace('\t', ' ')

            # Remove other control characters (ASCII 0-31 except space)
            import re
            cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', cleaned)

            # Compress multiple consecutive spaces into single space
            cleaned = re.sub(r'\s+', ' ', cleaned)

            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()

            # Truncate if too long
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + "..."

            return cleaned

        # Language display name mapping
        lang_names = {
            "en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese",
            "de": "German", "es": "Spanish", "fr": "French", "hi": "Hindi",
            "it": "Italian", "pl": "Polish", "pt": "Portuguese",
            "th": "Thai", "vi": "Vietnamese"
        }
        lang_code = language.split('-')[0] if '-' in language else language
        lang_name = lang_names.get(lang_code, language.upper())

        # Determine normalizer type from stats
        normalizer_type = wer_stats.get('normalizer', 'legacy')
        if normalizer_type == "open_asr_leaderboard":
            normalizer_display = "EnglishNorm" if lang_code == "en" else "MultilingualNorm"
        else:
            normalizer_display = "Legacy (no normalization)"

        # Unit type for display
        unit_type = "chars" if metric_type == 'cer' else "words"

        # Calculate token counts using tokenizer
        def count_tokens(text):
            """Count actual tokens in text using the tokenizer."""
            if not text or text.strip() == "":
                return 0
            # Remove audio locator tags before tokenizing
            text_without_audio = text.replace(self.audio_locator_tag, "").strip()
            if not text_without_audio:
                return 0
            tokens = self.tokenizer.text_to_ids(text_without_audio)
            return len(tokens) if tokens else 0

        # Count units in raw text (words or characters)
        if metric_type == 'cer':
            ref_raw_count = len(reference)
            pred_raw_count = len(prediction)
        else:
            ref_raw_count = len(reference.split())
            pred_raw_count = len(prediction.split())

        # Count units in normalized text
        total_units = wer_stats.get('total_units', ref_raw_count)
        if metric_type == 'cer':
            pred_norm_count = len(prediction_normalized)
        else:
            pred_norm_count = len(prediction_normalized.split())

        # Estimate audio embedding frames
        audio_frames = 0
        audio_duration = None
        if audio_len_samples is not None and audio_len_samples > 0:
            audio_duration = audio_len_samples / self.sampling_rate
            audio_frames = int(audio_duration / self.token_equivalent_duration)

        # Clean all text for single-line display (removes newlines, control chars, truncates)
        ref_raw_display = clean_text_for_display(reference, max_length=120)
        pred_raw_display = clean_text_for_display(prediction, max_length=120)
        ref_norm_display = clean_text_for_display(reference_normalized, max_length=120)
        pred_norm_display = clean_text_for_display(prediction_normalized, max_length=120)

        # Clean prompt for display (remove audio tag for cleaner output)
        if prompt:
            prompt_clean = prompt.replace(self.audio_locator_tag, "")
            prompt_display = clean_text_for_display(prompt_clean, max_length=150)
            prompt_tokens = count_tokens(prompt)
        else:
            prompt_display = "(no prompt)"
            prompt_tokens = 0

        # Create full LLM input (prompt + audio + reference) - cleaned for display
        if prompt:
            full_input_parts = [clean_text_for_display(prompt, max_length=300)]
            if self.audio_locator_tag not in prompt:
                full_input_parts.append(self.audio_locator_tag)
            full_input_parts.append(clean_text_for_display(reference, max_length=200))
            full_llm_input = " ".join(full_input_parts)
            # Truncate combined result if still too long
            if len(full_llm_input) > 200:
                full_llm_input = full_llm_input[:200] + "..."
        else:
            full_llm_input = clean_text_for_display(reference, max_length=200)

        full_tokens = prompt_tokens + audio_frames + count_tokens(reference)

        # Extract error breakdown from stats
        total_errors = wer_stats.get('total_errors', 0)
        subs = wer_stats.get('substitutions', 0)
        dels = wer_stats.get('deletions', 0)
        ins = wer_stats.get('insertions', 0)

        # Format dataset info
        dataset_info = f" [{dataset_name}]" if dataset_name else ""

        # Build aligned log with separator
        separator = "=" * 100
        metric_display = metric_type.upper()

        # Start with blank line and separator
        lines.append("")
        lines.append(separator)
        lines.append(f"{step_type} Sample {idx + 1}{dataset_info}")

        # Audio duration if available
        if audio_duration is not None:
            lines.append(f"Audio Duration          : {audio_duration:.2f} sec")

        # Language and metric info
        lines.append(f"Language                : {lang_name} ({language})")
        lines.append(f"Metric Type             : {metric_display}")
        lines.append(f"Normalizer              : {normalizer_display}")

        # Separator for better visual grouping
        lines.append("-" * 100)

        # Calculate reference tokens for comprehensive token breakdown
        ref_tokens = count_tokens(reference)

        # LLM Input Composition with comprehensive token information
        lines.append(f"Prompt                  : {prompt_display} ({prompt_tokens} tokens)")

        # Enhanced audio embeddings info with frame-to-token mapping
        if audio_duration is not None and audio_frames > 0:
            # Calculate detailed metrics
            frame_rate = audio_frames / audio_duration  # frames per second
            ms_per_frame = (audio_duration / audio_frames) * 1000  # milliseconds per frame

            # Get embedding dimension from perception module if available
            try:
                embed_dim = self.perception.modality_adapter.d_model  # Conformer adapter dimension
            except AttributeError:
                embed_dim = None

            # Calculate compression ratio if audio samples available
            compression_ratio = None
            if audio_len_samples is not None and audio_len_samples > 0 and audio_frames > 0:
                compression_ratio = audio_len_samples / audio_frames

            # Main audio embeddings line with frame=token mapping and dimension
            # In SALM, each audio frame becomes one token in LLM input
            if embed_dim is not None:
                lines.append(
                    f"Audio Embeddings        : {self.audio_locator_tag} → [audio_embeddings] "
                    f"({audio_frames} frames ≡ {audio_frames} tokens, {embed_dim}d)"
                )
            else:
                lines.append(
                    f"Audio Embeddings        : {self.audio_locator_tag} → [audio_embeddings] "
                    f"({audio_frames} frames ≡ {audio_frames} tokens)"
                )

            # Detailed temporal and compression information
            temporal_info = f"{audio_duration:.2f}s @ {ms_per_frame:.0f}ms/frame ({frame_rate:.1f}fps)"
            if compression_ratio is not None:
                lines.append(
                    f"  └─ Temporal           : {temporal_info} | "
                    f"Compression: {audio_len_samples:,} samples → {audio_frames} frames (↓{compression_ratio:.0f}x)"
                )
            else:
                lines.append(f"  └─ Temporal           : {temporal_info}")

            # Additional embedding size information
            if embed_dim is not None:
                total_values = audio_frames * embed_dim
                lines.append(f"  └─ Embedding Size     : {audio_frames} × {embed_dim}d = {total_values:,} values")
        else:
            # Fallback when audio duration not available
            try:
                embed_dim = self.perception.modality_adapter.d_model
                lines.append(
                    f"Audio Embeddings        : {self.audio_locator_tag} → [audio_embeddings] "
                    f"({audio_frames} frames ≡ {audio_frames} tokens, {embed_dim}d)"
                )
                # Embedding size even without duration
                total_values = audio_frames * embed_dim
                lines.append(f"  └─ Embedding Size     : {audio_frames} × {embed_dim}d = {total_values:,} values")
            except AttributeError:
                lines.append(
                    f"Audio Embeddings        : {self.audio_locator_tag} → [audio_embeddings] "
                    f"({audio_frames} frames ≡ {audio_frames} tokens)"
                )

        lines.append(f"Reference Text          : {clean_text_for_display(reference, max_length=80)} ({ref_tokens} tokens)")

        # Token Distribution Summary - comprehensive breakdown
        lines.append("-" * 100)
        lines.append(
            f"LLM Input Token Summary : {full_tokens} tokens total "
            f"(Prompt: {prompt_tokens} + Audio: {audio_frames} + Reference: {ref_tokens})"
        )

        # Calculate token usage percentage if model has max context length
        try:
            max_context = getattr(self.llm, 'max_position_embeddings', None) or \
                         getattr(self.llm.config, 'max_position_embeddings', None)
            if max_context:
                usage_pct = (full_tokens / max_context) * 100
                lines.append(f"Token Usage             : {usage_pct:.1f}% of max context length ({max_context:,} tokens)")
        except (AttributeError, TypeError):
            # Skip if max context not available
            pass

        # Separator for better visual grouping
        lines.append("-" * 100)

        # Raw text (before normalization, includes chat template markers)
        lines.append(f"Reference (raw)         : {ref_raw_display} ({ref_raw_count} {unit_type})")
        lines.append(f"Prediction (raw)        : {pred_raw_display} ({pred_raw_count} {unit_type})")

        # Normalized text (after chat template removal + text normalization for WER/CER)
        # Chat role markers like "assistant", "user", "system" are removed for accurate ASR metrics
        lines.append(f"Reference (normalized)  : {ref_norm_display} ({total_units} {unit_type})")
        lines.append(f"Prediction (normalized) : {pred_norm_display} ({pred_norm_count} {unit_type})")

        # Separator for better visual grouping
        lines.append("-" * 100)

        # WER/CER with detailed error breakdown
        lines.append(
            f"{metric_display}                     : {wer*100:.2f}% | "
            f"Errors: {total_errors} (S: {subs}, D: {dels}, I: {ins}) / {total_units} {unit_type}"
        )
        lines.append(separator)

        return lines

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token at the output of ``self.perception``.
        """
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        audio_embs, audio_emb_lens = self.perception(
            input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
        )
        audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]

        # Debug: Check if audio placeholders are present in input_ids
        num_audio_placeholders = (batch["input_ids"] == self.audio_locator_tag_id).sum().item()
        if num_audio_placeholders == 0:
            import logging
            logging.warning(f"WARNING: No audio placeholders found in input_ids!")
            logging.warning(f"Audio locator tag ID: {self.audio_locator_tag_id}")
            logging.warning(f"Batch input_ids shape: {batch['input_ids'].shape}")
            logging.warning(f"Number of audio embeddings: {len(audio_embs)}")
            # Log first 50 tokens to see what's there
            if batch["input_ids"].numel() > 0:
                sample = batch["input_ids"][0][:50] if batch["input_ids"][0].numel() > 50 else batch["input_ids"][0]
                logging.warning(f"First 50 tokens of first sample: {sample.tolist()}")
                try:
                    decoded = self.tokenizer.ids_to_text(sample.tolist())
                    logging.warning(f"Decoded text: {decoded}")
                except:
                    pass

        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self.embed_tokens(input_ids_to_embed)
        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=batch["input_ids"],
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=batch["input_ids"].where(batch["loss_mask"], -100),  # CrossEntropyLoss().ignore_index
        )
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )

        B, T = inputs["input_embeds"].shape[:2]
        learning_rate = torch.as_tensor(
            self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0
        )

        ans = {
            "loss": loss,
            "learning_rate": learning_rate,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }

        # Log all metrics to TensorBoard
        self.log_dict(ans, on_step=True, prog_bar=False)

        # Log key metrics to progress bar (logger=False to avoid duplicate logging to TensorBoard)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=False)
        self.log("lr", learning_rate, on_step=True, prog_bar=True, logger=False)
        self.log("bs", B, on_step=True, prog_bar=True, logger=False)

        # Training prediction logging (separate from validation)
        if self.log_prediction_train and (self.trainer.global_step % self.log_prediction_train_interval) == 0:
            # Generate predictions from logits
            pred_ids_2d = forward_outputs["logits"].argmax(dim=-1)  # (B, T)
            target_ids_2d = inputs["target_ids"]  # (B, T)

            # Decode predictions and references
            num_samples = min(self.log_prediction_train_samples, pred_ids_2d.shape[0])
            for i in range(num_samples):
                # Get valid positions (not padding/ignore_index)
                valid_mask = target_ids_2d[i] != -100
                if valid_mask.sum() == 0:
                    continue

                # Extract valid token IDs
                pred_tokens = pred_ids_2d[i][valid_mask].cpu().tolist()
                ref_tokens = target_ids_2d[i][valid_mask].cpu().tolist()

                # Decode to text
                prediction = self.tokenizer.ids_to_text(pred_tokens)
                reference = self.tokenizer.ids_to_text(ref_tokens)

                # Extract input prompt
                full_input_ids = batch["input_ids"][i]
                non_pad_mask = full_input_ids != self.text_pad_id
                input_tokens = full_input_ids[non_pad_mask].cpu().tolist()
                full_text = self.tokenizer.ids_to_text(input_tokens)

                # Extract prompt (everything before reference)
                if reference.strip() in full_text:
                    prompt = full_text.split(reference.strip())[0].strip()
                else:
                    prompt = full_text

                # Extract language/metric metadata from batch (defaults: en, wer)
                language = batch.get('language', ['en'] * len(batch.get('audios', [])))
                metric_type = batch.get('metric', ['wer'] * len(batch.get('audios', [])))

                # Get language/metric for this sample (fallback to 'en'/'wer' if not available)
                sample_language = language[i] if isinstance(language, list) and i < len(language) else (language if isinstance(language, str) else 'en')

                # Extract metric_type with robust type checking and conversion
                if isinstance(metric_type, list) and i < len(metric_type):
                    metric_value = metric_type[i]
                else:
                    metric_value = metric_type

                # Ensure metric_value is a string (handle int, float, or other types)
                if isinstance(metric_value, str):
                    sample_metric = metric_value
                else:
                    # Convert non-string types to string or use default
                    sample_metric = str(metric_value) if metric_value is not None else 'wer'
                    # If converted value is not a valid metric name, default to 'wer'
                    if sample_metric not in ['wer', 'cer']:
                        sample_metric = 'wer'

                # Compute sample WER using configured calculator
                # IMPORTANT: WER calculator automatically removes chat template role markers
                # (e.g., "assistant", "user", "system") to ensure accurate ASR-only metrics.
                # Both predictions and references are cleaned before WER calculation.
                # This happens in:
                #   - EnglishTextNormalizer/BasicMultilingualTextNormalizer (open_asr_leaderboard mode)
                #   - LegacyWERCalculator.calculate() (legacy mode)
                sample_wer, wer_stats, norm_preds, norm_refs = self.wer_calculator.calculate(
                    predictions=[prediction],
                    references=[reference],
                    language=sample_language,
                    metric_type=sample_metric
                )

                # Get audio length for this sample (in samples)
                audio_len = batch["audio_lens"][i].item() if i < len(batch["audio_lens"]) else None

                # Log using enhanced format with raw and normalized text
                log_lines = self._format_prediction_log(
                    idx=i,
                    prompt=prompt,
                    reference=reference,
                    prediction=prediction,
                    reference_normalized=norm_refs[0] if norm_refs else reference,
                    prediction_normalized=norm_preds[0] if norm_preds else prediction,
                    wer=sample_wer,
                    wer_stats=wer_stats,
                    language=sample_language,
                    step_type="Training",
                    audio_len_samples=audio_len,
                    metric_type=sample_metric
                )
                for line in log_lines:
                    logging.info(line)

        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)
        self._partial_wer_nums = defaultdict(list)  # Error numerator (edit distance)
        self._partial_wer_denoms = defaultdict(list)  # Error denominator (word/char count)
        self._dataset_metrics = {}  # Track metric type (wer/cer) for each dataset

    def on_validation_epoch_end(self) -> None:
        # Aggregate val_loss
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        # Aggregate val_acc
        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        # Aggregate error metrics (WER/CER) with proper distributed training support
        # Follow the pattern from transformer_bpe_models.py (Lines 529-541)
        error_metrics = []
        for name in self._partial_wer_nums.keys():
            # Sum across all validation steps in this GPU
            error_num = torch.stack(self._partial_wer_nums[name]).sum()
            error_denom = torch.stack(self._partial_wer_denoms[name]).sum()

            # All-reduce across all GPUs to get global sum
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(error_num, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(error_denom, op=torch.distributed.ReduceOp.SUM)

            # Compute error rate = total_edit_distance / total_units
            if error_denom > 0:
                error_rate = error_num / error_denom
            else:
                error_rate = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            # Get metric type for this dataset (default to 'wer' if not tracked)
            metric_type = self._dataset_metrics.get(name, 'wer')

            # Log with dynamic metric name (val_wer_{name} or val_cer_{name})
            self.log(f"val_{metric_type}_{name}", error_rate, on_epoch=True, sync_dist=False)
            error_metrics.append(error_rate)

        # Log overall error rate (average across datasets, keep as val_wer for backward compatibility)
        if len(error_metrics) > 0:
            self.log("val_wer", torch.stack(error_metrics).mean(), on_epoch=True, sync_dist=False)

        # Clear tracking dictionaries
        self._partial_val_losses.clear()
        self._partial_accuracies.clear()
        self._partial_wer_nums.clear()
        self._partial_wer_denoms.clear()
        self._dataset_metrics.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            # Token-level accuracy
            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            # WER calculation
            pred_ids_2d = forward_outputs["logits"].argmax(dim=-1)  # (B, T)
            target_ids_2d = inputs["target_ids"]  # (B, T)

            hypotheses = []
            references = []
            input_contexts = [] if self.log_prediction_valid else None

            for i in range(pred_ids_2d.shape[0]):
                # Get valid positions (not padding/ignore_index)
                valid_mask = target_ids_2d[i] != -100
                if valid_mask.sum() == 0:
                    continue

                # Extract valid token IDs
                pred_tokens = pred_ids_2d[i][valid_mask].cpu().tolist()
                ref_tokens = target_ids_2d[i][valid_mask].cpu().tolist()

                # Decode to text
                hypothesis = self.tokenizer.ids_to_text(pred_tokens)
                reference = self.tokenizer.ids_to_text(ref_tokens)

                hypotheses.append(hypothesis)
                references.append(reference)

                # Extract input context for logging (prompt before response)
                if self.log_prediction_valid:
                    # Get the full input_ids for this sample
                    full_input_ids = dataset_batch["input_ids"][i]
                    # Find non-padding tokens
                    non_pad_mask = full_input_ids != self.text_pad_id
                    input_tokens = full_input_ids[non_pad_mask].cpu().tolist()

                    # Decode the full input (includes prompt + response)
                    full_text = self.tokenizer.ids_to_text(input_tokens)

                    # Extract just the prompt part (everything before the reference/target)
                    # The target tokens are at the end, so we can find where they start
                    if reference.strip() in full_text:
                        # Split at the reference to get the prompt
                        prompt_text = full_text.split(reference.strip())[0].strip()
                    else:
                        # Fallback: just show what we have
                        prompt_text = full_text

                    input_contexts.append(prompt_text)

            # Log prediction samples (number of samples controlled by log_prediction_valid_samples)
            if self.log_prediction_valid and len(hypotheses) > 0:
                num_samples = min(self.log_prediction_valid_samples, len(hypotheses))
                for idx in range(num_samples):
                    # Get prompt for this sample
                    prompt = input_contexts[idx] if input_contexts and idx < len(input_contexts) else ""

                    # Extract language/metric metadata from batch (defaults: en, wer)
                    language = dataset_batch.get('language', ['en'] * len(dataset_batch.get('audios', [])))
                    metric_type = dataset_batch.get('metric', ['wer'] * len(dataset_batch.get('audios', [])))

                    # Get language/metric for this sample
                    sample_language = language[idx] if isinstance(language, list) and idx < len(language) else (language if isinstance(language, str) else 'en')

                    # Extract metric_type with robust type checking and conversion
                    if isinstance(metric_type, list) and idx < len(metric_type):
                        metric_value = metric_type[idx]
                    else:
                        metric_value = metric_type

                    # Ensure metric_value is a string (handle int, float, or other types)
                    if isinstance(metric_value, str):
                        sample_metric = metric_value
                    else:
                        # Convert non-string types to string or use default
                        sample_metric = str(metric_value) if metric_value is not None else 'wer'
                        # If converted value is not a valid metric name, default to 'wer'
                        if sample_metric not in ['wer', 'cer']:
                            sample_metric = 'wer'

                    # Compute WER for this sample using configured calculator
                    # IMPORTANT: WER calculator automatically removes chat template role markers
                    # (e.g., "assistant", "user", "system") to ensure accurate ASR-only metrics.
                    # Both predictions and references are cleaned before WER calculation.
                    # This happens in:
                    #   - EnglishTextNormalizer/BasicMultilingualTextNormalizer (open_asr_leaderboard mode)
                    #   - LegacyWERCalculator.calculate() (legacy mode)
                    sample_wer, wer_stats, norm_preds, norm_refs = self.wer_calculator.calculate(
                        predictions=[hypotheses[idx]],
                        references=[references[idx]],
                        language=sample_language,
                        metric_type=sample_metric
                    )

                    # Get audio length for this sample (in samples)
                    audio_len = dataset_batch["audio_lens"][idx].item() if idx < len(dataset_batch["audio_lens"]) else None

                    # Log using enhanced format with raw and normalized text
                    log_lines = self._format_prediction_log(
                        idx=idx,
                        prompt=prompt,
                        reference=references[idx],
                        prediction=hypotheses[idx],
                        reference_normalized=norm_refs[0] if norm_refs else references[idx],
                        prediction_normalized=norm_preds[0] if norm_preds else hypotheses[idx],
                        wer=sample_wer,
                        wer_stats=wer_stats,
                        language=sample_language,
                        step_type="Validation",
                        dataset_name=name,
                        audio_len_samples=audio_len,
                        metric_type=sample_metric
                    )
                    for line in log_lines:
                        logging.info(line)

            # Compute WER if we have valid samples
            if len(hypotheses) > 0:
                # Extract language/metric metadata from batch (defaults: en, wer)
                language = dataset_batch.get('language', ['en'] * len(hypotheses))
                metric_type = dataset_batch.get('metric', ['wer'] * len(hypotheses))

                # Ensure we have lists
                if not isinstance(language, list):
                    language = [language] * len(hypotheses)
                if not isinstance(metric_type, list):
                    metric_type = [metric_type] * len(hypotheses)

                # Calculate WER using configured calculator for the entire batch
                # We need numerator and denominator for proper distributed aggregation
                # IMPORTANT: WER calculator automatically removes chat template role markers
                # from both hypotheses and references before calculation (same as training_step)
                _, batch_stats, _, _ = self.wer_calculator.calculate(
                    predictions=hypotheses,
                    references=references,
                    language=language[0] if language else 'en',  # Use first sample's language
                    metric_type=metric_type[0] if metric_type else 'wer'  # Use first sample's metric
                )

                wer_num = torch.tensor(batch_stats['total_errors'], device=self.device, dtype=torch.float32)
                wer_denom = torch.tensor(batch_stats['total_units'], device=self.device, dtype=torch.float32)
            else:
                wer_num = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                wer_denom = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)
            self._partial_wer_nums[name].append(wer_num)
            self._partial_wer_denoms[name].append(wer_denom)

            # Store metric type for this dataset (use first sample's metric)
            if name not in self._dataset_metrics and len(hypotheses) > 0:
                self._dataset_metrics[name] = metric_type[0] if metric_type else 'wer'

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate LLM answers given text or mixed text+audio prompts.

        Example 1. High-level API using ``prompts`` to provide both text and audio::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [
            ...             {
            ...                 "role": "user",
            ...                 "content": f"Transcribe the following: {model.audio_locator_tag}",
            ...                 "audio": ["path/to/audio.wav"],
            ...             }
            ...         ]
            ...    ],
            ...    max_new_tokens=128,
            ... )

        You may also include a ``transformers.GenerationConfig`` object to customize decoding strategy::

            >>> answer_ids = model.generate(..., generation_config=GenerationConfig(do_sample=True, num_beams=5))

        Example 2. Lower-level API, using ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}],
            ...        [{"role": "user", "content": f"Transcribe the following in Polish: {model.audio_locator_tag}"}],
            ...    ],
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Example 3. Lower-level API, using pre-tokenized and pre-formatted ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=prompts,  # torch.Tensor, int64, of shape (batch, num_tokens)
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Inputs:
            prompts: batch of prompts Tensor or as list[dict] each in the following format
                [
                  # batch example id 0
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}]
                  # batch example id 1
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following in Polish: {model.audio_locator_tag}"}]
                ]
                "role" is LLM-specific, you can pass multiple turns as well.
                If ``prompts`` is a Tensor, we assume it was already formatted in the relevant chat template
                and tokenized with the model's tokenizer.
            audios: Optional. Time-domain audio signal zero-padded batch of shape (B, T).
                The number of audios must correspond to the number of occurrences of <audio_locator_tag> in prompts.
                Each prompt can have multiple audios.
            audio_lens: Optional. Length of each audio example.
            generation_config: Optional HuggingFace GenerationConfig object.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        if audios is not None:
            # Audio + text input for generation.
            # Prepare token embeddings and audio embeddings.
            tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
            token_embeds = self.embed_tokens(tokens_to_embed)
            # TODO: temporary workaround to perform batch_size=1 inference for audio encoder
            #   due to accuracy issues at bs>1
            audio_embeds, audio_embed_lens = self.perception(audios, audio_lens)
            audio_embeds = [audio_embeds[i, :elen] for i, elen in enumerate(audio_embed_lens)]
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
            generation_inputs = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}
        else:
            # Text-only generation.
            attention_mask = tokens != self.text_pad_id
            generation_inputs = {"input_ids": tokens, "attention_mask": attention_mask}
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )
        # Generate the answers using HF Generate API.
        # Note: we need to put the text embedding layer back to the LLM for processing.
        with move_embedding(self):
            answer_tokens = self.llm.generate(
                **generation_inputs,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens

    def configure_optimizers(self):
        return configure_optimizers(self)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        PyTorch Lightning hook called before saving checkpoint.

        Ensures complete perception config is saved in checkpoint for offline inference.
        Perception configs (preprocessor/encoder) are populated during training and
        must be explicitly saved to make checkpoints self-contained.
        """
        from omegaconf import OmegaConf

        # Verify perception module is initialized
        if not hasattr(self, 'perception') or self.perception is None:
            logging.warning("Perception module not initialized - checkpoint will have incomplete config.")
            return

        if not hasattr(self, 'cfg') or 'perception' not in self.cfg:
            logging.warning("Model config missing perception section - checkpoint will have incomplete config.")
            return

        perception_cfg = self.cfg.perception

        # Verify config completeness
        has_preprocessor = 'preprocessor' in perception_cfg and perception_cfg.preprocessor is not None
        has_encoder = 'encoder' in perception_cfg and perception_cfg.encoder is not None
        has_output_dim = 'output_dim' in perception_cfg and perception_cfg.output_dim is not None

        if not (has_preprocessor and has_encoder and has_output_dim):
            logging.warning(
                f"Incomplete perception config detected: "
                f"preprocessor={has_preprocessor}, encoder={has_encoder}, output_dim={has_output_dim}"
            )
            return

        # Update checkpoint with complete perception config
        if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
            complete_perception_cfg = OmegaConf.to_container(perception_cfg, resolve=True)
            checkpoint['hyper_parameters']['cfg']['perception'] = complete_perception_cfg

            logging.debug(
                f"Saved perception config to checkpoint: "
                f"preprocessor={complete_perception_cfg.get('preprocessor', {}).get('_target_', 'unknown')}, "
                f"encoder={complete_perception_cfg.get('encoder', {}).get('_target_', 'unknown')}, "
                f"output_dim={complete_perception_cfg.get('output_dim', 'unknown')}"
            )
        else:
            logging.warning("Unexpected checkpoint structure - cannot update perception config.")

    def to_config_file(self, path2yaml_file: str):
        """
        Saves current model configuration to YAML config file.

        Extracts complete model configuration including perception config
        and saves it to a YAML file suitable for .nemo format.

        Args:
            path2yaml_file: Path to yaml file where model configuration will be saved
        """
        from omegaconf import OmegaConf

        config_to_save = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))

        # Verify perception configuration if available
        if hasattr(self, 'perception') and self.perception is not None and 'perception' in config_to_save:
            perception_cfg = config_to_save.perception
            has_preprocessor = 'preprocessor' in perception_cfg and perception_cfg.preprocessor is not None
            has_encoder = 'encoder' in perception_cfg and perception_cfg.encoder is not None
            has_output_dim = 'output_dim' in perception_cfg and perception_cfg.output_dim is not None

            if not (has_preprocessor and has_encoder and has_output_dim):
                logging.warning(
                    f"Incomplete perception config when saving .nemo: "
                    f"preprocessor={has_preprocessor}, encoder={has_encoder}, output_dim={has_output_dim}"
                )

        # Save config to YAML file
        with open(path2yaml_file, 'w', encoding='utf-8') as fout:
            OmegaConf.save(config=config_to_save, f=fout, resolve=True)

        logging.debug(f"Saved model config to: {path2yaml_file}")

    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file.

        .nemo file is a tar archive (uncompressed) containing:
            - model_config.yaml: Complete model configuration
            - model_weights.ckpt: Model state_dict

        Note: Follows NeMo's standard approach (SaveRestoreConnector) of using
        uncompressed tar for fast saving. This prioritizes speed over file size.

        Args:
            save_path: Path to .nemo file where model instance should be saved

        Example:
            >>> model.save_to("salm_model.nemo")
        """
        import os
        import tarfile
        import tempfile
        import time
        from pathlib import Path
        from nemo.utils.app_state import AppState
        from nemo.utils.get_rank import is_global_rank_zero

        if not isinstance(save_path, Path):
            save_path = Path(save_path).expanduser().resolve()

        app_state = AppState()

        # Check for model parallel mode
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            logging.warning(
                "Model parallel size > 1 detected. .nemo saving may not work correctly. "
                "Consider using checkpoint format (.ckpt) for model-parallel models."
            )

        if is_global_rank_zero():
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Saving model to .nemo format: {save_path}")
            start_time = time.time()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save config
                config_path = os.path.join(tmpdir, "model_config.yaml")
                self.to_config_file(config_path)

                # Save state_dict
                weights_path = os.path.join(tmpdir, "model_weights.ckpt")
                torch.save(self.state_dict(), weights_path)

                # Create tar archive (uncompressed for speed - same as ASR models)
                with tarfile.open(save_path, "w:") as tar:
                    tar.add(config_path, arcname="model_config.yaml")
                    tar.add(weights_path, arcname="model_weights.ckpt")

            elapsed = time.time() - start_time
            logging.info(f"Successfully saved model to {save_path} (took {elapsed:.1f}s)")
        else:
            logging.debug(f"Rank {app_state.global_rank} skipping save (not rank 0)")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ):
        """
        Restores model instance (weights and configuration) from .nemo file.

        Loads a SALM model from a .nemo archive file for offline inference.

        Args:
            restore_path: Path to .nemo file from which model should be instantiated
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True.

        Returns:
            An instance of SALM model with weights and config restored from .nemo file

        Example:
            >>> model = SALM.restore_from("salm_model.nemo")
        """
        import os
        import tarfile
        import tempfile
        from omegaconf import OmegaConf

        restore_path = os.path.abspath(os.path.expanduser(restore_path))

        if not os.path.exists(restore_path):
            raise FileNotFoundError(f"Cannot find .nemo file: {restore_path}")

        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"Restoring SALM model from {restore_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract archive (auto-detect compression)
            # Try uncompressed first (NeMo 1.7.0+), fallback to gzip if needed
            tar_mode = "r:"
            try:
                with tarfile.open(restore_path, tar_mode) as tar:
                    tar.extractall(tmpdir)
            except tarfile.ReadError:
                # Older checkpoint with gzip compression
                tar_mode = "r:gz"
                with tarfile.open(restore_path, tar_mode) as tar:
                    tar.extractall(tmpdir)

            # Load config
            config_path = os.path.join(tmpdir, "model_config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"model_config.yaml not found in .nemo archive: {restore_path}")

            conf = OmegaConf.load(config_path)

            # Verify perception config
            if 'perception' in conf:
                perception_cfg = conf.perception
                has_preprocessor = 'preprocessor' in perception_cfg and perception_cfg.preprocessor is not None
                has_encoder = 'encoder' in perception_cfg and perception_cfg.encoder is not None
                has_output_dim = 'output_dim' in perception_cfg and perception_cfg.output_dim is not None

                if not (has_preprocessor and has_encoder and has_output_dim):
                    raise RuntimeError(
                        f"Incomplete perception config in .nemo file:\n"
                        f"  - preprocessor: {'PRESENT' if has_preprocessor else 'MISSING'}\n"
                        f"  - encoder: {'PRESENT' if has_encoder else 'MISSING'}\n"
                        f"  - output_dim: {'PRESENT' if has_output_dim else 'MISSING'}\n"
                        f"\n"
                        f"This .nemo file was likely created before the checkpoint save enhancement.\n"
                        f"Please re-train and save the model with the updated code to create a complete .nemo file."
                    )

            # Initialize model
            cfg_dict = OmegaConf.to_container(conf, resolve=True)
            cfg_dict['pretrained_weights'] = False  # Use weights from .nemo file

            logging.info("Initializing model...")
            # Skip perception setup - will be initialized from state_dict instead
            instance = cls(cfg=cfg_dict, skip_perception_setup=True)

            # Initialize perception module from config (structure only, weights loaded below)
            from nemo.collections.speechlm2.modules import AudioPerceptionModule

            if 'perception' in instance.cfg:
                logging.info("Initializing perception module from config...")
                instance.perception = AudioPerceptionModule(instance.cfg.perception).train()
            else:
                raise RuntimeError(
                    "Perception config not found in .nemo file. "
                    "The .nemo file may be corrupted or from an incompatible version."
                )

            # Load weights
            weights_path = os.path.join(tmpdir, "model_weights.ckpt")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"model_weights.ckpt not found in .nemo archive: {restore_path}")

            logging.info("Loading weights...")
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            instance.load_state_dict(state_dict, strict=strict)

            instance = instance.to(map_location)
            logging.info(f"Successfully restored model from {restore_path}")

            return instance

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """
        Load a SALM model from various sources.

        This method is automatically used by evaluation scripts (salm_eval.py, salm_generate.py)
        and supports loading from:
        1. Local .nemo files (offline inference ready with Solution 1)
        2. Local .ckpt files (PyTorch Lightning checkpoints with Solution 1)
        3. HuggingFace Hub model IDs or local directories (original behavior)

        Args:
            model_id: Can be:
                - Path to .nemo file (e.g., "model.nemo" or "/path/to/model.nemo")
                - Path to .ckpt file (e.g., "checkpoint.ckpt" or "/path/to/checkpoint.ckpt")
                - HuggingFace model ID (e.g., "nvidia/canary-qwen-2.5b")
                - Path to directory with config.yaml and pytorch_model.bin

        Returns:
            SALM model instance ready for inference

        Examples:
            >>> # Load from .nemo file (offline inference)
            >>> model = SALM.from_pretrained("salm_model.nemo")

            >>> # Load from .ckpt file (offline inference)
            >>> model = SALM.from_pretrained("checkpoint.ckpt")

            >>> # Load from HuggingFace Hub
            >>> model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
        """
        import os
        from pathlib import Path

        # Expand user path and resolve
        model_path = Path(model_id).expanduser()

        # Check if it's a local file
        if model_path.exists() and model_path.is_file():
            suffix = model_path.suffix.lower()

            if suffix == '.nemo':
                # Load from .nemo file
                logging.info(f"Loading SALM model from .nemo file: {model_path}")
                return cls.restore_from(
                    restore_path=str(model_path),
                    map_location=torch.device(map_location),
                    strict=strict,
                )

            elif suffix == '.ckpt':
                # Load from PyTorch Lightning checkpoint
                logging.info(f"Loading SALM model from .ckpt file: {model_path}")

                # Use PyTorch Lightning's load_from_checkpoint
                model = cls.load_from_checkpoint(
                    str(model_path),
                    map_location=map_location,
                    strict=strict,
                )

                logging.info(f"Successfully loaded model from checkpoint")
                return model

        # Not a .nemo or .ckpt file, use original HFHubMixin behavior
        logging.info(f"Loading SALM model from HuggingFace Hub or directory: {model_id}")
        return super(SALM, cls)._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            **model_kwargs,
        )

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            # TODO: Distributing embeddings with TP in this setup is tricky
            #       because we're adding with the output of a non-parallelized
            #       speech encoder.
            # for m in (self.embed_tokens, self.embed_audio_tokens):
            #     parallelize_module(
            #         m,
            #         tp_mesh,
            #         ColwiseParallel(
            #             # input_layouts=Shard(1),
            #             # # Optional: Shard the output along the class dimension to compute the loss in parallel.
            #             # # See `loss_parallel` in `train.py`
            #             # output_layouts=Shard(1),
            #             # use_local_output=False,
            #         ),
            #     )

            # # Parallelize the first embedding and the last linear out projection
            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            # Parallelize each transformer block
            for transformer_block in llm.model.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                # Apply the plan for the current transformer block
                parallelize_module(transformer_block, tp_mesh, plan)

            parallelize_module(
                llm.lm_head,
                tp_mesh,
                ColwiseParallel(
                    input_layouts=Shard(1),
                    # Optional: Shard the output along the class dimension to compute the loss in parallel.
                    # See `loss_parallel` in `train.py`
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
            )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported
            self._use_fsdp = True
            fsdp_config = {"mesh": dp_mesh}
            for idx, layer in enumerate(llm.model.layers):
                llm.model.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            llm.lm_head = fully_shard(llm.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }


def replace_placeholders_and_build_targets(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    padding_id: int,
    placeholder_id: int,
    replacements: list[torch.Tensor],
    target_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Replaces each occurrence of the placeholder_id in input_ids with the corresponding tensor
    from the replacements list in the embeds tensor, and creates corresponding adjusted target_ids.

    Note: when padding is necessary, we apply left-padding to the examples not to introduce
        anomalies at generation time.

    Args:
      input_ids (Tensor): shape (batch, sequence_length); input token ids.
      embeds (Tensor): shape (batch, sequence_length, hidden_dim); embeddings for each token.
      padding_id (int): these IDs will be marked as ignore_index in target_ids.
      placeholder_id (int): an id to be replaced.
      replacements (list of Tensor): each Tensor has shape (L_i, hidden_dim), with L_i arbitrary.
      target_ids (Tensor): shape (batch, sequence_length); target token ids.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - Tensor of shape (batch, max_new_sequence_length, hidden_dim) corresponding to
          ``embeds`` after replacements.
        - Tensor of shape (batch, max_new_sequence_length) with adjusted target IDs where:
          * Original target values are preserved where input was not a placeholder or padding
          * Positions that were placeholders, padding, or added by replacements are set to -100
          Will be None if target_ids input was None.
        - Tensor of shape (batch, max_new_sequence_length) with attention padding masks
          updated to account for shape changes due to replacements.
    """
    batch_size, seq_len = input_ids.size()
    if target_ids is not None:
        assert target_ids.size() == input_ids.size(), "target_ids must have the same shape as input_ids"

    hidden_dim = embeds.size(2)
    device, dtype = embeds.device, embeds.dtype
    ignore_index = -100  # Standard ignore_index value for CrossEntropyLoss

    # Un-pad the tensors because we'll need to re-apply new padding after replacements anyway.
    input_ids, embeds, target_ids = _unpad_inputs(input_ids, embeds, target_ids, padding_id)

    output_sequences = []
    output_target_ids = []
    output_att_masks = []
    replacement_idx = 0

    for i in range(batch_size):
        # Find all placeholder positions at once using tensor operations
        placeholder_positions = (input_ids[i] == placeholder_id).nonzero(as_tuple=True)[0]

        # Handle the case with no placeholders more efficiently
        if len(placeholder_positions) == 0:
            output_sequences.append(embeds[i])

            # Start with original target_ids and replace positions where input was padding
            if target_ids is not None:
                new_target_ids = target_ids[i].clone()
                new_target_ids[input_ids[i] == padding_id] = ignore_index
                output_target_ids.append(new_target_ids)
            output_att_masks.append(input_ids[i] != padding_id)
            continue

        # Build segments between placeholders
        segments = []  # For embeddings
        target_segments = []  # For target IDs
        att_masks = []
        prev_pos = 0

        for pos in placeholder_positions:
            # Add segment before placeholder (if any)
            if pos > prev_pos:
                segments.append(embeds[i][prev_pos:pos])

                # For target IDs: keep original targets but mark positions that were padding in input
                if target_ids is not None:
                    segment_target_ids = target_ids[i][prev_pos:pos].clone()
                    segment_target_ids[segment_target_ids == padding_id] = ignore_index
                    target_segments.append(segment_target_ids)
                att_masks.append(input_ids[i][prev_pos:pos] != padding_id)

            # Add replacement for embeddings
            rep = replacements[replacement_idx]
            segments.append(rep)

            # For target IDs: all replacement positions get ignore_index
            target_segments.append(torch.full((rep.size(0),), ignore_index, dtype=torch.long, device=device))
            att_masks.append(torch.ones((rep.size(0),), dtype=torch.bool, device=device))

            replacement_idx += 1
            prev_pos = pos + 1  # Skip placeholder

        # Add remaining segment after last placeholder (if any)
        if prev_pos < seq_len:
            segments.append(embeds[i][prev_pos:seq_len])

            # For target IDs: keep original targets but mark positions that were padding in input
            if target_ids is not None:
                segment_target_ids = target_ids[i][prev_pos:seq_len].clone()
                segment_target_ids[segment_target_ids == padding_id] = ignore_index
                target_segments.append(segment_target_ids)
            att_masks.append(input_ids[i][prev_pos:seq_len] != padding_id)

        # Concatenate all segments for this example
        output_sequences.append(torch.cat(segments, dim=0))
        output_att_masks.append(torch.cat(att_masks, dim=0))
        if target_ids is not None:
            output_target_ids.append(torch.cat(target_segments, dim=0))

    # Verify all replacements were used
    if replacement_idx != len(replacements):
        raise ValueError(f"Expected {len(replacements)} replacements but used {replacement_idx}")

    # Create padded output tensors
    max_seq_length = max(seq.size(0) for seq in output_sequences)
    output = torch.zeros(batch_size, max_seq_length, hidden_dim, device=device, dtype=dtype)
    if target_ids is not None:
        new_target_ids = torch.full((batch_size, max_seq_length), ignore_index, dtype=torch.long, device=device)
    else:
        new_target_ids = None
    attention_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.bool, device=device)

    if target_ids is None:
        output_target_ids = repeat(None)
    for i, (seq, tgt, att) in enumerate(zip(output_sequences, output_target_ids, output_att_masks)):
        seq_len = seq.size(0)
        output[i, -seq_len:] = seq
        if tgt is not None:
            new_target_ids[i, -seq_len:] = tgt
        attention_masks[i, -seq_len:] = att

    return output, new_target_ids, attention_masks


def _unpad_inputs(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    target_ids: Optional[torch.Tensor],
    padding_id: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    def first_index_not_value(tensor, value):
        mask = tensor != value
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() > 0:
            return indices[0].item()
        else:
            return -1

    input_ids_unpad, embeds_unpad = [], []
    target_ids_unpad = [] if target_ids is not None else None
    for i in range(input_ids.shape[0]):
        idx = first_index_not_value(input_ids[i], padding_id)
        input_ids_unpad.append(input_ids[i, idx:])
        embeds_unpad.append(embeds[i, idx:])
        if target_ids is not None:
            target_ids_unpad.append(target_ids[i, idx:])
    return input_ids_unpad, embeds_unpad, target_ids_unpad


def _resolve_audios_in_prompt(
    prompts: list[list[dict]], sampling_rate: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    from lhotse import Recording

    paths = []
    for conversation in prompts:
        for turn in conversation:
            if "audio" in turn:
                turn_audio = turn["audio"]
                if isinstance(turn_audio, (str, Path)):
                    turn_audio = [turn_audio]
                for p in turn_audio:
                    assert isinstance(p, (str, Path)), f"Invalid value under prompt key 'audio': {p}"
                    paths.append(p)
    if not paths:
        return None
    cuts = CutSet([Recording.from_file(p).to_cut() for p in paths])
    with torch.device("cpu"):  # workaround for a Lhotse issue when default device is CUDA during collation
        audio, audio_lens = cuts.resample(sampling_rate).load_audio(collate=True)
    return (
        torch.as_tensor(audio).to(device, non_blocking=True),
        torch.as_tensor(audio_lens).to(device, non_blocking=True),
    )
