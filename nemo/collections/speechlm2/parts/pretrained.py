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
from contextlib import contextmanager
from pathlib import Path
import logging

import torch
from omegaconf import open_dict
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from torch import nn

from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Enhanced with better error handling for incompatible checkpoints.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        try:
            # Try standard loading first
            return cls.restore_from(model_path_or_name)
        except (ModuleNotFoundError, TypeError) as e:
            # Suppress verbose error logging for expected failures
            logger.debug(f"Standard loading failed (expected): {e}")

            # Try with specific model classes for ASR
            if cls == ASRModel:
                # Try known ASR model classes
                from nemo.collections.asr.models import (
                    EncDecCTCModelBPE,
                    EncDecHybridRNNTCTCBPEModel,
                    EncDecRNNTBPEModel,
                )
                # Also try the custom hybrid model
                from nemo.collections.asr.models.hybrid_transformer_transcribe_ctc_bpe_models import (
                    HybridTransformerTranscribeCTCBPEModel,
                )

                for model_class in [
                    EncDecHybridRNNTCTCBPEModel,
                    HybridTransformerTranscribeCTCBPEModel,
                    EncDecCTCModelBPE,
                    EncDecRNNTBPEModel,
                ]:
                    try:
                        logger.debug(f"Trying to load with {model_class.__name__}...")
                        model = model_class.restore_from(model_path_or_name)
                        logger.info(f"✓ Successfully loaded ASR model with {model_class.__name__}")
                        return model
                    except Exception as sub_e:
                        # Suppress verbose error messages from failed attempts
                        logger.debug(f"Failed with {model_class.__name__}: {sub_e}")
                        continue

                # If all specific classes fail, raise the original error
                raise RuntimeError(
                    f"Could not load ASR model from {model_path_or_name}. "
                    f"The checkpoint may be incompatible with the current NeMo version. "
                    f"Consider using an official pretrained model like 'nvidia/canary-1b-v2' instead. "
                    f"Original error: {e}"
                )
            else:
                # For non-ASR models, raise the original error
                raise e
    else:
        return cls.from_pretrained(model_path_or_name)


def load_custom_asr_encoder_only(nemo_path: str):
    """
    Load only the encoder and preprocessor from a custom ASR .nemo checkpoint.
    This is useful when the full model class is not available but the encoder structure is compatible.
    """
    import tarfile
    import tempfile
    import torch
    from omegaconf import OmegaConf
    from pathlib import Path

    logger.debug(f"Extracting encoder from: {nemo_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the .nemo file (which is a tar archive)
        with tarfile.open(nemo_path, 'r') as tar:
            tar.extractall(tmpdir)

        # Load the config
        config_path = Path(tmpdir) / "model_config.yaml"
        if not config_path.exists():
            raise RuntimeError(f"Could not find model_config.yaml in {nemo_path}")

        config = OmegaConf.load(config_path)

        # Load the state dict
        weights_path = Path(tmpdir) / "model_weights.ckpt"
        if not weights_path.exists():
            raise RuntimeError(f"Could not find model_weights.ckpt in {nemo_path}")

        state_dict = torch.load(weights_path, map_location='cpu')

        # Create a simple namespace object to hold the config and state dict
        # This mimics the structure expected by setup_speech_encoder
        class CustomASRModel:
            def __init__(self, cfg, state_dict):
                self.cfg = cfg
                self._state_dict = state_dict

            def eval(self):
                return self

            def state_dict(self):
                return self._state_dict

        # Return the custom model object
        return CustomASRModel(config, state_dict)


def _load_qwen3_as_qwen2(model_path_or_name: str, dtype=torch.float32):
    """
    Load Qwen3 model using Qwen2 compatibility mode.

    Qwen3 and Qwen2 share nearly identical architectures, so we can load Qwen3 weights
    into a Qwen2 model structure when the transformers library doesn't support Qwen3.

    This creates a temporary modified config.json with model_type changed from 'qwen3' to 'qwen2',
    then loads the model with that modified config.

    Args:
        model_path_or_name: Path to Qwen3 model directory
        dtype: Model dtype for loading

    Returns:
        Loaded model with Qwen3 weights in Qwen2 architecture

    Raises:
        RuntimeError: If the model is not actually Qwen3 or if loading fails
    """
    import json
    import tempfile
    import shutil
    from pathlib import Path

    model_path = Path(model_path_or_name)

    # Verify this is a local path
    if not model_path.exists():
        raise RuntimeError(
            f"Qwen2 compatibility mode requires a local model directory, but path does not exist: {model_path}"
        )

    # Load original config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Verify this is actually Qwen3
    original_model_type = config.get('model_type')
    if original_model_type != 'qwen3':
        raise RuntimeError(
            f"Expected model_type='qwen3' but got '{original_model_type}'. "
            f"Qwen2 compatibility mode only works for Qwen3 models."
        )

    logger.info(f"  Model type: {original_model_type} → qwen2 (compatibility mode)")

    # Create temporary directory with modified config
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy all files to temp directory (preserving structure)
        logger.debug(f"  Copying model files to temporary directory...")
        for item in model_path.iterdir():
            if item.is_file():
                shutil.copy2(item, tmpdir_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, tmpdir_path / item.name)

        # Modify config.json: qwen3 → qwen2
        config['model_type'] = 'qwen2'
        modified_config_path = tmpdir_path / "config.json"
        with open(modified_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"  Loading with Qwen2 architecture (compatible with Qwen3)...")

        # Load model with modified config
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(tmpdir_path),
                dtype=dtype,
                local_files_only=True  # Don't try to download anything
            )
            logger.info("  ✓ Successfully loaded Qwen3 model using Qwen2 compatibility mode")
            return model

        except Exception as e:
            logger.error(f"  Failed to load model in Qwen2 compatibility mode: {e}")
            raise RuntimeError(
                f"Could not load Qwen3 model even with Qwen2 compatibility. "
                f"Please upgrade transformers: pip install --upgrade 'transformers>=4.37.0'"
            ) from e


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM with Qwen3 compatibility.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Enhanced with Qwen3 support:
    - Tries trust_remote_code first for newer architectures
    - Falls back to Qwen2 mapping for Qwen3 models (architecturally compatible)
    - Handles dtype parameter compatibility across transformers versions
    - Provides clear error messages for version issues
    """
    if pretrained_weights:
        try:
            # First attempt: trust_remote_code with dtype for custom/newer models
            return AutoModelForCausalLM.from_pretrained(
                model_path_or_name,
                dtype=dtype,
                trust_remote_code=True
            )
        except TypeError as type_err:
            # Handle dtype parameter not supported in older transformers versions
            if "dtype" in str(type_err) or "unexpected keyword argument" in str(type_err):
                logger.warning(
                    f"dtype parameter not supported in current transformers version.\n"
                    f"  Retrying without dtype parameter (will use model's default dtype)..."
                )
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        model_path_or_name,
                        trust_remote_code=True
                    )
                except Exception as retry_error:
                    logger.error(f"Retry without dtype failed: {retry_error}")
                    logger.error(
                        "\n" + "="*80 + "\n"
                        "SOLUTION: Upgrade transformers to support dtype parameter\n"
                        "="*80 + "\n"
                        "Your transformers library is too old to support dtype parameter for Qwen3.\n"
                        "\n"
                        "Fix by running on your GPU server:\n"
                        "  pip install --upgrade 'transformers>=4.55.0'\n"
                        "\n"
                        "Or if using conda:\n"
                        "  conda install -c conda-forge 'transformers>=4.55.0'\n"
                        "\n"
                        "After upgrade, restart your script.\n"
                        "="*80
                    )
                    raise RuntimeError(
                        f"Cannot load model: transformers library too old (requires >=4.55.0 for full Qwen3 support)\n"
                        f"Original error: {type_err}"
                    ) from type_err
            else:
                # Different TypeError, re-raise
                raise
        except (KeyError, ValueError) as e:
            error_str = str(e).lower()

            # Check if this is a Qwen3 compatibility issue
            if 'qwen3' in error_str and 'does not recognize' in error_str:
                logger.warning(
                    f"Qwen3 model type not recognized by transformers library.\n"
                    f"  Attempting Qwen2 compatibility mode (architectures are nearly identical)..."
                )

                # Try Qwen2 compatibility mode
                try:
                    return _load_qwen3_as_qwen2(model_path_or_name, dtype)
                except Exception as qwen2_error:
                    logger.error(f"Qwen2 compatibility mode failed: {qwen2_error}")
                    logger.error(
                        "\n" + "="*80 + "\n"
                        "SOLUTION: Upgrade transformers to support Qwen3\n"
                        "="*80 + "\n"
                        "Your transformers library is too old to support Qwen3 models.\n"
                        "\n"
                        "Fix by running on your GPU server:\n"
                        "  pip install --upgrade 'transformers>=4.55.0'\n"
                        "\n"
                        "Or if using conda:\n"
                        "  conda install -c conda-forge 'transformers>=4.55.0'\n"
                        "\n"
                        "After upgrade, restart your script.\n"
                        "="*80
                    )
                    raise RuntimeError(
                        f"Cannot load Qwen3 model: transformers library too old (requires >=4.55.0)\n"
                        f"Original error: {e}"
                    ) from e
            else:
                # Different error, re-raise
                raise
    else:
        try:
            config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
            # Try with dtype first, fall back to without dtype if not supported
            try:
                return AutoModelForCausalLM.from_config(config, dtype=dtype)
            except TypeError as type_err:
                if "dtype" in str(type_err) or "unexpected keyword argument" in str(type_err):
                    logger.warning(
                        f"dtype parameter not supported for from_config in current transformers version.\n"
                        f"  Using model's default dtype instead..."
                    )
                    return AutoModelForCausalLM.from_config(config)
                else:
                    raise
        except (KeyError, ValueError) as e:
            if 'qwen3' in str(e).lower():
                logger.error(
                    "Qwen3 model type not supported. Please upgrade transformers:\n"
                    "  pip install --upgrade 'transformers>=4.55.0'"
                )
            raise


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def _reconstruct_perception_from_state_dict(perception_state_dict: dict, output_dim: int):
    """
    Reconstruct perception module from state_dict when config is incomplete.

    This function provides a fallback when checkpoint's perception config is missing
    preprocessor/encoder subconfigs. It uses a public ASR model to get the config structure,
    then loads the checkpoint weights.

    Args:
        perception_state_dict: Perception weights from checkpoint
        output_dim: LLM hidden size for perception output projection

    Returns:
        AudioPerceptionModule with checkpoint weights loaded
    """
    logger.info("Loading config from public ASR model: nvidia/canary-1b-v2")

    # Suppress ASR model's tokenizer initialization log
    asr_logger = logging.getLogger("nemo.collections.asr")
    original_level = asr_logger.level
    asr_logger.setLevel(logging.WARNING)

    try:
        # Load public ASR model to get config structure
        asr = load_pretrained_nemo(ASRModel, 'nvidia/canary-1b-v2').eval()

        # Create perception config from ASR model
        from omegaconf import DictConfig, OmegaConf

        perception_cfg = {
            'preprocessor': OmegaConf.to_container(asr.cfg.preprocessor, resolve=True),
            'encoder': OmegaConf.to_container(asr.cfg.encoder, resolve=True),
            'modality_adapter': {
                '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                'feat_in': asr.cfg.encoder.d_model,
                'd_model': asr.cfg.encoder.d_model,
                'n_layers': 2,  # Typical modality adapter depth
                'subsampling_factor': 1,
            },
            'output_dim': output_dim,
        }

        perception_cfg = DictConfig(perception_cfg)

        logger.info("Creating AudioPerceptionModule with reconstructed config")
        perception = AudioPerceptionModule(perception_cfg).train()

        # Load checkpoint weights
        missing_keys, unexpected_keys = perception.load_state_dict(perception_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"  - Missing keys during reconstruction: {len(missing_keys)}")
            for key in missing_keys[:3]:
                logger.warning(f"      {key}")
            if len(missing_keys) > 3:
                logger.warning(f"      ... +{len(missing_keys)-3} more")

        if unexpected_keys:
            logger.warning(f"  - Unexpected keys during reconstruction: {len(unexpected_keys)}")
            for key in unexpected_keys[:3]:
                logger.warning(f"      {key}")
            if len(unexpected_keys) > 3:
                logger.warning(f"      ... +{len(unexpected_keys)-3} more")

        return perception

    except Exception as e:
        logger.error(f"Failed to reconstruct perception from public ASR model: {e}")
        logger.error(
            "\n" + "="*80 + "\n"
            "SOLUTION: Checkpoint has incomplete perception config\n"
            "="*80 + "\n"
            "Your checkpoint is missing preprocessor/encoder config in perception.\n"
            "\n"
            "Possible fixes:\n"
            "1. Use a complete checkpoint with full perception config\n"
            "2. Ensure network access to download nvidia/canary-1b-v2 for config\n"
            "3. Provide pretrained_asr parameter explicitly in your config\n"
            "="*80
        )
        raise RuntimeError(
            f"Cannot reconstruct perception module from incomplete checkpoint.\n"
            f"Original error: {e}"
        ) from e
    finally:
        # Restore original logging level
        asr_logger.setLevel(original_level)


def _load_perception_config_from_external(config_path: str, output_dim: int) -> dict:
    """
    Load perception config from external source (ASR training config YAML or pretrained ASR .nemo).

    This function provides a fallback when checkpoint's perception config is incomplete.
    It supports two input formats:
    1. ASR training config YAML file: Extracts preprocessor/encoder from ASR training config
       (e.g., recipes/ConfTransfASR/configs/fast-conformer_hybrid_transformer_ctc_bpe_eos_bos_vocab32k_lr4.yaml)
    2. Pretrained ASR .nemo file: Loads ASR model and extracts preprocessor/encoder configs

    NOTE: This requires **ASR training config**, not SALM training config!
          SALM training config only has 'pretrained_asr' path without actual preprocessor/encoder configs.

    Args:
        config_path: Path to ASR training config YAML or pretrained ASR .nemo file
        output_dim: LLM hidden size for perception output projection

    Returns:
        Complete perception config dict with preprocessor, encoder, modality_adapter, output_dim

    Raises:
        RuntimeError: If config_path doesn't exist or has invalid format
        ValueError: If perception config cannot be extracted from the file

    Examples:
        # From ASR training config YAML
        cfg = _load_perception_config_from_external(
            "recipes/ConfTransfASR/configs/fast-conformer_hybrid_transformer_ctc_bpe_eos_bos_vocab32k_lr4.yaml",
            2560
        )

        # From pretrained ASR .nemo file
        cfg = _load_perception_config_from_external("nvidia/canary-1b-v2", 2560)

        # From local ASR .nemo file
        cfg = _load_perception_config_from_external("/path/to/canary-1b-v2.nemo", 2560)
    """
    from omegaconf import OmegaConf, DictConfig
    from pathlib import Path

    config_path_obj = Path(config_path)

    # Validate input path for local files
    if not config_path.startswith('nvidia/') and not config_path_obj.exists():
        raise RuntimeError(
            f"External config path does not exist: {config_path}\n"
            f"Please provide a valid path to:\n"
            f"  - ASR training config YAML file (e.g., recipes/ConfTransfASR/configs/fast-conformer_hybrid_transformer_ctc_bpe_eos_bos_vocab32k_lr4.yaml)\n"
            f"  - Pretrained ASR .nemo file (e.g., /path/to/model.nemo)\n"
            f"  - HuggingFace model name (e.g., nvidia/canary-1b-v2)\n"
            f"\n"
            f"NOTE: This should be an ASR training config, not a SALM training config.\n"
            f"      SALM training config only has 'pretrained_asr' path, not the actual ASR config."
        )

    # Case 1: ASR training config YAML file
    if config_path_obj.suffix in ['.yaml', '.yml']:
        logger.info(f"Loading perception config from ASR training config YAML: {config_path}")

        try:
            cfg = OmegaConf.load(config_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YAML config from {config_path}: {e}") from e

        # Extract preprocessor and encoder from ASR training config
        # Expected structure: cfg.model.preprocessor and cfg.model.encoder
        preprocessor_cfg = None
        encoder_cfg = None

        if 'model' in cfg:
            if 'preprocessor' in cfg.model and cfg.model.preprocessor is not None:
                preprocessor_cfg = cfg.model.preprocessor
            if 'encoder' in cfg.model and cfg.model.encoder is not None:
                encoder_cfg = cfg.model.encoder

        # Also check top-level keys (alternative structure)
        if preprocessor_cfg is None and 'preprocessor' in cfg and cfg.preprocessor is not None:
            preprocessor_cfg = cfg.preprocessor
        if encoder_cfg is None and 'encoder' in cfg and cfg.encoder is not None:
            encoder_cfg = cfg.encoder

        # Validate both are present
        if preprocessor_cfg is None:
            raise ValueError(
                f"Could not find preprocessor config in ASR training config YAML: {config_path}\n"
                f"Expected config structure (ASR training config):\n"
                f"  model:\n"
                f"    preprocessor:\n"
                f"      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor\n"
                f"      ...\n"
                f"    encoder:\n"
                f"      _target_: nemo.collections.asr.modules.ConformerEncoder\n"
                f"      ...\n"
                f"\n"
                f"NOTE: This should be an ASR training config, not a SALM training config.\n"
                f"      SALM training config only has 'pretrained_asr' path, not the actual ASR config."
            )

        if encoder_cfg is None:
            raise ValueError(
                f"Could not find encoder config in ASR training config YAML: {config_path}\n"
                f"Expected config structure (ASR training config):\n"
                f"  model:\n"
                f"    encoder:\n"
                f"      _target_: nemo.collections.asr.modules.ConformerEncoder\n"
                f"      ...\n"
                f"\n"
                f"NOTE: This should be an ASR training config, not a SALM training config."
            )

        # Create perception config from ASR config
        perception_dict = {
            'preprocessor': OmegaConf.to_container(preprocessor_cfg, resolve=True),
            'encoder': OmegaConf.to_container(encoder_cfg, resolve=True),
            'output_dim': output_dim,
        }

        # Create default modality adapter based on encoder d_model
        encoder_d_model = perception_dict['encoder'].get('d_model', 1024)
        perception_dict['modality_adapter'] = {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': encoder_d_model,
            'd_model': encoder_d_model,
            'n_layers': 2,  # Default adapter depth
            'subsampling_factor': 1,
        }

        logger.info("✓ Successfully loaded perception config from ASR training YAML")
        logger.info(f"  - preprocessor: {perception_dict['preprocessor'].get('_target_', 'unknown')}")
        logger.info(f"  - encoder: {perception_dict['encoder'].get('_target_', 'unknown')}")
        logger.info(f"  - encoder d_model: {encoder_d_model}")
        logger.info(f"  - output_dim: {output_dim}")

        return perception_dict

    # Case 2: Pretrained ASR .nemo file (local path or HuggingFace model name)
    else:
        logger.info(f"Loading perception config from pretrained ASR model: {config_path}")

        # Suppress ASR model's tokenizer initialization log
        asr_logger = logging.getLogger("nemo.collections.asr")
        original_level = asr_logger.level
        asr_logger.setLevel(logging.WARNING)

        try:
            # Load ASR model
            asr = load_pretrained_nemo(ASRModel, config_path).eval()

            # Extract preprocessor and encoder configs
            perception_dict = {
                'preprocessor': OmegaConf.to_container(asr.cfg.preprocessor, resolve=True),
                'encoder': OmegaConf.to_container(asr.cfg.encoder, resolve=True),
                'output_dim': output_dim,
            }

            # If modality_adapter exists in original config, preserve it
            # Otherwise, create a default one based on encoder output dimension
            if hasattr(asr.cfg, 'modality_adapter'):
                perception_dict['modality_adapter'] = OmegaConf.to_container(asr.cfg.modality_adapter, resolve=True)
            else:
                # Create default modality adapter config
                perception_dict['modality_adapter'] = {
                    '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                    'feat_in': asr.cfg.encoder.d_model,
                    'd_model': asr.cfg.encoder.d_model,
                    'n_layers': 2,  # Default adapter depth
                    'subsampling_factor': 1,
                }

            logger.info("✓ Successfully loaded perception config from ASR model")
            logger.info(f"  - preprocessor: {perception_dict['preprocessor'].get('_target_', 'unknown')}")
            logger.info(f"  - encoder: {perception_dict['encoder'].get('_target_', 'unknown')}")
            logger.info(f"  - output_dim: {output_dim}")

            return perception_dict

        except Exception as e:
            logger.error(f"Failed to load ASR model from {config_path}: {e}")
            raise RuntimeError(
                f"Cannot load perception config from ASR model: {config_path}\n"
                f"Original error: {e}"
            ) from e
        finally:
            # Restore original logging level
            asr_logger.setLevel(original_level)


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(
    model: torch.nn.Module,
    pretrained_weights: bool = True,
    checkpoint_path: str = None,
    external_config_path: str = None
):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel`` or from a checkpoint.
    The result is assigned to ``model.perception`` attribute and is trainable.

    Enhanced with checkpoint-only loading support and external config fallback for offline inference.

    Args:
        model: SALM model instance to setup perception module for
        pretrained_weights: Whether to load pretrained weights (default: True)
        checkpoint_path: Optional path to SALM checkpoint for direct perception loading.
                        If provided, loads perception config and weights from checkpoint,
                        bypassing pretrained_asr file requirement. This enables offline
                        inference with only the checkpoint file.
        external_config_path: Optional path to external config source for incomplete checkpoints.
                             Supports two formats:
                             1. ASR training config YAML file (e.g., recipes/ConfTransfASR/configs/fast-conformer_hybrid_transformer_ctc_bpe_eos_bos_vocab32k_lr4.yaml)
                             2. Pretrained ASR .nemo file or HuggingFace model (e.g., nvidia/canary-1b-v2)
                             Used as fallback when checkpoint's perception config is incomplete.

                             NOTE: This should be an ASR training config, not a SALM training config.
                                   SALM training config only has 'pretrained_asr' path without actual preprocessor/encoder configs.

    Modes:
        1. Checkpoint mode (checkpoint_path provided):
           - Extracts perception config from checkpoint
           - If config incomplete and external_config_path provided:
             → Loads complete config from external source
             → Merges with checkpoint config (preserves modality_adapter if present)
           - If config incomplete and no external_config_path:
             → Falls back to reconstructing from state_dict using public ASR model
           - Initializes AudioPerceptionModule with complete config
           - Loads perception weights from checkpoint state_dict
           - ✓ No pretrained_asr file needed (offline-friendly with external config)

        2. Pretrained mode (pretrained_weights=True, no checkpoint_path):
           - Loads ASR model from cfg.pretrained_asr
           - Extracts config and weights from ASR model
           - Standard training workflow

        3. Random init mode (pretrained_weights=False):
           - Not yet supported, raises NotImplementedError

    Examples:
        # Standard checkpoint loading (complete config)
        setup_speech_encoder(model, checkpoint_path="checkpoint.ckpt")

        # Checkpoint with external ASR training config
        setup_speech_encoder(
            model,
            checkpoint_path="checkpoint.ckpt",
            external_config_path="recipes/ConfTransfASR/configs/fast-conformer_hybrid_transformer_ctc_bpe_eos_bos_vocab32k_lr4.yaml"
        )

        # Checkpoint with external ASR .nemo file
        setup_speech_encoder(
            model,
            checkpoint_path="checkpoint.ckpt",
            external_config_path="/path/to/canary-1b-v2.nemo"
        )

        # Checkpoint with HuggingFace ASR model
        setup_speech_encoder(
            model,
            checkpoint_path="checkpoint.ckpt",
            external_config_path="nvidia/canary-1b-v2"
        )
    """
    # Mode 1: Checkpoint-only loading (offline inference)
    if checkpoint_path is not None:
        logger.info(f"Setting up speech encoder from checkpoint: {checkpoint_path}")

        # Load checkpoint - supports both .ckpt and .nemo formats
        checkpoint_path_obj = Path(checkpoint_path)
        suffix = checkpoint_path_obj.suffix.lower()

        if suffix == '.nemo':
            # Load from .nemo archive (tar file containing model_weights.ckpt and model_config.yaml)
            import tarfile
            import tempfile
            from omegaconf import OmegaConf

            logger.info(f"Loading perception from .nemo file: {checkpoint_path}")
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract .nemo archive
                try:
                    with tarfile.open(checkpoint_path, "r:") as tar:
                        tar.extractall(tmpdir)
                except tarfile.ReadError:
                    with tarfile.open(checkpoint_path, "r:gz") as tar:
                        tar.extractall(tmpdir)

                # Load config
                config_path = Path(tmpdir) / "model_config.yaml"
                if not config_path.exists():
                    raise RuntimeError(f"model_config.yaml not found in .nemo archive: {checkpoint_path}")
                nemo_cfg = OmegaConf.load(config_path)

                # Load weights
                weights_path = Path(tmpdir) / "model_weights.ckpt"
                if not weights_path.exists():
                    raise RuntimeError(f"model_weights.ckpt not found in .nemo archive: {checkpoint_path}")
                nemo_state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

            # Build checkpoint-like structure for unified processing
            checkpoint = {
                'hyper_parameters': {'cfg': OmegaConf.to_container(nemo_cfg, resolve=True)},
                'state_dict': nemo_state_dict
            }
        elif suffix == '.ckpt':
            # Load from PyTorch Lightning checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            raise ValueError(
                f"Unsupported checkpoint format: {suffix}. "
                f"Expected .nemo or .ckpt file."
            )

        # Extract perception config from checkpoint
        if 'hyper_parameters' not in checkpoint or 'cfg' not in checkpoint['hyper_parameters']:
            raise RuntimeError(
                f"Checkpoint does not contain hyperparameters/config: {checkpoint_path}\n"
                f"This checkpoint may be corrupted or in an unexpected format."
            )

        ckpt_cfg = checkpoint['hyper_parameters']['cfg']

        if 'perception' not in ckpt_cfg or ckpt_cfg['perception'] is None:
            raise RuntimeError(
                f"Checkpoint config does not contain perception configuration.\n"
                f"This checkpoint may not be a SALM model checkpoint."
            )

        # Extract perception weights from checkpoint
        perception_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('perception.'):
                # Remove 'perception.' prefix
                new_key = key[len('perception.'):]
                perception_state_dict[new_key] = value

        if not perception_state_dict:
            raise RuntimeError(
                f"No perception weights found in checkpoint.\n"
                f"Checkpoint may not contain trained perception module."
            )

        # Check if perception config is complete
        perception_cfg = ckpt_cfg['perception']
        has_preprocessor = 'preprocessor' in perception_cfg and perception_cfg['preprocessor'] is not None
        has_encoder = 'encoder' in perception_cfg and perception_cfg['encoder'] is not None

        if not has_preprocessor or not has_encoder:
            logger.warning("⚠ Checkpoint perception config is INCOMPLETE")
            logger.warning("  - preprocessor config: %s", "PRESENT" if has_preprocessor else "MISSING")
            logger.warning("  - encoder config: %s", "PRESENT" if has_encoder else "MISSING")

            # Try external config first (if provided)
            if external_config_path is not None:
                logger.info(f"Loading perception config from external source: {external_config_path}")

                try:
                    # Load complete config from external source
                    output_dim = model.llm.config.hidden_size if hasattr(model, 'llm') else 2560
                    external_cfg = _load_perception_config_from_external(external_config_path, output_dim)

                    # Merge with checkpoint config: use external for preprocessor/encoder,
                    # preserve checkpoint's modality_adapter if present
                    from omegaconf import DictConfig

                    complete_cfg = {
                        'preprocessor': external_cfg['preprocessor'],
                        'encoder': external_cfg['encoder'],
                        'output_dim': external_cfg['output_dim'],
                    }

                    # Preserve modality_adapter from checkpoint if it exists
                    if 'modality_adapter' in perception_cfg and perception_cfg['modality_adapter'] is not None:
                        logger.info("  - Preserving modality_adapter config from checkpoint")
                        complete_cfg['modality_adapter'] = perception_cfg['modality_adapter']
                    elif 'modality_adapter' in external_cfg:
                        logger.info("  - Using modality_adapter config from external source")
                        complete_cfg['modality_adapter'] = external_cfg['modality_adapter']

                    # Create AudioPerceptionModule with complete config
                    complete_cfg = DictConfig(complete_cfg)
                    model.perception = AudioPerceptionModule(complete_cfg).train()

                    # Load checkpoint weights
                    missing_keys, unexpected_keys = model.perception.load_state_dict(perception_state_dict, strict=False)

                    logger.info("✓ Perception module loaded with external config")
                    logger.info(f"  - Perception parameters: {sum(p.numel() for p in model.perception.parameters()):,}")

                    if missing_keys:
                        logger.warning(f"  - Missing keys: {len(missing_keys)}")
                        for key in missing_keys[:3]:
                            logger.warning(f"      {key}")
                        if len(missing_keys) > 3:
                            logger.warning(f"      ... +{len(missing_keys)-3} more")

                    if unexpected_keys:
                        logger.warning(f"  - Unexpected keys: {len(unexpected_keys)}")
                        for key in unexpected_keys[:3]:
                            logger.warning(f"      {key}")
                        if len(unexpected_keys) > 3:
                            logger.warning(f"      ... +{len(unexpected_keys)-3} more")

                    return

                except Exception as e:
                    logger.error(f"Failed to load external config: {e}")
                    logger.warning("Falling back to state_dict reconstruction...")
                    # Continue to fallback method below

            # Fallback: Reconstruct perception module from state_dict structure
            logger.info("Attempting to reconstruct perception from state_dict...")
            model.perception = _reconstruct_perception_from_state_dict(
                perception_state_dict,
                model.llm.config.hidden_size if hasattr(model, 'llm') else 2560
            )

            logger.info(f"✓ Perception module reconstructed from state_dict")
            logger.info(f"  - Perception parameters: {sum(p.numel() for p in model.perception.parameters()):,}")
            return

        # Normal path: Complete config available
        logger.info("Initializing AudioPerceptionModule from checkpoint config")

        # Store perception config in model.cfg for AudioPerceptionModule initialization
        with open_dict(model.cfg):
            model.cfg.perception = perception_cfg

        # Create AudioPerceptionModule with proper config (same as Mode 2)
        model.perception = AudioPerceptionModule(model.cfg.perception).train()

        # Load perception weights from checkpoint state_dict
        missing_keys, unexpected_keys = model.perception.load_state_dict(perception_state_dict, strict=False)

        logger.info(f"✓ Perception module loaded from checkpoint")
        logger.info(f"  - Perception parameters: {sum(p.numel() for p in model.perception.parameters()):,}")

        if missing_keys:
            logger.warning(f"  - Missing keys: {len(missing_keys)}")
            for key in missing_keys[:3]:
                logger.warning(f"      {key}")
            if len(missing_keys) > 3:
                logger.warning(f"      ... +{len(missing_keys)-3} more")

        if unexpected_keys:
            logger.warning(f"  - Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:3]:
                logger.warning(f"      {key}")
            if len(unexpected_keys) > 3:
                logger.warning(f"      ... +{len(unexpected_keys)-3} more")

        return

    # Mode 2: Standard pretrained loading (training workflow)
    if pretrained_weights:
        # Temporarily suppress ASR model's tokenizer initialization log to avoid confusion
        # (SALM uses LLM's tokenizer, not ASR's tokenizer)
        asr_logger = logging.getLogger("nemo.collections.asr")
        original_level = asr_logger.level
        asr_logger.setLevel(logging.WARNING)

        try:
            asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
        except RuntimeError as e:
            if "abstract class ASRModel" in str(e) or "abstract methods" in str(e):
                # This is likely a custom model class that's not available
                # Try to load just the encoder and preprocessor
                logger.debug(f"Expected error - model class not available: {e}")
                logger.info("Loading encoder from custom ASR checkpoint...")
                try:
                    asr = load_custom_asr_encoder_only(model.cfg.pretrained_asr)
                    logger.info("Successfully loaded encoder from custom ASR model")
                except Exception as custom_e:
                    logger.error(f"Failed to load custom ASR encoder: {custom_e}")
                    logger.info(
                        "Consider using an official pretrained model:\n"
                        "  pretrained_asr: nvidia/canary-1b-v2\n"
                        "  pretrained_asr: nvidia/canary-1b-flash"
                    )
                    raise
            else:
                logger.error(f"Failed to load ASR model: {e}")
                logger.info(
                    "Please consider using an official pretrained model by setting:\n"
                    "  pretrained_asr: nvidia/canary-1b-v2\n"
                    "or\n"
                    "  pretrained_asr: nvidia/canary-1b-flash\n"
                    "in your configuration file."
                )
                raise
        finally:
            # Restore original logging level
            asr_logger.setLevel(original_level)

        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size

        # Use global import from line 24 (no local import needed)
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        model.perception.load_state_dict(asr.state_dict(), strict=False)
    else:
        # Mode 3: Random initialization (not supported)
        raise NotImplementedError("Random initialization of speech encoder is not yet supported")