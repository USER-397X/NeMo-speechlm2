#!/usr/bin/env python3
# Training script for SALM Canary-Qwen-2.5B ASR model
# Based on speechlm2 SALM recipe with lhotse shar format

import os
import socket
import sys
import logging

# Add NeMo path to sys.path
nemo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if nemo_path not in sys.path:
    sys.path.insert(0, nemo_path)

# Configure logging to reduce verbosity
logging.getLogger("nemo.core.classes.common").setLevel(logging.WARNING)
logging.getLogger("nemo.collections.asr").setLevel(logging.WARNING)

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

# Import SALM components directly to avoid peft dependency issue with duplex models
from nemo.collections.speechlm2.models.salm import SALM
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset
from nemo.collections.speechlm2.data.datamodule import DataModule
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_node_count():
    """Get number of nodes in distributed training using MPI."""
    try:
        from mpi4py import MPI
    except ImportError:
        logging.warning("mpi4py not installed, assuming single node")
        return 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Get the hostname of the current process
    hostname = socket.gethostname()

    # Gather all hostnames at the root process
    all_hostnames = comm.gather(hostname, root=0)

    # Root process computes number of unique hostnames (nodes)
    if rank == 0:
        unique_nodes = set(all_hostnames)
        num_nodes = len(unique_nodes)
    else:
        num_nodes = None

    # Broadcast the node count to all processes
    num_nodes = comm.bcast(num_nodes, root=0)

    return num_nodes


class SALMProgressBar(TQDMProgressBar):
    """
    Custom progress bar for SALM training with step-based display and comprehensive metrics.

    Displays:
        - Step progress (step/max_steps) instead of epoch-based
        - Training metrics: train_loss (4 decimals), lr (scientific), bs (integer)
        - GPU memory usage
        - Step timing

    Example output:
        Training: 23%|████| 1234/5000 [02:45<08:22, 3.5it/s, train_loss=2.1234, lr=3.00e-04, bs=4, GPU_Mem=45.23GB]
    """

    # Display format constants
    TRAIN_BAR_FORMAT = "Training: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    VAL_BAR_FORMAT = "Validation: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"

    # Metric format specifications
    METRIC_FORMATS = {
        'train_loss': lambda v: f"{v:.4f}",      # 4 decimals
        'lr': lambda v: f"{v:.2e}",              # Scientific notation
        'bs': lambda v: f"{int(v)}",             # Integer
        'step_time': lambda v: f"{v:.2f}s",      # 2 decimals + unit
    }

    # Keys to remove from display
    EXCLUDE_KEYS = {'v_num'}

    # Key renames
    KEY_RENAMES = {
        'train_step_timing in s': 'step_time',
        'batch size': 'bs',
    }

    def init_train_tqdm(self):
        """Initialize training progress bar with custom format."""
        bar = super().init_train_tqdm()
        bar.bar_format = self.TRAIN_BAR_FORMAT
        return bar

    def init_validation_tqdm(self):
        """Initialize validation progress bar with custom format."""
        bar = super().init_validation_tqdm()
        bar.bar_format = self.VAL_BAR_FORMAT
        return bar

    def get_metrics(self, trainer, pl_module):
        """
        Collect and format metrics for progress bar display.

        Returns:
            dict: Formatted metrics ready for display
        """
        items = super().get_metrics(trainer, pl_module)

        # Remove excluded keys
        for key in self.EXCLUDE_KEYS:
            items.pop(key, None)

        # Rename keys
        for old_key, new_key in self.KEY_RENAMES.items():
            if old_key in items:
                items[new_key] = items.pop(old_key)

        # Format specific metrics
        for metric_key, format_fn in self.METRIC_FORMATS.items():
            if metric_key in items:
                items[metric_key] = format_fn(self._extract_value(items[metric_key]))

        # Format generic loss/lr metrics (backward compatibility)
        self._format_loss_lr_metrics(items)

        # Add GPU memory usage
        self._add_gpu_memory(items)

        return items

    # Helper methods
    @staticmethod
    def _extract_value(value):
        """Extract numeric value from Tensor or scalar."""
        return value.item() if isinstance(value, torch.Tensor) else value

    def _format_loss_lr_metrics(self, items):
        """Format loss and learning rate metrics with scientific notation."""
        for key in list(items.keys()):
            if key not in self.METRIC_FORMATS:
                if key.startswith(("loss", "lr")) or key.endswith(("loss", "lr")):
                    try:
                        value = self._extract_value(items[key])
                        items[key] = f"{value:.2e}"
                    except (ValueError, TypeError, AttributeError):
                        pass

    @staticmethod
    def _add_gpu_memory(items):
        """Add GPU memory usage to metrics if CUDA is available."""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            mem_used_gb = (total - free) / (1024 ** 3)
            items["GPU_Mem"] = f"{mem_used_gb:.2f}GB"


@hydra_runner(
    config_path="configs/",
    config_name="salm_canary_qwen_2.5b",
)
def main(cfg: DictConfig):
    """Main training function for SALM model."""

    # Only show main logs on rank 0
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Configure logging based on rank
    if rank != 0:
        # Reduce logging on non-primary ranks
        logging.getLogger().setLevel(logging.WARNING)

    # Log configuration (only in debug mode to reduce verbosity)
    logging.debug(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if rank == 0:
        logging.info("=" * 80)
        logging.info("SALM Training - Canary-Qwen ASR Model")
        logging.info("=" * 80)

    # Configure trainer
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)

    # Auto-detect number of nodes if not specified
    if cfg.trainer.num_nodes == -1:
        nodes = get_node_count()
        trainer_cfg['num_nodes'] = nodes
        logging.info(f'Auto-detected {nodes} node(s)')
    else:
        nodes = cfg.trainer.num_nodes
        trainer_cfg['num_nodes'] = nodes

    # Auto-detect number of GPUs if not specified
    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()
        trainer_cfg['devices'] = cfg.trainer.devices

    if rank == 0:
        logging.info(f"Training Configuration: {cfg.trainer.devices} GPU(s) on {nodes} node(s)")

    # ============================================================================
    # SALM Checkpoint Resume Configuration
    # ============================================================================
    # Two modes available:
    #   - weights_only: Load model weights only, fresh optimizer/scheduler
    #   - full_resume:  Load everything (weights + optimizer + scheduler + step)
    # ============================================================================
    resume_from_salm = cfg.model.get('resume_from_salm', None)
    resume_mode = cfg.model.get('resume_mode', 'weights_only')

    # ============================================================================
    # Common typo detection and warning
    # ============================================================================
    # Check for common typos in yaml config that can cause silent failures
    common_typos = {
        'resume_model': 'resume_mode',
        'resume_salm': 'resume_from_salm',
        'resumemode': 'resume_mode',
        'resumefromsalm': 'resume_from_salm',
    }
    for typo, correct in common_typos.items():
        if typo in cfg.model:
            logging.warning("=" * 80)
            logging.warning(f"⚠️  YAML CONFIG WARNING: Detected '{typo}' in config!")
            logging.warning(f"    Did you mean '{correct}'?")
            logging.warning(f"    Current value of '{typo}': {cfg.model.get(typo)}")
            logging.warning(f"    This key is being IGNORED - using default for '{correct}'")
            logging.warning("=" * 80)

    # Validate resume_mode (only when resume_from_salm is set)
    if resume_from_salm and resume_mode not in ['weights_only', 'full_resume']:
        raise ValueError(
            f"Invalid resume_mode: '{resume_mode}'. "
            f"Must be 'weights_only' or 'full_resume' when resume_from_salm is set."
        )

    # For full_resume mode, configure exp_manager to use the checkpoint
    if resume_from_salm and resume_mode == 'full_resume':
        if rank == 0:
            logging.info("=" * 80)
            logging.info("[FULL RESUME MODE] Configuring complete training state restoration")
            logging.info(f"  Checkpoint: {resume_from_salm}")
            logging.info("  Will restore: model weights + optimizer + scheduler + global_step + epoch")
            logging.info("=" * 80)

        # Validate: full_resume requires .ckpt file (not .nemo)
        if resume_from_salm.endswith('.nemo'):
            raise ValueError(
                f"full_resume mode requires a .ckpt file, but got .nemo file: {resume_from_salm}\n"
                f".nemo files do not contain optimizer/scheduler state.\n"
                f"Use resume_mode: weights_only for .nemo files."
            )

        # Set exp_manager to use the checkpoint for full resume
        # This will make PyTorch Lightning restore the complete training state
        with open_dict(cfg):
            if 'exp_manager' not in cfg:
                cfg.exp_manager = {}
            cfg.exp_manager.resume_from_checkpoint = resume_from_salm
            cfg.exp_manager.resume_if_exists = True
            cfg.exp_manager.resume_ignore_no_checkpoint = True  # Allow resume even if log_dir is different

    elif resume_from_salm and resume_mode == 'weights_only':
        if rank == 0:
            logging.info("=" * 80)
            logging.info("[WEIGHTS ONLY MODE] Loading model weights with fresh optimizer")
            logging.info(f"  Checkpoint: {resume_from_salm}")
            logging.info("  Will restore: model weights only")
            logging.info("  Fresh start:  optimizer, scheduler, global_step (starts from 0)")
            logging.info("=" * 80)

    # Initialize trainer with custom progress bar
    trainer = Trainer(**trainer_cfg, callbacks=[SALMProgressBar()])

    # Setup experiment manager (handles full_resume if configured above)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Initialize model
    # When resuming from SALM checkpoint, pass the checkpoint path to SALM constructor
    # so it can load perception config and weights from the checkpoint instead of pretrained_asr.
    # This is needed for BOTH modes:
    #   - weights_only: perception loaded here, then load_weights_from() loads rest
    #   - full_resume: perception loaded here, then PTL loads full state (overwrites weights)
    if rank == 0:
        if resume_from_salm:
            logging.info(f"Initializing SALM model (resume_from_salm={resume_from_salm}, mode={resume_mode})...")
        else:
            logging.info("Initializing SALM model...")
    with trainer.init_module():
        model = SALM(
            OmegaConf.to_container(cfg.model, resolve=True),
            resume_from_salm=resume_from_salm  # Pass in both modes for perception init
        )

    # ============================================================================
    # Load weights from SALM checkpoint (weights_only mode)
    # ============================================================================
    # Note: When resume_from_salm is set, SALM.__init__() already loads perception weights
    # from the checkpoint. Here we load the remaining weights (LLM, embed_tokens, LoRA, etc.)
    #
    # For full_resume mode, we skip this step because PyTorch Lightning will restore
    # the complete state (including optimizer, scheduler, global_step) during trainer.fit()
    if resume_from_salm and resume_mode == 'weights_only':
        if rank == 0:
            logging.info(f"[WEIGHTS ONLY] Loading remaining weights from: {resume_from_salm}")
        model.load_weights_from(resume_from_salm)
    elif resume_from_salm and resume_mode == 'full_resume':
        if rank == 0:
            logging.info(f"[FULL RESUME] Model weights will be loaded by PyTorch Lightning during trainer.fit()")
            logging.info(f"  Checkpoint: {resume_from_salm}")
            logging.info(f"  Will restore: model weights, optimizer state, scheduler state, global_step")

    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model initialized successfully")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logging.info(f"  LLM Tokenizer: {model.tokenizer.__class__.__name__} with {model.tokenizer.vocab_size:,} tokens")
        logging.info(f"  Audio locator tag: '{model.audio_locator_tag}' (ID: {model.audio_locator_tag_id})")

    # Setup data module
    if rank == 0:
        logging.info("Setting up data module...")

    # Convert shar_path to input_cfg format if needed (for backward compatibility)
    with open_dict(cfg.data):
        if "train_ds" in cfg.data and "shar_path" in cfg.data.train_ds:
            shar_paths = cfg.data.train_ds.shar_path

            # Get context prompt from config, or use official default
            context_prompt = cfg.data.train_ds.get('asr_context_prompt', 'Transcribe the following: ')

            # ENHANCED: Get custom metadata text selection flags from config
            # These flags control priority-based reference text selection from lhotse shar custom fields
            # - use_itn: Enable ITN (Inverse Text Normalization) text usage (default: True)
            # - use_whisper_result: Enable whisper_result ASR output usage (default: False)
            # Matching se-trainer implementation for consistent metadata handling
            use_itn = cfg.data.train_ds.get('use_itn', True)
            use_whisper_result = cfg.data.train_ds.get('use_whisper_result', False)

            # ENHANCED: Get text case normalization flags from config (defaults to True)
            # These flags control text case normalization applied to all selected reference text
            # - convert_all_uppercase_to_lowercase: Convert all-uppercase text to lowercase (e.g., "HELLO" → "hello")
            # - capitalize_first_letter: Capitalize first letter if all lowercase (e.g., "hello" → "Hello")
            convert_all_uppercase_to_lowercase = cfg.data.train_ds.get('convert_all_uppercase_to_lowercase', True)
            capitalize_first_letter = cfg.data.train_ds.get('capitalize_first_letter', True)

            if rank == 0 and (not use_itn or not use_whisper_result):
                logging.info(f"Custom metadata text selection: use_itn={use_itn}, use_whisper_result={use_whisper_result}")

            if rank == 0 and (convert_all_uppercase_to_lowercase or capitalize_first_letter):
                logging.info(f"Text case normalization: convert_all_uppercase_to_lowercase={convert_all_uppercase_to_lowercase}, capitalize_first_letter={capitalize_first_letter}")

            # Create input_cfg list
            input_cfg_list = []
            for shar_item in shar_paths:
                if isinstance(shar_item, (list, ListConfig)) and len(shar_item) >= 1:
                    shar_dir = shar_item[0]
                    weight = shar_item[1] if len(shar_item) >= 2 else 1.0
                    language = shar_item[2] if len(shar_item) >= 3 else 'en'
                    metric = shar_item[3] if len(shar_item) >= 4 else 'wer'
                    # ENHANCED: Extract optional meta_db_path from 5th element (index 4)
                    # Format: [shar_dir, weight, language, metric, meta_db_path]
                    # This maintains 100% backward compatibility - meta_db_path is optional
                    meta_db_path = shar_item[4] if len(shar_item) >= 5 else None

                    # Build tags dictionary
                    tags = {
                        'context': context_prompt,
                        'lang': language,
                        'metric': metric,
                    }

                    # Add meta_db_path to tags if provided
                    # This allows downstream processing to access external metadata if needed
                    if meta_db_path:
                        tags['meta_db_path'] = meta_db_path
                        if rank == 0:
                            logging.debug(f"Dataset '{shar_dir}' has meta_db_path: {meta_db_path}")

                    input_cfg_list.append({
                        'type': 'lhotse_as_conversation',
                        'shar_path': shar_dir,
                        'weight': weight,
                        'audio_locator_tag': cfg.model.audio_locator_tag,
                        'token_equivalent_duration': cfg.data.train_ds.token_equivalent_duration,
                        'prompt_format': cfg.model.prompt_format,  # Add prompt_format here
                        'tags': tags,
                        # ENHANCED: Add custom metadata text selection flags to input_cfg
                        # These are propagated to read_lhotse_as_conversation() in cutset.py
                        'use_itn': use_itn,
                        'use_whisper_result': use_whisper_result,
                        # ENHANCED: Add text case normalization flags to input_cfg
                        # These are propagated to read_lhotse_as_conversation() → cut_to_conversation() → get_reference_text_with_priority()
                        'convert_all_uppercase_to_lowercase': convert_all_uppercase_to_lowercase,
                        'capitalize_first_letter': capitalize_first_letter,
                    })

            # Only update if we have valid configs
            if input_cfg_list:
                if rank == 0:
                    logging.info(f"Converted {len(input_cfg_list)} shar_path entries to input_cfg format")
                cfg.data.train_ds.input_cfg = input_cfg_list
                # Remove the old shar_path configuration
                del cfg.data.train_ds.shar_path
            else:
                logging.warning("No valid shar_path entries found, keeping original config")

            cfg.data.train_ds.use_lhotse = True

    # Create dataset and datamodule
    # Fix the prompt_format interpolation issue in all dataset configs
    with open_dict(cfg.data):
        # Fix training dataset prompt_format if it contains interpolation
        if 'train_ds' in cfg.data and 'prompt_format' in cfg.data.train_ds:
            cfg.data.train_ds.prompt_format = cfg.model.prompt_format

        # Replace ${model.prompt_format} with actual value for validation datasets
        if 'validation_ds' in cfg.data:
            if 'prompt_format' in cfg.data.validation_ds:
                cfg.data.validation_ds.prompt_format = cfg.model.prompt_format
            # Also update for each individual dataset if needed
            if 'datasets' in cfg.data.validation_ds:
                for dataset_name in cfg.data.validation_ds.datasets:
                    if 'prompt_format' in cfg.data.validation_ds.datasets[dataset_name]:
                        cfg.data.validation_ds.datasets[dataset_name].prompt_format = cfg.model.prompt_format

        # Same for test datasets if they exist
        if 'test_ds' in cfg.data:
            if 'prompt_format' in cfg.data.test_ds:
                cfg.data.test_ds.prompt_format = cfg.model.prompt_format
            if 'datasets' in cfg.data.test_ds:
                for dataset_name in cfg.data.test_ds.datasets:
                    if 'prompt_format' in cfg.data.test_ds.datasets[dataset_name]:
                        cfg.data.test_ds.datasets[dataset_name].prompt_format = cfg.model.prompt_format

    data_cfg = cfg.data

    # Get prompt format from model config
    prompt_format = None
    if hasattr(cfg.model, 'prompt_format'):
        from nemo.collections.common.prompts import PromptFormatter
        prompt_format = PromptFormatter.resolve(cfg.model.prompt_format)(model.tokenizer)

    dataset = SALMDataset(tokenizer=model.tokenizer, prompt_format=prompt_format)
    datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)

    # Start training
    if rank == 0:
        logging.info("-" * 80)
        logging.info("Starting training...")
        logging.info("-" * 80)
    trainer.fit(model, datamodule)

    # Run testing if test dataset is provided
    if hasattr(cfg.data, 'test_ds') and cfg.data.test_ds is not None:
        if rank == 0:
            logging.info("Running test evaluation...")
        trainer.test(model, datamodule)


if __name__ == '__main__':
    main()