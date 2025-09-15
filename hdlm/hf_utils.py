"""
Hugging Face compatibility utilities for loading models from both local and Hugging Face paths.

This module provides smart loading functions that automatically detect whether a path
refers to a Hugging Face model or a local checkpoint and loads accordingly.
"""

from itertools import chain
import os
import logging
import torch
from typing import Optional, Tuple, Any
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download, model_info
from accelerate import Accelerator
from diffusers.training_utils import EMAModel as ExponentialMovingAverage

import hdlm.gamma_hybrid.noise_lib as noise_lib
import hdlm.utils as utils
from hdlm.model.diffusion_lm import HDLM, HDLM_Gamma
from hdlm.metaschedule import make_simple_annealing, make_block_annealing, make_hybrid_annealing

logger = logging.getLogger(__name__)


def is_huggingface_path(path: str) -> bool:
    """
    Check if a path is a Hugging Face model identifier.
    
    Args:
        path: Path to check (can be local path or HF model ID)
        
    Returns:
        True if the path looks like a Hugging Face model ID
    """
    if '/' in path and not os.path.exists(path):
        try:
            model_info(path)
            return True
        except Exception:
            return path.count('/') == 1 and not path.startswith('./') and not path.startswith('/')
    return False


def load_config_from_hf(model_id: str) -> OmegaConf:
    """
    Load configuration from a Hugging Face model.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        OmegaConf configuration object
    """
    try:
        # Download config file from HF
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.yaml",
            cache_dir=None
        )
        return OmegaConf.load(config_path)
    except Exception as e:
        logger.warning(f"Could not load config.yaml from {model_id}: {e}")
        # Try to load config.json as fallback
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=None
            )
            config_dict = torch.load(config_path, map_location='cpu')
            return OmegaConf.create(config_dict)
        except Exception as e2:
            raise ValueError(f"Could not load configuration from {model_id}: {e2}")


def save_model_to_hf(model, config, save_dir: str, model_id: str, token: Optional[str] = None):
    """
    Save model and configuration to Hugging Face Hub in BD3-LMs format.
    
    Args:
        model: The model to save
        config: Configuration object
        save_dir: Local directory to save files before uploading
        model_id: Hugging Face model identifier
        token: Hugging Face token for authentication
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration in multiple formats
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as f:
        OmegaConf.save(config=config, f=f)
    
    # Also save as config.json for better HF compatibility
    config_json_path = os.path.join(save_dir, "config.json")
    config_dict = OmegaConf.to_container(config, resolve=True)
    import json
    with open(config_json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save model weights as pytorch_model.bin
    model_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer files (GPT-2 tokenizer for now)
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.save_pretrained(save_dir)
    
    # Create generation config
    generation_config = {
        "do_sample": True,
        "temperature": 1.0,
        "max_length": 100,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "model_type": "hdlm",
        "architectures": ["HDLM"] if hasattr(model, '__class__') and 'HDLM' in model.__class__.__name__ else ["MetaSEDD"]
    }
    
    generation_config_path = os.path.join(save_dir, "generation_config.json")
    with open(generation_config_path, 'w') as f:
        json.dump(generation_config, f, indent=2)
    
    # Create .gitattributes for large files
    gitattributes_content = """*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
"""
    gitattributes_path = os.path.join(save_dir, ".gitattributes")
    with open(gitattributes_path, 'w') as f:
        f.write(gitattributes_content)
    
    # Push to Hugging Face Hub
    model.push_to_hub(
        model_id,
        token=token,
        private=False,  # Set to True for private models
        commit_message="Upload HDLM model with complete HF integration"
    )
    
    logger.info(f"Model saved to Hugging Face Hub: {model_id}")
    logger.info("Files uploaded:")
    logger.info("  - config.yaml (OmegaConf format)")
    logger.info("  - config.json (HF compatible)")
    logger.info("  - pytorch_model.bin (model weights)")
    logger.info("  - tokenizer files (GPT-2 compatible)")
    logger.info("  - generation_config.json (generation settings)")
    logger.info("  - .gitattributes (LFS configuration)")


def load_epsilon_hybrid_model(model_path: str, device: str = "cuda") -> Tuple[Any, OmegaConf, str, Accelerator]:
    """
    Load an epsilon-hybrid model from either local path or Hugging Face.
    
    Args:
        model_path: Path to model (local or Hugging Face ID)
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, device, accelerator)
    """
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    if is_huggingface_path(model_path):
        logger.info(f"Loading epsilon-hybrid model from Hugging Face: {model_path}")
        return _load_epsilon_hybrid_from_hf(model_path, device, accelerator)
    else:
        logger.info(f"Loading epsilon-hybrid model from local path: {model_path}")
        return _load_epsilon_hybrid_from_local(model_path, device, accelerator)


def load_gamma_hybrid_model(model_path: str, device: str = "cuda") -> Tuple[Any, OmegaConf, str, Accelerator]:
    """
    Load a gamma-hybrid model from either local path or Hugging Face.
    
    Args:
        model_path: Path to model (local or Hugging Face ID)
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, device, accelerator)
    """
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    if is_huggingface_path(model_path):
        logger.info(f"Loading gamma-hybrid model from Hugging Face: {model_path}")
        return _load_gamma_hybrid_from_hf(model_path, device, accelerator)
    else:
        logger.info(f"Loading gamma-hybrid model from local path: {model_path}")
        return _load_gamma_hybrid_from_local(model_path, device, accelerator)


def _load_epsilon_hybrid_from_hf(model_id: str, device: str, accelerator: Accelerator) -> Tuple[Any, OmegaConf, str, Accelerator]:
    """Load epsilon-hybrid model from Hugging Face."""
    try:
        cfg = load_config_from_hf(model_id)
    except:
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=None
            )
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            cfg = OmegaConf.create(config_dict)
        except Exception as e:
            raise ValueError(f"Could not load configuration from {model_id}: {e}")
    
    model = HDLM.from_pretrained(model_id)
    
    optimizer = utils.get_optimizer(cfg, model.parameters())
    scheduler = utils.get_scheduler(cfg, optimizer, eta_min=utils.SCHEDULER_ETA_MIN_EPSILON)
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    model = model.to(device)
    model.eval()
    
    return model, cfg, device, accelerator


def _load_epsilon_hybrid_from_local(checkpoint_path: str, device: str, accelerator: Accelerator) -> Tuple[Any, OmegaConf, str, Accelerator]:
    """Load epsilon-hybrid model from local checkpoint."""
    config_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # Go up two levels from checkpoint.pth
    cfg = utils.load_hydra_config_from_run(config_dir)
    
    model = HDLM(cfg)
    
    optimizer = utils.get_optimizer(cfg, model.parameters())
    scheduler = utils.get_scheduler(cfg, optimizer, eta_min=utils.SCHEDULER_ETA_MIN_EPSILON)
    
    ema = ExponentialMovingAverage(model, decay=cfg.training.ema, update_after_step=2000)
    
    state = dict(optimizer=optimizer, scheduler=scheduler, model=model, ema=ema, step=0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    state = utils.restore_checkpoint(checkpoint_path, state, accelerator)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, cfg, device, accelerator


def _load_gamma_hybrid_from_hf(model_id: str, device: str, accelerator: Accelerator) -> Tuple[Any, OmegaConf, str, Accelerator]:
    """Load gamma-hybrid model from Hugging Face."""
    try:
        cfg = load_config_from_hf(model_id)
    except:
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=None
            )
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            cfg = OmegaConf.create(config_dict)
        except Exception as e:
            raise ValueError(f"Could not load configuration from {model_id}: {e}")
    
    model = HDLM_Gamma.from_pretrained(model_id)
    
    # Import noise_lib to get noise parameters (matching run_train.py)
    import hdlm.gamma_hybrid.noise_lib as noise_lib
    noise = noise_lib.get_noise(cfg)
    
    # Create optimizer with both model and noise parameters (matching run_train.py)
    from itertools import chain
    optimizer = utils.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
    scheduler = utils.get_scheduler(cfg, optimizer)
    
    model, noise, optimizer, scheduler = accelerator.prepare(model, noise, optimizer, scheduler)
    
    model = model.to(device)
    model.eval()
    
    return model, cfg, device, accelerator


def _load_gamma_hybrid_from_local(checkpoint_path: str, device: str, accelerator: Accelerator) -> Tuple[Any, OmegaConf, str, Accelerator]:
    """Load gamma-hybrid model from local checkpoint."""
    config_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # Go up two levels from checkpoint.pth
    cfg = utils.load_hydra_config_from_run(config_dir)
    
    model = HDLM_Gamma(cfg)
    
    # Import noise_lib to get noise parameters (matching run_train.py)
    import hdlm.gamma_hybrid.noise_lib as noise_lib
    noise = noise_lib.get_noise(cfg)
    
    # Create optimizer with both model and noise parameters (matching run_train.py)
    from itertools import chain
    optimizer = utils.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
    scheduler = utils.get_scheduler(cfg, optimizer)
    
    ema = ExponentialMovingAverage(model, decay=cfg.training.ema, update_after_step=2000)
    
    state = dict(optimizer=optimizer, scheduler=scheduler, model=model, noise=noise, ema=ema, step=0)
    model, noise, optimizer, scheduler = accelerator.prepare(model, noise, optimizer, scheduler)
    
    state = utils.restore_checkpoint(checkpoint_path, state, accelerator)
    
    model = model.to(device)
    model.eval()
    
    return model, cfg, device, accelerator


def create_metaschedule(cfg) -> Optional[Any]:
    """
    Create metaschedule based on configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Metaschedule object or None
    """
    if not hasattr(cfg, 'annealing'):
        return None
        
    if cfg.annealing.type == "simple":
        return make_simple_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
    elif cfg.annealing.type == "block":
        return make_block_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
    elif cfg.annealing.type == "hybrid":
        return make_hybrid_annealing(cfg.annealing.width, cfg.annealing.eval_tau, cfg.model.length)
    else:
        return None


def smart_model_loader(model_path: str, model_type: str = "auto", device: str = "cuda") -> Tuple[Any, OmegaConf, str, Accelerator, Optional[Any]]:
    """
    Smart model loader that automatically detects model type and loading method.
    
    Args:
        model_path: Path to model (local or Hugging Face ID)
        model_type: Model type ("epsilon_hybrid", "gamma_hybrid", or "auto")
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, device, accelerator, metaschedule)
    """
    if model_type == "auto":
        # Try to detect model type from path or configuration
        if is_huggingface_path(model_path):
            # For HF models, we need to load config first to determine type
            try:
                cfg = load_config_from_hf(model_path)
                # Check for model type indicators in config
                if hasattr(cfg, 'model_type'):
                    model_type = cfg.model_type
                elif hasattr(cfg, 'graph') and hasattr(cfg.graph, 'type'):
                    # Infer from graph type
                    if cfg.graph.type in ['QGamma', 'gamma']:
                        model_type = "gamma_hybrid"
                    else:
                        model_type = "epsilon_hybrid"
                else:
                    # Default to epsilon_hybrid
                    model_type = "epsilon_hybrid"
            except Exception as e:
                logger.warning(f"Could not determine model type from HF config: {e}")
                model_type = "epsilon_hybrid"
        else:
            # For local paths, try to load config to determine type
            try:
                config_dir = os.path.dirname(os.path.dirname(model_path))
                cfg = utils.load_hydra_config_from_run(config_dir)
                if hasattr(cfg, 'model_type'):
                    model_type = cfg.model_type
                elif hasattr(cfg, 'graph') and hasattr(cfg.graph, 'type'):
                    if cfg.graph.type in ['QGamma', 'gamma']:
                        model_type = "gamma_hybrid"
                    else:
                        model_type = "epsilon_hybrid"
                else:
                    model_type = "epsilon_hybrid"
            except Exception as e:
                logger.warning(f"Could not determine model type from local config: {e}")
                model_type = "epsilon_hybrid"
    
    # Load model based on type
    if model_type == "epsilon_hybrid":
        model, cfg, device, accelerator = load_epsilon_hybrid_model(model_path, device)
    elif model_type == "gamma_hybrid":
        model, cfg, device, accelerator = load_gamma_hybrid_model(model_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create metaschedule
    metaschedule = create_metaschedule(cfg)
    
    return model, cfg, device, accelerator, metaschedule


def load_model_and_config(checkpoint_path, config_path=None):
    """
    Smart model and configuration loader that handles both local and Hugging Face paths.

    Args:
        checkpoint_path: Path to model checkpoint (local or Hugging Face ID)
        config_path: Path to configuration directory (optional, used for local models)

    Returns:
        Tuple of (model, config, device, accelerator, metaschedule)
    """
    try:
        if is_huggingface_path(checkpoint_path):
            print(f"Loading model from Hugging Face: {checkpoint_path}")
            # For Hugging Face models, use smart_model_loader
            model, cfg, device, accelerator, metaschedule = smart_model_loader(
                checkpoint_path,
                model_type="auto",
                device="cuda"
            )

            # Load overrides if present in HF model
            try:
                from hdlm.hf_utils import load_config_from_hf
                hf_cfg = load_config_from_hf(checkpoint_path)
                if hasattr(hf_cfg, 'overrides'):
                    print("Overrides found in HF model config")
                    for override in hf_cfg.overrides:
                        override_dict = OmegaConf.from_dotlist([override])
                        cfg = OmegaConf.merge(cfg, override_dict)
            except Exception as e:
                print(f"Could not load overrides from HF model: {e}")

        else:
            print(f"Loading model from local path: {checkpoint_path}")

            # For local models, use the original approach
            if config_path is None:
                # Try to infer config path from checkpoint path
                config_path = os.path.dirname(os.path.dirname(checkpoint_path))
                print(f"Inferred config path: {config_path}")

            # Load config
            cfg = utils.load_hydra_config_from_run(config_path)

            # Load overrides if present
            overrides_path = os.path.join(config_path, ".hydra/overrides.yaml")
            if os.path.exists(overrides_path):
                overrides = OmegaConf.load(overrides_path)
                if OmegaConf.is_list(overrides):
                    for override in overrides:
                        override_dict = OmegaConf.from_dotlist([override])
                        cfg = OmegaConf.merge(cfg, override_dict)
                    print("Overrides applied from overrides.yaml")
                else:
                    print("overrides.yaml is not a list. Skipping overrides.")
            else:
                print("No overrides.yaml found. Using base config.")

            accelerator = Accelerator(mixed_precision='fp16')
            device = accelerator.device

            # Parts that differ
            if cfg['model']['name'] == 'epsilon_hdlm':
                model = HDLM(cfg)
                noise = None
                optimizer = utils.get_optimizer(cfg, model.parameters())
                scheduler = utils.get_scheduler(cfg, optimizer, eta_min=utils.SCHEDULER_ETA_MIN_EPSILON)
            elif cfg['model']['name'] == 'gamma_hdlm':
                model = HDLM_Gamma(cfg)
                noise = noise_lib.get_noise(cfg)
                optimizer = utils.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
                scheduler = utils.get_scheduler(cfg, optimizer)
            else:
                raise NotImplementedError(f"Unknown model name: {cfg['model']['name']}")

            # Back to shared code
            ema = ExponentialMovingAverage(model, decay=cfg.training.ema, update_after_step=2000)

            if noise:
                state = dict(optimizer=optimizer, scheduler=scheduler, model=model, noise=noise, ema=ema, step=0)
                model, noise, optimizer, scheduler = accelerator.prepare(model, noise, optimizer, scheduler)
            else:
                state = dict(optimizer=optimizer, scheduler=scheduler, model=model, ema=ema, step=0)
                model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

            state = utils.restore_checkpoint(checkpoint_path, state, accelerator)

            # Create metaschedule
            if cfg.annealing.type == "simple":
                metaschedule = make_simple_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
            elif cfg.annealing.type == "block":
                metaschedule = make_block_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
            elif cfg.annealing.type == "hybrid":
                metaschedule = make_hybrid_annealing(cfg.annealing.width, cfg.annealing.eval_tau, cfg.model.length)
            else:
                metaschedule = None

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()

        return model, cfg, device, accelerator, metaschedule
    except Exception as e:
        print(f"Error in smart model loading: {e}")
        raise

