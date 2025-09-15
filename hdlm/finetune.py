import hydra
import os
import sys
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import hdlm.utils as utils
from hdlm.hf_utils import load_config_from_hf


def yaml_to_environ(
        yaml_file_name: str, 
        default: dict[str, str]={}
    ):
    """Load environment variables from YAML file."""
    if os.path.exists(yaml_file_name):
        with open(yaml_file_name, "rt") as fp:
            dictionary = yaml.safe_load(fp)
    else:
        dictionary = default
    for key, value in dictionary.items():
        os.environ[key] = str(value)


@hydra.main(version_base=None, config_path="../configs", config_name="epsilon_finetune_from_hf")
def main(cfg):
    """
    Main entry point for fine-tuning from Hugging Face models.
    
    This script loads a pre-trained model from Hugging Face and fine-tunes it.
    """
    # Setup WandB API key and environment
    yaml_to_environ(
        "wandb_api_key_secret.yaml",
        {"WANDB_MODE": "offline"}
    )
    
    hydra_cfg = HydraConfig.get()
    run_train = utils.get_run_train(cfg)

    if hasattr(cfg, "hf_model_id") and cfg.hf_model_id:
        print(f"Loading HuggingFace config for model: {cfg.hf_model_id}")
        hf_cfg = load_config_from_hf(cfg.hf_model_id)
        merged_cfg = OmegaConf.merge(hf_cfg, cfg)
        cfg = merged_cfg

    work_dir = hydra_cfg.run.dir
    utils.makedirs(work_dir)
    print(f"Starting fine-tuning run in work_dir: {work_dir}")

    with open_dict(cfg):
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    # Log fine-tuning configuration
    logger.info(f"Fine-tuning from Hugging Face model: {cfg.hf_model_id}")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    try:
        run_train(cfg)
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.critical(f"Fine-tuning failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
