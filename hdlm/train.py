import hydra
import os
import sys
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hdlm.utils as utils


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


@hydra.main(version_base=None, config_path="../configs", config_name="epsilon_hdlm")
def main(cfg):
    # You may store your WANDB_API_KEY here
    yaml_to_environ(
        "wandb_api_key_secret.yaml",
        {"WANDB_MODE": "offline"}
    )
    hydra_cfg = HydraConfig.get()

    if "load_dir" in cfg and cfg.load_dir:
        # Resuming from a previous run
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra", "hydra.yaml")
        if os.path.exists(hydra_cfg_path):
            cfg_loaded = utils.load_hydra_config_from_run(cfg.load_dir)

            # Update current config with loaded config
            cfg = OmegaConf.merge(cfg_loaded, cfg)

            # Set work_dir from loaded config
            work_dir = cfg.work_dir
            utils.makedirs(work_dir)
            print(f"Resuming training from load_dir: {cfg.load_dir}")
        else:
            raise FileNotFoundError(f"No hydra.yaml found in load_dir: {cfg.load_dir}")
    else:
        work_dir = hydra_cfg.run.dir
        utils.makedirs(work_dir)
        print(f"Starting new training run in work_dir: {work_dir}")

    # Update the config with the determined work_dir
    with open_dict(cfg):
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    run_train = utils.get_run_train(cfg, logger)
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    try:
        run_train(cfg)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.critical(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 
