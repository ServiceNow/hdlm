import torch

import os
import logging
from omegaconf import OmegaConf
import torch.optim as optim
import math


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def cfg_to_family(cfg) -> str:
    return cfg["model"]["name"].split("_")[0]


def get_run_train(cfg, logger = None):
    family = cfg_to_family(cfg)
    if family == "epsilon":
        import hdlm.epsilon_hybrid.run_train as run_train
        if logger:
            logger.info(f"Epsilon-hybrid training in {getattr(cfg, 'type', 'aligned')} mode")
    elif family == "gamma":
        import hdlm.gamma_hybrid.run_train as run_train
        if logger:
            logger.info(f"Gamma-hybrid training using graph type {cfg.graph.type}")
    else:
        raise NotImplementedError(f"Unknown family: {family}")
    return run_train.run_train


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, accelerator):
    if not ckpt_dir or not os.path.exists(ckpt_dir):
        if ckpt_dir:
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        else:
            logging.warning("No checkpoint path provided. Starting from scratch.")
        return state
    else:
        logging.warning(f"Checkpoint was found at {ckpt_dir}. Continuing the training")
        loaded_state = torch.load(ckpt_dir, map_location=accelerator.device)
        
        # Use Accelerate's methods to load model and optimizer states
        accelerator.unwrap_model(state['model']).load_state_dict(loaded_state['model'], strict=False)
        # state['optimizer'].load_state_dict(loaded_state['optimizer'])
        # state['ema'].load_state_dict(loaded_state['ema'])
        state['optimizer'] = loaded_state['optimizer']
        state['ema'] = loaded_state['ema']
        state['step'] = loaded_state['step']

        if "wandb_run_id" in loaded_state:
            state['wandb_run_id'] = loaded_state['wandb_run_id']
        
        return state


def smart_restore_checkpoint(ckpt_path, state, accelerator, model_type="auto"):
    """
    Smart checkpoint restoration that can handle both local and Hugging Face paths.
    
    Args:
        ckpt_path: Path to checkpoint (local or Hugging Face ID)
        state: Current state dictionary
        accelerator: Accelerator instance
        model_type: Model type for loading ("epsilon_hybrid", "gamma_hybrid", or "auto")
        
    Returns:
        Updated state dictionary
    """
    try:
        from hdlm.hf_utils import is_huggingface_path, smart_model_loader
        
        if is_huggingface_path(ckpt_path):
            logging.warning(f"Loading model from Hugging Face: {ckpt_path}")
            model, _, _, _, _ = smart_model_loader(ckpt_path, model_type)
            
            state['model'] = model
            return state
        else:
            return restore_checkpoint(ckpt_path, state, accelerator)
            
    except ImportError:
        logging.warning("Hugging Face utilities not available, falling back to local checkpoint loading")
        return restore_checkpoint(ckpt_path, state, accelerator)
    except Exception as e:
        logging.error(f"Error in smart checkpoint restoration: {e}")
        return restore_checkpoint(ckpt_path, state, accelerator)



def save_checkpoint(ckpt_dir, state, accelerator):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': accelerator.unwrap_model(state['model']).state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step'],
    }
    if 'wandb_run_id' in state:
        saved_state['wandb_run_id'] = state['wandb_run_id']
    
    accelerator.save(saved_state, ckpt_dir)


def compute_entropy(sentences, tokenizer, device='cpu', log_base=None):
    """
    Compute the entropy for a list of sentences based on token diversity.
    
    Args:
        sentences (list of str): List of sentences generated by the model.
        tokenizer: Tokenizer to convert sentences to token IDs.
        device (str or torch.device): Device to perform computations on.
        log_base (int, optional): Log base to use in entropy calculation. 
                                  If None, natural log is used.
    
    Returns:
        torch.Tensor: Tensor of entropy values, one per sentence.
    """
    entropies = []
    for sentence in sentences:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        L = len(token_ids)
        if L == 0:
            # If the sentence is empty, entropy is 0
            entropies.append(0.0)
            continue
        
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        
        _, counts = torch.unique(token_ids_tensor, return_counts=True)
        
        # Compute p_k = counts / L with higher precision
        p_k = counts / L
        
        # Compute p_k * log(p_k)
        if log_base == 2:
            log_p_k = torch.log2(p_k + 1e-8)
        elif log_base == 10:
            log_p_k = torch.log10(p_k + 1e-8)
        else:
            log_p_k = torch.log(p_k + 1e-8)  # Natural log
        
        # Compute entropy: -sum(p_k * log(p_k))
        entropy = -torch.sum(p_k * log_p_k)
        entropies.append(entropy.item())
    
    # Convert the list of entropies to a torch tensor with higher precision
    entropies_tensor = torch.tensor(entropies, device=device)
    return entropies_tensor


def print_binary_matrix(matrix):
    for row in matrix:
        print(" ".join("█" if val == 1 else "." for val in row))



def zero_shot_perplexity(accelerator, state, loss_fn, eval_ds, monte_carlo_timesteps=10):
    """
    Compute zero-shot perplexity with optional random sampling for large datasets.

    Args:
        accelerator: The distributed accelerator (e.g., from Hugging Face Accelerate).
        state: The model state.
        loss_fn: A function computing loss, accepting (state, batch).
        eval_ds: The evaluation dataset.
        monte_carlo_timesteps: Number of Monte Carlo timesteps for loss averaging.
        max_random_batches: (int or None) 
            - If `None` (or <= 0), iterate over the **entire** dataset.
            - If an integer (e.g., `100`), sample at most `max_random_batches` **random batches** 
              from `eval_ds` when possible.

    Returns:
        ppl (float): The computed perplexity.
        total_loss (float): The average loss.
    """
    import math
    import itertools
    import random
    import torch
    from tqdm import tqdm

    def count_batches(iterable):
        """Helper function to count the number of batches in eval_iter."""
        return sum(1 for _ in iterable)

    print(f"Zero-shot evaluation with {monte_carlo_timesteps} Monte Carlo timesteps.")
    eval_iter = iter(eval_ds)
    # Create two copies of eval_iter to count the number of batches
    eval_iter, eval_iter_copy = itertools.tee(eval_iter)

    try:
        total_batches = count_batches(eval_iter_copy)
        print(f"Total available batches: {total_batches}")
    except TypeError:
        # If eval_iter is a streaming dataset (generator), we cannot count batches in advance
        total_batches = None
        print("Total batches unknown (streaming dataset detected).")
    if total_batches > 50 and total_batches < 100:
        monte_carlo_timesteps = 100
    elif total_batches >= 100 and total_batches < 200:
        monte_carlo_timesteps = 50
    elif total_batches >= 200:
        monte_carlo_timesteps = 10
        
    total_loss = 0
    token_num = 0
    batch_num = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(eval_iter, desc="Evaluating", unit="batch")):
            cur_loss = 0
            for i in range(monte_carlo_timesteps):
                loss = loss_fn(state, batch)
                cur_loss += loss.sum()

            cur_loss /= monte_carlo_timesteps
            total_loss += cur_loss
            token_num += loss.numel()
            batch_num += 1

    accelerator.reduce(total_loss, reduction="mean")
    ppl = math.exp(min(total_loss / token_num, 100))

    return ppl, total_loss


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


SCHEDULER_ETA_MIN_EPSILON = 4e-5


def get_scheduler(config, optimizer, eta_min=0):
    def lr_lambda(current_step, warmup_steps, total_steps):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if config.optim.scheduler == 'linear':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / config.training.n_iters)
    elif config.optim.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.n_iters, eta_min=eta_min)
    elif config.optim.scheduler == 'lambda':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, config.optim.warmup, config.training.n_iters))
    else:
        raise NotImplementedError(
            f'Scheduler {config.optim.scheduler} not supported yet!')
    return scheduler
