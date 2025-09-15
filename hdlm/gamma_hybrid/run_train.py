import os
from itertools import chain
import sys
import numpy as np
import torch
import torch.nn.functional as F

import hdlm.utils as utils
import hdlm.data as data
from hdlm.model.diffusion_lm import HDLM_Gamma
from diffusers.training_utils import EMAModel as ExponentialMovingAverage

import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hdlm.gamma_hybrid.losses as losses
import hdlm.gamma_hybrid.graph_lib as graph_lib
import hdlm.gamma_hybrid.noise_lib as noise_lib

import hdlm.gamma_hybrid.sampling as sampling
import hdlm.gamma_hybrid.sampling_sar as sampling_sar

from hdlm.metaschedule import make_simple_annealing, make_block_annealing, make_hybrid_annealing

torch.backends.cudnn.benchmark = True

def run_train(cfg):
    work_dir = cfg.work_dir

    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    safe_tensor = os.path.join(work_dir, 'safe_tensor')
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision='fp16',
        log_with="wandb",
        project_dir=os.path.join(work_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))
        utils.makedirs(safe_tensor)

    if accelerator.is_main_process:
        logger = get_logger(__file__)

    def mprint(msg):
        if accelerator.is_main_process:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = accelerator.device

    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")
    cfg.ngpus = accelerator.num_processes

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    train_ds, eval_ds = data.get_dataloaders(cfg, distributed=accelerator.use_distributed)

    graph = graph_lib.get_graph(cfg, device)
    
    score_model = HDLM_Gamma(cfg)
    state = dict(model=score_model, step=0)  # only minimal, rest added below

    if hasattr(cfg, 'hf_model_id') and cfg.hf_model_id:
        mprint(f"Loading model from Hugging Face: {cfg.hf_model_id}")
        try:
            state = utils.smart_restore_checkpoint(
                cfg.hf_model_id, 
                state, 
                accelerator, 
                model_type="epsilon_hybrid"
            )
            # Reset step to 0 for fine-tuning (or keep if you want to continue from HF step)
            if getattr(cfg, 'reset_step_for_finetuning', True):
                state['step'] = 0
                mprint("Reset training step to 0 for fine-tuning")
        except Exception as e:
            mprint(f"Failed to load from Hugging Face: {e}")
            mprint("Falling back to local checkpoint loading...")
            state = utils.smart_restore_checkpoint(checkpoint_meta_dir, state, accelerator, model_type="epsilon_hybrid")
    else:
        state = utils.smart_restore_checkpoint(checkpoint_meta_dir, state, accelerator, model_type="epsilon_hybrid")
    score_model = state['model']
    
    noise = noise_lib.get_noise(cfg)
    sampling_eps = cfg.annealing.sampling_eps

    def calculate_perplexity(cfg, score_model, device, mc_num=50, ema=None):
        """
        Calculate zero-shot perplexity using the gamma_hybrid zeroshot calculator.
        
        Args:
            cfg: Configuration object
            score_model: The model to evaluate
            device: Device to run on
            mc_num: Number of Monte Carlo samples
            ema: EMA model for evaluation (optional)
            
        Returns:
            tuple: (perplexity, nll_type, eval_type_name)
        """
        from hdlm.zeroshot_ppl import ZeroShot_calculator
        
        _, fresh_eval_ds = data.get_dataloaders(cfg, distributed=False)
        
        if hasattr(score_model, 'module'):
            model_config = score_model.module.config
        else:
            model_config = score_model.config
            
        token_dim = model_config.tokenizer.tokens + 1 if hasattr(model_config, 'tokenizer') else 50258
        
        nll_type = 'mc'
        training_type = getattr(cfg, 'type', 'aligned')
        
        was_training = score_model.training
        score_model.eval()
        
        original_params = None
        if ema is not None:
            original_params = [p.clone().detach() for p in score_model.parameters()]
            ema.copy_to(score_model.parameters())
        
        try:
            zeroshot_calculator = ZeroShot_calculator(
                model=score_model, 
                device=device,
                nll_type='mc',
                mask_id=token_dim - 1,  # Critical: Set proper mask_id
                mc_num=mc_num,
                batch_size=cfg.training.batch_size // (cfg.ngpus * cfg.gradient_accumulation_steps),  
                sampling_eps=1e-3,  
                mode='aligned' if training_type == 'aligned' else 'shifted',
                disable_tqdm=True
            )
            
            with torch.no_grad():
                ppl = zeroshot_calculator.evaluate_perplexity_from_dataloader(
                    fresh_eval_ds, 
                    nll_type=nll_type,
                )
            
            return ppl, nll_type, training_type
            
        finally:
            if original_params is not None:
                for param, original_param in zip(score_model.parameters(), original_params):
                    param.data.copy_(original_param.data)
            
            if was_training:
                score_model.train()

    # build optimization state
    optimizer = utils.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scheduler = utils.get_scheduler(cfg, optimizer)
    mprint(f"Scheduler: {scheduler}")

    # meta-schedule
    if cfg.annealing.type == "simple":
        meta_schedule = make_simple_annealing(cfg.annealing.width, cfg.annealing.tau)
        eval_meta_schedule = make_simple_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
    elif cfg.annealing.type == "block":
        meta_schedule = make_block_annealing(cfg.annealing.width, cfg.annealing.tau)
        eval_meta_schedule = make_block_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
    elif cfg.annealing.type == "hybrid":
        meta_schedule = make_hybrid_annealing(cfg.annealing.width, cfg.annealing.tau, cfg.model.length)
        eval_meta_schedule = make_block_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
    else:
        raise ValueError(f"Annealing type {cfg.annealing.type} not supported for now.")
    
    
    # Prepare everything with Accelerate
    score_model, noise, optimizer, scheduler, train_ds, eval_ds = accelerator.prepare(
        score_model, noise, optimizer, scheduler, train_ds, eval_ds
    )

    train_ds = data.cycle_loader(train_ds)
    # eval_ds = data.cycle_loader(eval_ds)
    
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    
    ema = ExponentialMovingAverage(score_model, decay=cfg.training.ema, update_after_step=2000)
    state = dict(optimizer=optimizer, scheduler=scheduler, model=score_model, noise=noise, ema=ema, step=0) 


    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    mprint(score_model)
    mprint(f"EMA: {ema}")
    initial_step = int(state['step'])

    wandb_run_id = None  # Default to None for new runs

    if "wandb_run_id" in state and state['wandb_run_id']:
        mprint(f'continue wandb with id:{state["wandb_run_id"]}')
        wandb_run_id = state["wandb_run_id"]
        resume = "allow"  # Ensures that WandB resumes the run
    else:
        mprint(f'start new wandb')
        resume = False  
        
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.experiment.wandb_project,
            init_kwargs={
                "wandb": {
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'id': wandb_run_id,
                    'resume': resume
                }
            }
        )

    if 'wandb_run_id' in state and state['wandb_run_id']:
        wandb_run_id = state['wandb_run_id']
        mprint(f"Resuming training from step {initial_step} with WandB run ID: {wandb_run_id}")
    else:
        mprint("Starting training from scratch.")

    # Build one-step training and evaluation functions
    train_step_fn = losses.get_step_fn(cfg, accelerator, noise, graph, True, metaschedule=meta_schedule)
    eval_step_fn = losses.get_step_fn(cfg, accelerator, noise, graph, False, metaschedule=eval_meta_schedule)

    
    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.gradient_accumulation_steps), cfg.model.length)
        if cfg.annealing.sampling_method == "NAR":
            sampling_fn = sampling.get_sa_sampling_fn(cfg, graph, noise, eval_meta_schedule, sampling_shape, sampling_eps, device)
        elif cfg.annealing.sampling_method == "SAR":
            sampling_fn = sampling_sar.get_sa_sampling_fn(cfg, graph, noise, eval_meta_schedule, sampling_shape, sampling_eps, device)
        else:
            raise ValueError(f"Sampling method should be either NAR(non-autoregressive) or SAR(semi-autoregressive).")

    num_train_steps = cfg.training.n_iters
    list_loss_ce = []
    list_loss_diff = []
    while state['step'] < num_train_steps + 1:
        step = state['step']       
        batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss_diff, loss_ce = train_step_fn(state, batch)
        list_loss_ce.append(loss_ce.item())
        list_loss_diff.append(loss_diff.item())
        
        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                loss_diff = accelerator.gather(loss_diff).mean()
                loss_ce = accelerator.gather(loss_ce).mean()
                mprint("step: %d, training_diffusion_loss: %.5e" % (step, loss_diff.item()))
                mprint("step: %d, training_ce_loss: %.5e" % (step, loss_ce.item()))
                if accelerator.is_main_process:
                    accelerator.log({"loss diffusion": loss_diff}, step=step)
                    accelerator.log({"loss ce": loss_ce}, step=step)
                if step % 200 == 0:
                    mprint(f"Global statistics: Average CE loss: {np.mean(list_loss_ce)}, Std CE loss: {np.std(list_loss_ce)}")
                    mprint(f"Global statistics: Average Diffusion loss: {np.mean(list_loss_diff)}, Std Diffusion loss: {np.std(list_loss_diff)}")
                    list_loss_ce = []
                    list_loss_diff = []
            if step % cfg.training.snapshot_freq_for_preemption == 0 and accelerator.is_main_process:
                wandb_tracker = accelerator.get_tracker("wandb")
                state['wandb_run_id'] = wandb_tracker.run.id
                utils.save_checkpoint(checkpoint_meta_dir, state, accelerator)
                accelerator.save_state(safe_tensor)
            
            if step % cfg.training.eval_freq == 0:
                try:
                    eval_data = next(eval_iter)
                except StopIteration:
                    # Reinitialize the iterator when it's exhausted
                    eval_iter = iter(eval_ds)
                    eval_data = next(eval_iter)

                eval_batch = eval_data

                eval_loss_diff, eval_loss_ce = eval_step_fn(state, eval_batch)
                eval_loss_diff = accelerator.gather(eval_loss_diff).mean()
                eval_loss_ce = accelerator.gather(eval_loss_ce).mean()
                
                mprint("step: %d, evaluation__diffusion_loss: %.5e" % (step, eval_loss_diff.item()))
                mprint("step: %d, evaluation_ce_loss: %.5e" % (step, eval_loss_ce.item()))
                if accelerator.is_main_process:
                    wandb.log({"eval_loss": eval_loss_diff.item()}, step=step)
                    wandb.log({"eval_loss_ce": eval_loss_ce.item()}, step=step)
                
                # Calculate zero-shot perplexity periodically
                if step % (cfg.training.eval_freq * 1) == 0:  # Every 5 evaluation cycles
                    if accelerator.is_main_process:
                        try:
                            mprint("Calculating zero-shot perplexity...")
                            ppl, nll_type, eval_type = calculate_perplexity(
                                cfg, score_model, device, mc_num=100, ema=ema
                            )
                            mprint(f"Zero-shot perplexity at step {step}: {ppl:.4f} (using {nll_type} method, {eval_type} mode)")
                            wandb.log({"zeroshot_perplexity": ppl}, step=step)
                        except Exception as e:
                            mprint(f"Error calculating zero-shot perplexity: {e}")
                    
                    accelerator.wait_for_everyone()
            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                save_step = step // cfg.training.snapshot_freq
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    state['wandb_run_id'] = wandb_tracker.run.id
                    if step % 20_000 == 0:
                        utils.save_checkpoint(os.path.join(
                            checkpoint_dir, f'checkpoint_{save_step}.pth'), state, accelerator)
                if cfg.training.snapshot_sampling:
                    if step < cfg.training.warmup_iter:
                        continue
                    
                    sample = sampling_fn(score_model)
                    gathered_samples = accelerator.gather(sample)

                    sentences = tokenizer.batch_decode(gathered_samples)
                    if accelerator.is_main_process:
                        mprint(f"Generating text at step: {step}")

                        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                        utils.makedirs(this_sample_dir)
                        file_name = os.path.join(this_sample_dir, f"sample.txt")
                        with open(file_name, 'w') as file:
                            for sentence in sentences:
                                file.write(sentence + "\n")
                                file.write("============================================================================================\n")
                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            try:
                                entropy = utils.compute_entropy(sentences, tokenizer, device=device).mean()
                                entropy = accelerator.gather(entropy)
                                entropy = entropy.mean().item()
                                print(f"Entropy at step: {step}. Entropy: {entropy:.3f}.")
                            except Exception:
                                entropy = 0

                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = gathered_samples.shape[0] // cfg.eval.perplexity_batch_size
                            if batches == 0:
                                mprint(f"Warning: Sample size ({gathered_samples.shape[0]}) is smaller than perplexity batch size ({cfg.eval.perplexity_batch_size}). Skipping perplexity calculation.")
                                total_perplexity = torch.tensor(0.0, device=device)
                            else:
                                total_perplexity = 0
                                for i in range(batches):
                                    s = gathered_samples[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                    loss, logits = eval_model(s, labels=s)[:2]
                                    logits = logits.transpose(-1, -2)
                                    perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                    total_perplexity += perplexity
                                total_perplexity /= batches
                            
                            total_perplexity = accelerator.reduce(total_perplexity, reduction="sum")
                            total_perplexity /= accelerator.num_processes

                            if accelerator.is_main_process:
                                mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")
                                wandb.log({"perplexity": total_perplexity}, step=step)
                                wandb.log({"entropy": entropy}, step=step)
                            del eval_model, logits, loss 