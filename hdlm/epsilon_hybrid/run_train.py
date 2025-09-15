import os
import numpy as np
import torch

import hdlm.utils as utils
import hdlm.data as data
from hdlm.model.diffusion_lm import HDLM
from diffusers.training_utils import EMAModel as ExponentialMovingAverage

import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import GPT2TokenizerFast
import hdlm.epsilon_hybrid.losses as losses
import hdlm.epsilon_hybrid.sample as sample_utils
import hdlm.calculate_metrics as metrics
from hdlm.metaschedule import make_simple_annealing, make_block_annealing, make_hybrid_annealing


def get_sampling_functions(cfg, eval_meta_schedule):
    """
    Get appropriate sampling functions based on configuration type and sampling method.
    
    Args:
        cfg: Configuration object
        eval_meta_schedule: Evaluation meta schedule
        
    Returns:
        tuple: (sampling_fn, sampling_algorithm)
    """
    training_type = getattr(cfg, 'type', 'aligned')
    sampling_method = getattr(cfg.annealing, 'sampling_method', 'NAR')
    
    if sampling_method == "SAR" and eval_meta_schedule is not None:
        if training_type == 'shifted':
            return sample_utils.semi_diff_shifted, 'original'
        else:
            return sample_utils.semi_diff, 'original'
    else:
        if training_type == 'shifted':
            return sample_utils.full_diff_shifted, 'original'
        else:
            return sample_utils.full_diff, 'original'


def calculate_perplexity(cfg, score_model, eval_ds, eval_meta_schedule, device, mc_num=50):
    """
    Calculate zero-shot perplexity based on configuration type.
    
    Args:
        cfg: Configuration object
        score_model: The model
        eval_ds: Evaluation dataset
        eval_meta_schedule: Evaluation meta schedule
        device: Device to run on
        mc_num: Monte Carlo number for sampling
        
    Returns:
        tuple: (perplexity, nll_type, eval_type_name)
    """
    from hdlm.zeroshot_ppl import ZeroShot_calculator
    
    nll_type = 'mc'
    training_type = getattr(cfg, 'type', 'aligned')
    
    zeroshot_calculator = ZeroShot_calculator(
        model=score_model, 
        device=device,
        sampling_eps=cfg.annealing.sampling_eps,
        metaschedule=eval_meta_schedule,
        annealing=cfg.annealing if hasattr(cfg, 'annealing') else None,
        mc_num=mc_num,
        mode='aligned' if training_type == 'aligned' else 'shifted'
    )
    
    ppl = zeroshot_calculator.evaluate_perplexity_from_dataloader(eval_ds, nll_type=nll_type)
    
    return ppl, nll_type, training_type


def run_train(cfg):
    """
    Main training function supporting both aligned and shifted modes.
    """
    work_dir = cfg.work_dir
    training_type = getattr(cfg, 'type', 'aligned')

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

    mprint(f"Starting {training_type} training in: {work_dir}")
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

    score_model = HDLM(cfg)
    state = dict(model=score_model, step=0)

    if hasattr(cfg, 'hf_model_id') and cfg.hf_model_id:
        mprint(f"Loading model from Hugging Face: {cfg.hf_model_id}")
        try:
            state = utils.smart_restore_checkpoint(
                cfg.hf_model_id, 
                state, 
                accelerator, 
                model_type="epsilon_hybrid"
            )
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

    optimizer = utils.get_optimizer(cfg, score_model.parameters())
    mprint(f"Optimizer: {optimizer}")
    scheduler = utils.get_scheduler(cfg, optimizer, eta_min=utils.SCHEDULER_ETA_MIN_EPSILON)
    mprint(f"Scheduler: {scheduler}")    

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
        meta_schedule = None
        eval_meta_schedule = None
    
    score_model, optimizer, scheduler, train_ds, eval_ds = accelerator.prepare(
        score_model, optimizer, scheduler, train_ds, eval_ds
    )

    train_ds = data.cycle_loader(train_ds)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    
    ema = ExponentialMovingAverage(score_model, decay=cfg.training.ema, update_after_step=2000)
    state = dict(optimizer=optimizer, scheduler=scheduler, model=score_model, ema=ema, step=0) 

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")
    mprint(f"Training mode: {training_type}")
    mprint(score_model)
    mprint(f"EMA: {ema}")
    mprint("Restoring checkpoint...")

    sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.gradient_accumulation_steps), cfg.model.length)
    sampling_fn, sampling_alg = get_sampling_functions(cfg, eval_meta_schedule)
    
    mc_num = 50
    initial_step = int(state['step'])

    wandb_run_id = None

    if "wandb_run_id" in state and state['wandb_run_id']:
        mprint(f'Continuing WandB with ID: {state["wandb_run_id"]}')
        wandb_run_id = state["wandb_run_id"]
        resume = "allow"
    else:
        mprint('Starting new WandB run')
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

    train_step_fn = losses.get_step_fn(cfg, accelerator, True, metaschedule=meta_schedule, annealing=cfg.annealing, type=training_type)
    eval_step_fn = losses.get_step_fn(cfg, accelerator, False, metaschedule=eval_meta_schedule, annealing=cfg.annealing, type=training_type)
        
    num_train_steps = cfg.training.n_iters
    list_loss = []
    
    mprint(f"Starting {training_type} training loop...")
    
    while state['step'] < num_train_steps + 1:
        step = state['step']       
        batch = next(train_iter)
        loss = train_step_fn(state, batch)
        
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                loss = accelerator.gather(loss).mean()
                list_loss.append(loss.item())
                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
                if accelerator.is_main_process:
                    accelerator.log({"loss": loss}, step=step)
                if step % 200 == 0:
                    mprint(f"Global statistics: Average loss: {np.mean(list_loss)}, Std loss: {np.std(list_loss)}")
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and accelerator.is_main_process:
                wandb_tracker = accelerator.get_tracker("wandb")
                state['wandb_run_id'] = wandb_tracker.run.id
                utils.save_checkpoint(checkpoint_meta_dir, state, accelerator)
                accelerator.save_state(safe_tensor)
            
            if step % cfg.training.eval_freq == 0:
                try:
                    eval_data = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_ds)
                    eval_data = next(eval_iter)
                
                eval_batch = eval_data
                eval_loss = eval_step_fn(state, eval_batch)
                eval_loss = accelerator.gather(eval_loss).mean()
                
                ppl, nll_type, eval_type_name = calculate_perplexity(
                    cfg, score_model, eval_ds, eval_meta_schedule, device, mc_num
                )
                
                if eval_meta_schedule is not None and hasattr(cfg, 'annealing'):
                    mprint(f"step: {step}, zeroshot_{nll_type}_ppl_{eval_type_name}_annealing (on {cfg.data.valid}): {ppl:.5e}")
                    wandb_metric_name = f"zeroshot_{nll_type}_ppl_{eval_type_name}_annealing (on {cfg.data.valid})"
                else:
                    mprint(f"step: {step}, zeroshot_{nll_type}_ppl (on {cfg.data.valid}): {ppl:.5e}")
                    wandb_metric_name = f"zeroshot_{nll_type}_ppl (on {cfg.data.valid})"
                
                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))
                
                if accelerator.is_main_process:
                    wandb.log({"eval_loss": eval_loss.item()}, step=step)
                    wandb.log({wandb_metric_name: ppl}, step=step)
            
            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    state['wandb_run_id'] = wandb_tracker.run.id
                    if step % cfg.training.snapshot_freq == 0:
                        utils.save_checkpoint(os.path.join(
                            checkpoint_dir, f'checkpoint_{step}.pth'), state, accelerator)
                
                if cfg.training.snapshot_sampling and cfg.training.warmup_iter < step:
                    sample = sampling_fn(
                        score_model,
                        batch_size=sampling_shape[0], 
                        context_length=sampling_shape[1], 
                        alg=sampling_alg
                    )
                    
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
                                file.write("=" * 80 + "\n")
                    
                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            mprint(f"Calculating generation metrics at step: {step}...")
                            
                            entropy_mean_tensor = 0.0
                            gpt2_perplexity_tensor = 0.0
                            mauve_score_tensor = 0.0
                            
                            try:
                                entropy_sentences = sentences[:min(len(sentences), 32)]
                                
                                entropy = utils.compute_entropy(entropy_sentences, tokenizer, device=device).mean()
                                entropy = accelerator.gather(entropy)
                                entropy_mean_tensor = entropy.mean().item()
                                mprint(f"Entropy at step: {step}. Entropy: {entropy_mean_tensor:.3f}.")
                                
                                del entropy, entropy_sentences
                                torch.cuda.empty_cache()
                                
                            except Exception:
                                entropy_mean_tensor = 0.0

                            if accelerator.is_main_process:
                                import gc
                                
                                if getattr(cfg.eval, 'calculate_gpt2_perplexity', True):
                                    try:
                                        reduced_batch_size = min(cfg.eval.perplexity_batch_size, 4)
                                        
                                        sample_subset = sample[:min(sample.shape[0], 16)]
                                        
                                        gpt2_perplexity_tensor = metrics.calculate_gpt2_perplexity(
                                            sample_subset, 
                                            device=device,
                                            batch_size=reduced_batch_size
                                        )
                                        
                                        gpt2_perplexity_tensor = gpt2_perplexity_tensor.item() if hasattr(gpt2_perplexity_tensor, 'item') else gpt2_perplexity_tensor
                                        
                                        mprint(f"GPT-2 Perplexity at step: {step}. Perplexity: {gpt2_perplexity_tensor:.4f}")
                                        
                                        del sample_subset
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                        
                                    except Exception as e:
                                        mprint(f"Error calculating GPT-2 perplexity: {e}")
                                        gpt2_perplexity_tensor = 0.0
                                        torch.cuda.empty_cache()
                                        gc.collect()

                                if getattr(cfg.eval, 'calculate_mauve', True):
                                    try:
                                        generated_subset = sentences[:min(len(sentences), 16)]  # Limit to 16 samples
                                        
                                        human_samples = metrics.extract_human_samples(eval_ds, tokenizer, num_samples=len(generated_subset))
                                        
                                        mauve_score_tensor = metrics.calculate_mauve_score(
                                            generated_subset, 
                                            human_samples
                                        )
                                        
                                        mprint(f"MAUVE Score at step: {step}. Score: {mauve_score_tensor:.4f}")
                                        
                                        del generated_subset, human_samples
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                        
                                    except ImportError:
                                        mprint("MAUVE library not installed. Install with: pip install mauve-text")
                                        mauve_score_tensor = 0.0
                                    except Exception as e:
                                        mprint(f"Error calculating MAUVE score: {e}")
                                        mauve_score_tensor = 0.0
                                        torch.cuda.empty_cache()
                                        gc.collect()
                            
                            if accelerator.num_processes > 1:
                                gpt2_perplexity_tensor = torch.tensor(gpt2_perplexity_tensor, device=device)
                                mauve_score_tensor = torch.tensor(mauve_score_tensor, device=device)
                                
                                torch.distributed.broadcast(gpt2_perplexity_tensor, src=0)
                                torch.distributed.broadcast(mauve_score_tensor, src=0)
                                
                                gpt2_perplexity_tensor = gpt2_perplexity_tensor.item()
                                mauve_score_tensor = mauve_score_tensor.item()

                            if accelerator.is_main_process:
                                log_dict = {
                                    f"generation_metrics_{training_type}/text_entropy_mean": entropy_mean_tensor,
                                    f"generation_metrics_{training_type}/gpt2_perplexity": gpt2_perplexity_tensor,
                                }
                                
                                if mauve_score_tensor > 0:
                                    log_dict[f"generation_metrics_{training_type}/mauve_score"] = mauve_score_tensor
                                
                                wandb.log(log_dict, step=step)
                                
                            mprint(f"Generation metrics calculation completed at step: {step}")
                            
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            import gc
                            gc.collect()

    mprint(f"Training completed! Final step: {state['step']}")