import torch
import numpy as np
from hdlm.model import utils as model_utils
import hdlm.metaschedule_utils as msutils
from hdlm.metaschedule import get_block_annealing_efficient, get_simple_annealing_efficient
import torch.nn.functional as F


def get_loss_fn(noise, graph, train, annealing, sampling_eps=1e-6, metaschedule=None):
    
    def loss_fn(model, batch, perturbed_batch=None):
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        act_seq_len = seq_len // 2

        if metaschedule is None:
            raise ValueError('Metaschedule must be set when using annealing.')
        if annealing.efficient:
            if metaschedule.type.value == 'block_annealing':
                ms_state, _ = get_block_annealing_efficient(metaschedule, act_seq_len, sampling=True)
                t = msutils.compute_t_and_attn_efficient_training(metaschedule, ms_state, device, sampling_eps)
                attn_mask = msutils.generate_attention_mask_efficient_training(batch_size, act_seq_len, act_seq_len // metaschedule.width, context_attention_type=annealing.attention.context_type, block_attention_type=annealing.attention.block_type)
            elif metaschedule.type.value == 'simple_annealing':
                ms_state, ms_steps = get_simple_annealing_efficient(metaschedule, act_seq_len)
                t = msutils.compute_t_and_attn_efficient_training(metaschedule, ms_state, device, sampling_eps)
                attn_mask = msutils.generate_attention_mask_simple_efficient_training(
                    bs=batch_size,
                    seq_len=act_seq_len,
                    steps=ms_steps,
                    context_attention_type=annealing.attention.context_type,
                    block_attention_type=annealing.attention.block_type,   # or "full"
                )
            else:
                raise ValueError(f'Metaschedule shoule be either block_annealing or simple_annealing.')
            active_mask = torch.zeros(seq_len, dtype=torch.bool)
            settled_mask = torch.zeros(seq_len, dtype=torch.bool)
            weight_mask = torch.ones(seq_len, device=device)
            settled_mask[:act_seq_len] = True
            active_mask[act_seq_len:] = True
            weight_mask[:act_seq_len] = 0.0
        else:
            # Sample valid current_step (similar to inference)
            current_step = np.random.randint(1, metaschedule.max_step(seq_len))

            t, attn_mask, _, active_mask, settled_mask, weight_mask = msutils.compute_t_and_attn(
                batch_size, seq_len, metaschedule, current_step,
                context_attention_type=annealing.attention.context_type, block_attention_type=annealing.attention.block_type,
                device=device, eps=sampling_eps)

        if not annealing.match_inference:
            active_eps = 10 * sampling_eps
            step_size = (1 - 2 * active_eps) / (metaschedule.worthless - 1)
            t[active_mask] += step_size * torch.rand_like(active_mask, device=device)
            t[active_mask] = torch.clamp(t, min=active_eps, max=1 - active_eps)
        t = t.unsqueeze(0).expand(batch_size, seq_len)

        sigma, dsigma = noise(t)
        
        perturbed_batch = graph.sample_transition(input_ids, sigma)    
        settled_positions_mask = settled_mask.unsqueeze(0).expand(batch_size, seq_len)  # [batch_size, seq_len]
        
        log_score_fn = model_utils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma, positions=positions, attn_mask=attn_mask, settled_tokens=settled_positions_mask if not annealing.efficient else None)
        diffusion_log_score = log_score.clone()
        # diffusion_log_score = diffusion_log_score - diffusion_log_score.gather(dim=-1, index=input_ids.unsqueeze(-1))
        masked_dsigma = dsigma[:, ~settled_mask]
        masked_weight = weight_mask[~settled_mask]  # shape [sum_of_true_in_mask]
        loss = graph.score_entropy(
            diffusion_log_score,
            sigma,
            perturbed_batch,
            input_ids
        )
        
        diff_loss = (masked_dsigma * loss[:, ~settled_mask] * masked_weight).sum(dim=-1)
        if not annealing.efficient:
            num_settled = metaschedule(current_step).num_settled
            num_settled = min(seq_len, num_settled)
            block_weighting = model_utils.get_block_weighting(num_settled, metaschedule.width, device)
            if num_settled == 0:
                return diff_loss, torch.zeros_like(diff_loss)
        else:
            block_weighting = torch.ones(act_seq_len, device=device)
        
        ignore_index = 50257
        # Reshape logits and input_ids for compatibility with CrossEntropyLoss
        batch_size, seq_len, vocab_dim = log_score.size()
        ce_logits = log_score[:, settled_mask].reshape(-1, vocab_dim)  # Shape: [batch_size * seq_len, vocab_dim]
        ce_input_ids = input_ids[:, settled_mask].reshape(-1)      # Shape: [batch_size * seq_len]

        # Calculate cross-entropy loss
        per_token_loss = F.cross_entropy(ce_logits, ce_input_ids, reduction='none', ignore_index=ignore_index)  # Scalar loss
        # Reshape per-token loss to [batch_size, seq_len]
        per_token_loss = per_token_loss.view(batch_size, -1) * block_weighting
        valid_tokens_mask = (ce_input_ids.view(batch_size, -1) != ignore_index).float()
        ce_loss = (per_token_loss * valid_tokens_mask).sum(dim=-1) / (valid_tokens_mask.sum(dim=-1) + 1e-8)

        return diff_loss, ce_loss
   
    return loss_fn


def get_step_fn(config, accelerator, noise, graph, train, metaschedule=None):

    loss_fn = get_loss_fn(
        noise, graph, train,
        annealing=config.annealing,
        metaschedule=metaschedule, 
        sampling_eps=config.annealing.sampling_eps
    )
    def step_fn(state, batch):
        model = state['model']
        if train:
            optimizer = state['optimizer']
            scheduler = state['scheduler']
            with accelerator.accumulate(model):
                # Zero gradients
                optimizer.zero_grad()

                loss_diff, loss_ce = loss_fn(model, batch)
                loss_diff = loss_diff.mean()
                loss_ce = loss_ce.mean()
                loss = config.annealing.diffusion_loss_weight * loss_diff + config.annealing.ce_loss_weight * loss_ce
                
                accelerator.backward(loss)
                state['step'] += 1
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"Parameter {name} did not receive a gradient.")
                optimizer.step()
                
                scheduler.step()
                # Update EMA
                state['ema'].step(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss_diff, loss_ce = loss_fn(model, batch)
                loss_diff = loss_diff.mean()
                loss_ce = loss_ce.mean()
                ema.restore(model.parameters())

        return loss_diff.detach(), loss_ce.detach()

    return step_fn
