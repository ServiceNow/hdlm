import torch
import torch.optim as optim
import numpy as np
from hdlm.model import utils as model_utils
import hdlm.metaschedule_utils as msutils
from hdlm.metaschedule import get_block_annealing_efficient, get_simple_annealing_efficient
import torch.nn.functional as F
import math


def get_DCE_loss_fn(noise, graph, train, sampling_eps=1e-4, metaschedule=None, steps_per_level=0, efficient=False, graph_type=None):
        
    def ar_loss_fn(model, batch, perturbed_batch=None):
        """
        Autoregressive Loss Function with DCE and Cross-Entropy Losses.

        Args:
            model: The autoregressive language model to be trained.
            batch: Dictionary containing:
                - 'input_ids': Tensor of shape [batch_size, seq_len] with token IDs.
                - 'positions': Tensor of shape [batch_size, seq_len] with position IDs.
            perturbed_batch: Optional tensor with perturbed input IDs. If None, it will be sampled.

        Returns:
            Tuple of two tensors:
                - final_dce_loss: Tensor of shape [batch_size] representing the DCE loss.
                - ce_loss: Tensor of shape [batch_size] representing the Cross-Entropy loss.
        """
        if graph_type != 'absorb':
            raise ValueError('DCE loss is only supported for absorb graph type.')
        
        token_dim = 50_258  # Hard-coded for absorb
        input_ids = batch['input_ids']  # Shape: [batch_size, seq_len]
        positions = batch['positions']  # Shape: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if metaschedule is None:
            raise ValueError('Metaschedule must be set when using annealing.')
        if efficient:
            raise NotImplementedError('Efficient mode is not supported for autoregressive loss for now...')
        else:
            current_step = np.random.randint(1, metaschedule.max_step(seq_len))
            t_start, t_end, attn_mask, _, _, _, weight_mask = msutils.compute_t_and_attn_autoregressive(
                batch_size, seq_len - 1, metaschedule, current_step, device, sampling_eps
            )
        
        if steps_per_level > 0:
            delta_t = (t_start - t_end) / steps_per_level  # Shape: [seq_len]

            inner_step = torch.randint(
                low=0,
                high=steps_per_level,
                size=(1,),
                device=device
            )

            t = t_start - delta_t * inner_step  # Shape: [seq_len]
        else:
            t = t_end + (t_start - t_end) * torch.rand(seq_len, device=device)

        t = t.unsqueeze(0).expand(batch_size, seq_len)  # Shape: [batch_size, seq_len]

        sigma, dsigma = noise(t)  # Assume noise(t) returns two tensors based on t
        # sigma and dsigma shapes: [batch_size, seq_len]

        num_settled = metaschedule(current_step).num_settled
        num_settled = min(seq_len - 1, num_settled)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(input_ids, sigma)  # Shape: [batch_size, seq_len]

        masked_index = perturbed_batch[:, 1:] == (token_dim - 1)  # Assuming (token_dim - 1) is the mask token, dropping the first token for AR (in input)
        # ----------------------------
        # DCE Loss: Next Token Prediction for Masked Tokens
        # ----------------------------

        # Step 1: Create shifted targets for next token prediction
        shifted_input_ids = input_ids[:, 1:]  # Drop the first token for AR
        shifted_batch = shifted_input_ids[masked_index]  # Targets for masked

        # Step 2: Compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        logits = model(perturbed_batch, positions=positions, attn_mask=attn_mask)  # Shape: [batch_size, seq_len, token_dim]
        
        # Compute log probabilities
        logprobs = logits[:, :-1, :].clone() # exclude the last token
        logprobs[:, :, :-1] = logits[:, :-1, :-1].log_softmax(dim=-1)  # Apply log_softmax excluding the mask token, drop the last as it is redundant in autoregressive generation
        # logprobs = logits.log_softmax(dim=-1)  # Apply log_softmax over the token_dim
        # If the last token requires log_softmax, adjust accordingly

        # Compute esigm1 based on sigma
        esigm1 = torch.where(
            sigma[:, 1:] < 0.5,
            torch.expm1(sigma[:, 1:]),
            torch.exp(sigma[:, 1:]) - 1
        )  # Shape: [batch_size, seq_len]

        # Adjust log probabilities for DCE loss
        logprobs_dce = logprobs - esigm1.log()[..., None]  # Broadcasting over token_dim

        # Gather log probabilities of the target next tokens for DCE loss
        loss_dce = torch.zeros([input_ids.shape[0], input_ids.shape[1] - 1], device=input_ids.device, dtype=logprobs_dce.dtype)
        loss_dce[masked_index] = -torch.gather(logprobs_dce[masked_index], dim=-1, index=shifted_batch[..., None]).squeeze(-1)  # Shape: [num_masked]
        loss_dce = loss_dce / esigm1  # Shape: [num_masked]
        # Weight the loss by dsigma at masked positions and sum over all masked tokens
        final_dce_loss = (dsigma[:, 1:] * loss_dce * weight_mask[1:]).sum(dim=-1)  # Scalar
        if num_settled <= 1:
            # Create a dummy loss of zero with the same shape as final_dce_loss
            ce_loss = torch.zeros_like(final_dce_loss)
        else:
            ce_loss = - logprobs[:, :num_settled].gather(-1, shifted_input_ids[:, :num_settled, None])[:, :, 0]
        return final_dce_loss, ce_loss
    return ar_loss_fn

def get_loss_fn(noise, graph, train, annealing, sampling_eps=1e-6, metaschedule=None):
    def causal_loss_fn(model, batch, perturbed_batch=None):
        """
        Mock a loss function for training. 
        Returns dummy losses involving the model parameters.
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        ignore_index = 50257
        model.train()
        t = torch.ones(batch_size, seq_len, device=device) * sampling_eps

        sigma, dsigma = noise(t)

        attn_mask = msutils.create_causal_attention(batch_size, seq_len, seq_len, device)
       
        output = model(input_ids, sigma, positions=positions, attn_mask=attn_mask)
        # log_score = model(input_ids, positions=positions, attn_mask=attn_mask)
        logits = output[:, :-1].reshape(-1, ignore_index + 1)
        ce_input_ids = input_ids[:, 1:].reshape(-1)
        per_token_loss = F.cross_entropy(logits, ce_input_ids, reduction='none', ignore_index=ignore_index)  # Scalar loss
        ce_loss = per_token_loss.view(batch_size, seq_len - 1)
        return torch.zeros_like(ce_loss), ce_loss
        
    def ar_from_scratch_loss(model, batch, perturbed_batch=None):
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        act_seq_len = seq_len // 2
    
        if metaschedule is None:
            raise ValueError('Metaschedule must be set when using annealing.')
        if annealing.efficient:
            raise NotImplementedError('Efficient mode is not supported for autoregressive loss for now..., need to fix attention mask')
            if metaschedule.type.value == 'block_annealing':
                ms_state, _ = get_block_annealing_efficient(metaschedule, act_seq_len, sampling=False)
                t_start, t_end = msutils.compute_t_and_attn_efficient_training(metaschedule, ms_state, device, sampling_eps)
                attn_mask = msutils.generate_attention_mask_block_efficient_training_autoregressive(batch_size, act_seq_len, act_seq_len // metaschedule.width, context_attention_type=annealing.attention.context_type, block_attention_type=annealing.attention.block_type)
            elif metaschedule.type.value == 'simple_annealing':
                raise ValueError('Efficient mode on simple_annealing is not supported for autoregressive loss for now...')
                ms_state, ms_steps = get_simple_annealing_efficient(metaschedule, act_seq_len)
                t_start, t_end = msutils.compute_t_and_attn_efficient_training(metaschedule, ms_state, device, sampling_eps)
                attn_mask = msutils.generate_attention_mask_simple_efficient_training(batch_size, act_seq_len, ms_steps)
            else:
                raise ValueError(f'Metaschedule shoule be either block_annealing or simple_annealing.')
            active_mask = torch.zeros(seq_len, dtype=torch.bool)
            settled_mask = torch.zeros(seq_len, dtype=torch.bool)
            weight_mask = torch.ones(seq_len, device=device)
            settled_mask[:act_seq_len] = True
            active_mask[act_seq_len:] = True
            weight_mask[:act_seq_len] = 0.0
            
        else:
            current_step = np.random.randint(1, metaschedule.max_step(seq_len))
            t, attn_mask, _, active_mask, settled_mask, weight_mask = msutils.compute_t_and_attn_autoregressive(
                batch_size, seq_len, metaschedule, current_step, context_attention_type=annealing.attention.context_type, block_attention_type=annealing.attention.block_type, device=device, eps=sampling_eps)
        
        
        if not annealing.match_inference:
            active_eps = 10 * sampling_eps
            step_size = (1 - 2 * active_eps) / (metaschedule.worthless - 1)
            t[active_mask] += step_size * torch.rand_like(active_mask, device=device)
            t[active_mask] = torch.clamp(t, min=active_eps, max=1 - active_eps)
        t = t.unsqueeze(0).expand(batch_size, seq_len)

        sigma, dsigma = noise(t)
        
        perturbed_batch = graph.sample_transition(input_ids, sigma)    
        settled_positions_mask = settled_mask.unsqueeze(0).expand(batch_size, seq_len)  # [batch_size, seq_len]
        # Proceed with the rest of the loss computation
        log_score_fn = model_utils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma, positions=positions, attn_mask=attn_mask, settled_tokens=settled_positions_mask if not annealing.efficient else None)
        
        if not annealing.efficient:
            num_settled = metaschedule(current_step).num_settled
            num_settled = min(seq_len - 1, num_settled)
            num_settled = max(1, num_settled)
            block_weighting = model_utils.get_block_weighting(num_settled, metaschedule.width, device)
        else:
            num_settled = act_seq_len

        output = log_score[:, num_settled:-1]
        loss = graph.score_entropy(output, sigma[:, num_settled+1:], perturbed_batch[:, num_settled+1:], input_ids[:, num_settled+1:])
        diff_loss = (dsigma[:, 1+num_settled:] * loss * weight_mask[1 + num_settled:]).sum(dim=-1)
            
        if num_settled <= 1:
            # create a dummy loss of zero with the same shape as diff_loss
            ce_loss = torch.zeros_like(loss)
        else:
            if num_settled == seq_len:
                num_settled -= 1
            
            batch_size, seq_len, vocab_dim = log_score.size()
            ignore_index = 50257
            if annealing.efficient:
                num_settled -= 1
                block_weighting = torch.ones(act_seq_len, device=device)
            ce_logits = log_score[:, :num_settled].reshape(-1, vocab_dim)  # Shape: [batch_size * seq_len, vocab_dim]
            ce_input_ids = input_ids[:, 1:num_settled+1].reshape(-1)      # Shape: [batch_size * seq_len]
            per_token_loss = F.cross_entropy(ce_logits, ce_input_ids, reduction='none', ignore_index=ignore_index)  # Scalar loss
            per_token_loss = per_token_loss.view(batch_size, num_settled) * block_weighting
            valid_tokens_mask = (ce_input_ids.view(batch_size, num_settled) != ignore_index).float()
            ce_loss = ((per_token_loss * valid_tokens_mask).sum(dim=-1) / (valid_tokens_mask.sum(dim=-1) + 1e-8))

        return diff_loss, ce_loss

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
   
    if annealing.sampling_method in {'AR', 'AR_SCRATCH', 'AR_BAD'}:
        return ar_from_scratch_loss
    elif annealing.sampling_method in {'Causal', 'Causal_SCRATCH'}:
        return causal_loss_fn
    elif annealing.sampling_method in {'Doublehead_AR', 'Doublehead_AR_SCRATCH'}:
        return ar_doublehead_loss
    else:
        return loss_fn


def get_ppl_function(noise, graph, train, annealing, sampling_eps=1e-6, metaschedule=None, method='AR'):
    def sar_ppl_fn(model, batch, perturbed_batch=None):
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
        
        ignore_index = 50257
        
        
        log_probs = F.log_softmax(log_score, dim=-1)
        bs, seq_len, vocab_dim = log_probs.shape
        
        # If active_mask is provided as [seq_len], expand it to [bs, seq_len]
        if active_mask.dim() == 1:
            active_mask = active_mask.unsqueeze(0).expand(bs, seq_len)
        
        active_mask_per_sample = active_mask.view(bs, seq_len)

        target_flat = input_ids[active_mask_per_sample].view(-1)
        log_probs_flat = log_probs[active_mask_per_sample].view(-1, vocab_dim)
        
        
        loss_per_token = F.nll_loss(log_probs_flat, target_flat, reduction='none', ignore_index=ignore_index)  # shape: [bs*seq_len]
        return loss_per_token.view(bs, -1)
    

    def sar_ppl_fn2(model, batch, perturbed_batch=None):
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

        bs, seq_len, vocab_dim = log_score.shape
        if annealing.efficient:
            loss = graph.score_entropy(log_score[:, 1024:, :], sigma[:, 1024:], perturbed_batch[:, 1024:], input_ids[:, 1024:])
            return dsigma[:, 1024:] * loss
        
        # If active_mask is provided as [seq_len], expand it to [bs, seq_len]
        if active_mask.dim() == 1:
            active_mask = active_mask.unsqueeze(0).expand(bs, seq_len)
        
        active_mask_per_sample = active_mask.view(bs, seq_len).to(device)        

        loss = graph.score_entropy(log_score, sigma, perturbed_batch, input_ids) 
        loss = (dsigma * loss * active_mask_per_sample * weight_mask) # Shape: [batch_size, seq_len - 1] or O([bs, seq_len])
        active_counts = active_mask_per_sample.sum(dim=1).float() 
        return loss
        reweight_factors = seq_len / active_counts  # shape: [bs]
        return (loss * reweight_factors.unsqueeze(1))
    
    def sar_ppl_fn3(model, batch, perturbed_batch=None):
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        act_seq_len = seq_len // 2
        annealing.efficient = False

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
        num_settled = metaschedule(current_step).num_settled
        log_score = log_score_fn(perturbed_batch, sigma, positions=positions, attn_mask=attn_mask, settled_tokens=settled_positions_mask if not annealing.efficient else None)

        bs, seq_len, vocab_dim = log_score.shape
        if annealing.efficient:
            loss = graph.score_entropy(log_score[:, 1024:, :], sigma[:, 1024:], perturbed_batch[:, 1024:], input_ids[:, 1024:])
            return dsigma[:, 1024:] * loss
        
        # If active_mask is provided as [seq_len], expand it to [bs, seq_len]
        if active_mask.dim() == 1:
            active_mask = active_mask.unsqueeze(0).expand(bs, seq_len)
        
        loss = graph.score_entropy(log_score, sigma, perturbed_batch, input_ids)
        active_loss = loss[:, num_settled:num_settled+metaschedule.width] 
        loss = (dsigma[:, num_settled:num_settled+metaschedule.width] * active_loss * weight_mask[num_settled:num_settled+metaschedule.width] ) # Shape: [batch_size, seq_len - 1] or O([bs, seq_len])
        return loss
    
    def ar_ppl_fn2(model, batch, perturbed_batch=None):
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if metaschedule is None:
            raise ValueError('Metaschedule must be set when using annealing.')
        if annealing.efficient:
            raise ValueError(f'Efficent mode is not supported for AR PPL for now...')
        else:
            # Sample valid current_step (similar to inference)
            current_step = np.random.randint(1, metaschedule.max_step(seq_len))

            t, attn_mask, _, active_mask, settled_mask, weight_mask = msutils.compute_t_and_attn_autoregressive(
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
        
        log_score_fn = model_utils.get_score_fn(model, train=False, sampling=True)
        log_score = log_score_fn(perturbed_batch, sigma, positions=positions, attn_mask=attn_mask, settled_tokens=settled_positions_mask if not annealing.efficient else None)
        
        bs, seq_len, vocab_dim = log_score.shape
        num_settled = metaschedule(current_step).num_settled
        loss = graph.score_entropy(log_score[:, :-1], sigma[:, 1:], perturbed_batch[:, 1:], input_ids[:, 1:])
        active_loss = loss[:, num_settled:num_settled+metaschedule.width] 
        loss = (dsigma[:, num_settled:num_settled+active_loss.shape[1]] * active_loss * weight_mask[num_settled:num_settled+active_loss.shape[1]]) # Shape: [batch_size, seq_len - 1] or O([bs, seq_len])
        return loss

    def causal_ppl_fn(model, batch):
        """
        Modified loss function for training and evaluation.
        Returns the per-token cross-entropy loss tensor.
        """
        import torch.nn.functional as F
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        ignore_index = 50257
        
        model.train()  # Ensure the model is in train mode if needed
        
        t = torch.ones(batch_size, seq_len, device=device) * sampling_eps
        sigma, dsigma = noise(t)
        
        attn_mask = msutils.create_causal_attention(batch_size, seq_len, seq_len, device)
        
        if hasattr(model.module, "causal_head"):
            output = model.module.causal_head(input_ids, sigma, positions=positions, attn_mask=attn_mask)
        else:
            output = model(input_ids, sigma, positions=positions, attn_mask=attn_mask)
        
        # Remove the last token for prediction
        logits = output[:, :-1].reshape(-1, ignore_index + 1)
        # Shift input_ids to get target tokens
        ce_input_ids = input_ids[:, 1:].reshape(-1)
        
        # Compute per-token cross entropy loss without reduction.
        per_token_loss = F.cross_entropy(logits, ce_input_ids, reduction='none', ignore_index=ignore_index)
        ce_loss = per_token_loss.view(batch_size, seq_len - 1)
        
        return ce_loss
    
    if method == 'AR':
        return ar_ppl_fn2
    elif method == 'Causal':
        return causal_ppl_fn
    elif method == 'SAR':
        return sar_ppl_fn3
    else:
        raise ValueError(f'PPL method {method} not supported yet!')

def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer



def get_scheduler(config, optimizer):
    def lr_lambda(current_step, warmup_steps, total_steps):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if config.optim.scheduler == 'linear':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / config.training.n_iters)
    elif config.optim.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.n_iters)
    elif config.optim.scheduler == 'lambda':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, config.optim.warmup, config.training.n_iters))
    else:
        raise NotImplementedError(
            f'Scheduler {config.optim.scheduler} not supported yet!')
    return scheduler


def get_step_fn(config, accelerator, noise, graph, train, metaschedule=None):
    # auto_regressive = True if config.annealing.sampling_method == "AR" or config.annealing.sampling_method == "Causal" else False
    if hasattr(config.training, 'loss_type') and config.training.loss_type == 'DCE':
        loss_fn = get_DCE_loss_fn(
            noise, graph, train, 
            metaschedule=metaschedule, 
            steps_per_level=config.annealing.steps_per_level, 
            efficient=config.annealing.efficient,
            graph_type=config.graph.type
        )
    else:
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
                # adjust ce_loss_weight from it's original value down to 0 using a linear schedule based on the current step and config.training.warmup_iter value
                # if config.training.warmup_iter > 0:
                #     cd_loss_weight = max(0.5, config.annealing.ce_loss_weight - (config.annealing.ce_loss_weight / config.training.warmup_iter) * state['step'])
                # else:
                #     cd_loss_weight = config.annealing.ce_loss_weight
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



def get_zeroshot_ppl_fn(config, noise, graph, train, metaschedule=None, method='AR'):
    loss_fn = get_ppl_function(
            noise, graph, train,
            annealing=config.annealing,
            metaschedule=metaschedule, 
            sampling_eps=config.annealing.sampling_eps,
            method=method
        )
    def step_fn(state, batch):
        model = state['model']
    
        with torch.no_grad():
            loss = loss_fn(model, batch)
        return loss.detach()
    return step_fn