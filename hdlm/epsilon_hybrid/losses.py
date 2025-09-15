import torch
import numpy as np
from hdlm.model import utils as model_utils
import hdlm.metaschedule_utils as msutils
from flash_attn.losses.cross_entropy import CrossEntropyLoss


def get_loss_fn(train, sampling_eps=1e-4, mask_index = 50_257, metaschedule=None, annealing=None, type='aligned'):
    loss_func = CrossEntropyLoss(reduction='none')
    def aligned_loss_fn(model, batch):
        """
        loss function from https://arxiv.org/abs/2410.18514
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        t = torch.rand((batch_size,), device=device)

        p_mask = (1 - sampling_eps) * t + sampling_eps
        p_mask = p_mask[:, None].repeat(1, seq_len)

        mask_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        noisy_batch = torch.where(mask_indices, mask_index, input_ids)
        if train:
            model.train()
        else:
            model.eval()
        logits = model(noisy_batch, positions)
        loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        return loss
    
    def shifted_loss_fn(model, batch):
        """
        Simple shifted (next token) prediction loss function without annealing schedule.
        This corresponds to the diff_sample_shifted function's approach.
        The key difference from aligned_loss_fn is that we predict the next token after a visible token.
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        
        # Get random masking probability for each batch
        t = torch.rand((batch_size,), device=device)
        p_mask = (1 - sampling_eps) * t + sampling_eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        mask_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        
        mask_indices[:, 0] = False  # Never mask the first token (e.g., BOS token)
        
        noisy_batch = torch.where(mask_indices, mask_index, input_ids)
        
        if train:
            model.train()
        else:
            model.eval()
        
        logits = model(noisy_batch, positions)
        
        logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
        targets = input_ids[:, 1:]  # (batch_size, seq_len-1)
        
        current_visible = ~mask_indices[:, :-1]  # Token at position i is visible
        next_masked = mask_indices[:, 1:]       # Token at position i+1 is masked
        
        valid_positions = current_visible & next_masked
        
        if valid_positions.any():
            flat_logits = logits.reshape(-1, logits.size(-1))  # (batch_size * (seq_len-1), vocab_size)
            flat_targets = targets.reshape(-1)  # (batch_size * (seq_len-1))
            
            token_losses = loss_func(flat_logits, flat_targets)
            
            token_losses = token_losses.view(batch_size, seq_len - 1)
            
            scaling_factor = torch.ones_like(p_mask[:, 1:])
            scaling_factor[valid_positions] = 1.0 / p_mask[:, 1:][valid_positions].clamp(min=1e-4)
            
            scaled_loss = token_losses * scaling_factor * valid_positions.float()
            
            loss = scaled_loss.sum() / valid_positions.sum().clamp(min=1)
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def aligned_annealing_loss_fn(model, batch):
        """
        a borrow of https://arxiv.org/abs/2410.18514 loss function based on annealing schedule
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape

        current_step = np.random.randint(1, metaschedule.max_step(seq_len))

        t, attn_mask, _, active_mask, _, _ = msutils.compute_t_and_attn(
            batch_size, seq_len, metaschedule, current_step,
            context_attention_type=annealing.attention.context_type, block_attention_type=annealing.attention.block_type,
            device=device, eps=sampling_eps)
        if not annealing.match_inference:
            step = 1 / (metaschedule.worthless - 1)
            start_t = t[active_mask]
            end_t = start_t - step
            t[active_mask] = torch.clamp(torch.rand((1, ), device=device) * (start_t - end_t) + end_t, min=sampling_eps, max=1 - sampling_eps)
        
        active_mask = active_mask.unsqueeze(0).repeat(batch_size, 1)
        p_mask = t.unsqueeze(0).repeat(batch_size, 1)

        mask_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        mask_indices = mask_indices & active_mask
        noisy_batch = torch.where(mask_indices, mask_index, input_ids)
        if train:
            model.train()
        else:
            model.eval()
        logits = model(noisy_batch, positions, attn_mask=attn_mask)
        loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
        loss = loss.sum() / active_mask.sum()
        return loss
    
    def shifted_annealing_loss_fn(model, batch):
        """
        a borrow of https://arxiv.org/abs/2410.18514 loss function based on annealing schedule
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape

        current_step = np.random.randint(1, metaschedule.max_step(seq_len))

        t, attn_mask, _, active_mask, _, _ = msutils.compute_t_and_attn_autoregressive(
            batch_size, seq_len, metaschedule, current_step,
            context_attention_type=annealing.attention.context_type, block_attention_type=annealing.attention.block_type,
            device=device, eps=sampling_eps, post_active_context=False)
        if not annealing.match_inference:
            step = 1 / (metaschedule.worthless - 1)
            start_t = t[active_mask]
            end_t = start_t - step
            t[active_mask] = torch.clamp(torch.rand((1, ), device=device) * (start_t - end_t) + end_t, min=sampling_eps, max=1 - sampling_eps)
        t[~active_mask] = sampling_eps
        
        active_mask = active_mask.unsqueeze(0).repeat(batch_size, 1)
        p_mask = t.unsqueeze(0).repeat(batch_size, 1)

        mask_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        mask_indices = mask_indices & active_mask
        loss_mask = ~active_mask | mask_indices
        noisy_batch = torch.where(mask_indices, mask_index, input_ids)
        if train:
            model.train()
        else:
            model.eval()
        logits = model(noisy_batch, positions, attn_mask=attn_mask)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))  # [batch_size*(seq_len-1), vocab_size]
        targets = input_ids[:, 1:].reshape(-1)                   # [batch_size*(seq_len-1)]
        
        token_losses = loss_func(logits, targets)
        token_losses = token_losses.view(batch_size, seq_len - 1)
        loss_mask = loss_mask[:, 1:]
        mask_indices = mask_indices[:, 1:]
        p_mask = p_mask[:, 1:]
        scaling_factor = torch.ones_like(p_mask)

        scaling_factor[mask_indices] = 1.0 / p_mask[mask_indices].clamp(min=1e-4)
        block_weighting = model_utils.get_block_weighting(seq_len - 1, metaschedule.width, device)
        scaled_loss = token_losses * scaling_factor * loss_mask * block_weighting
        return scaled_loss.sum() / loss_mask.sum().clamp(min=1)
        
    if 'aligned' in type:
        if metaschedule is not None:
            return aligned_annealing_loss_fn
        else:
            return aligned_loss_fn    
    elif 'shifted' in type:
        if metaschedule is not None:
            return shifted_annealing_loss_fn
        else:
            return shifted_loss_fn
    else:
        raise ValueError(f'Loss type {type} not supported yet!')    


def get_loss_fn_hybrid(train, sampling_eps=1e-4, mask_index=50_257, epsilon=0.01, lamb=1.0, metaschedule=None, annealing=None, type='aligned'):
    """
    Implementation of the ϵ-hybrid loss function.
    
    This combines token masking with a small probability of random token corruption (shuffling).
    Different weights are applied based on the type of corruption:
    - Masked tokens: weight = 1.0
    - Unmasked but shuffled tokens: weight = λ(1-ϵ)
    - Unmasked and not shuffled tokens: weight = λϵ
    
    Args:
        train: Whether in training mode
        sampling_eps: Minimum probability for masking
        mask_index: Token ID used for masking
        epsilon: Probability of shuffling a non-masked token
        lamb: Weighting factor for unmasked tokens
        metaschedule: Schedule for masking (optional)
        annealing: Annealing parameters (optional)
        type: 'aligned' or 'shifted' prediction style
    """
    loss_func = CrossEntropyLoss(reduction='none')
    
    def hybrid_aligned_loss_fn(model, batch):
        """
        Hybrid loss function for the aligned token prediction.
        Combines masking with random token shuffling and weighted loss.
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        
        # Determine masking probability (noise level dependent)
        t = torch.rand((batch_size,), device=device)
        p_mask = (1 - sampling_eps) * t + sampling_eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        # Apply token masking based on p_mask
        mask_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        noisy_batch = input_ids.clone()
        
        shuffled_indices = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        
        unmasked_indices = ~mask_indices
        shuffle_candidates = torch.rand((batch_size, seq_len), device=device) < epsilon
        shuffle_indices = unmasked_indices & shuffle_candidates
        
        if shuffle_indices.any():
            
            random_tokens = torch.randint(0, mask_index, size=(shuffle_indices.sum().item(),), device=device)
            
            noisy_batch[shuffle_indices] = random_tokens
            shuffled_indices[shuffle_indices] = True
        
        noisy_batch[mask_indices] = mask_index
        
        # Run the model on corrupted inputs
        if train:
            model.train()
        else:
            model.eval()
        
        logits = model(noisy_batch, positions)
        
        token_losses = loss_func(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1))
        token_losses = token_losses.view(batch_size, seq_len)
        
        weights = torch.zeros_like(token_losses, device=device)
        
        weights[mask_indices] = 1.0 / p_mask[mask_indices].clamp(min=1e-4)
        
        unmasked_and_shuffled = (~mask_indices) & shuffled_indices
        weights[unmasked_and_shuffled] = lamb * (1 - epsilon) / (1 - p_mask[unmasked_and_shuffled]) # divide by (1 - p_mask[unmasked_and_shuffled])
        
        unmasked_and_not_shuffled = (~mask_indices) & (~shuffled_indices)
        weights[unmasked_and_not_shuffled] = lamb * epsilon  / (1 - p_mask[unmasked_and_not_shuffled]) # divide by (1 - p_mask[unmasked_and_not_shuffled]
        
        weighted_loss = token_losses * weights
        final_loss = weighted_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        return final_loss
    
    def hybrid_aligned_annealing_loss_fn(model, batch):
        """
        Hybrid loss function with annealing schedule for the aligned token prediction.
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        
        current_step = np.random.randint(1, metaschedule.max_step(seq_len))
        
        t, attn_mask, _, active_mask, _, _ = msutils.compute_t_and_attn(
            batch_size, seq_len, metaschedule, current_step,
            context_attention_type=annealing.attention.context_type, 
            block_attention_type=annealing.attention.block_type,
            device=device, eps=sampling_eps)
            
        if not annealing.match_inference:
            step = 1 / (metaschedule.worthless - 1)
            start_t = t[active_mask]
            end_t = start_t - step
            t[active_mask] = torch.clamp(torch.rand((1, ), device=device) * (start_t - end_t) + end_t, 
                                        min=sampling_eps, max=1 - sampling_eps)
        
        # Expand masks to batch size
        active_mask_batch = active_mask.unsqueeze(0).repeat(batch_size, 1)
        p_mask = t.unsqueeze(0).repeat(batch_size, 1)
        
        # Apply token masking based on p_mask and active_mask
        mask_probs = torch.rand((batch_size, seq_len), device=device)
        mask_indices = (mask_probs < p_mask) & active_mask_batch
        
        # Create corrupted batch
        noisy_batch = input_ids.clone()
        
        # Create a tensor to track which tokens were shuffled
        shuffled_indices = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        
        # For all tokens that will not be masked, apply random shuffling with probability epsilon
        unmasked_indices = ~mask_indices & active_mask_batch
        shuffle_candidates = torch.rand((batch_size, seq_len), device=device) < epsilon
        shuffle_indices = unmasked_indices & shuffle_candidates
        
        if shuffle_indices.any():
            # Get the vocabulary size from input_ids
            vocab_size = mask_index
            
            # Generate random token IDs for shuffling
            random_tokens = torch.randint(0, vocab_size, size=(shuffle_indices.sum().item(),), device=device)
            
            # Apply the shuffling
            noisy_batch[shuffle_indices] = random_tokens
            shuffled_indices[shuffle_indices] = True
        
        # Apply masking after shuffling
        noisy_batch[mask_indices] = mask_index
        
        # Run the model on corrupted inputs
        if train:
            model.train()
        else:
            model.eval()
            
        logits = model(noisy_batch, positions, attn_mask=attn_mask)
        
        # Calculate token-level losses
        token_losses = loss_func(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1))
        token_losses = token_losses.view(batch_size, seq_len)
        
        # Apply different weights based on corruption type and active mask
        weights = torch.zeros_like(token_losses, device=device)
        
        # Only consider tokens in the active mask
        mask_indices_active = mask_indices & active_mask_batch
        unmasked_and_shuffled = (~mask_indices) & shuffled_indices & active_mask_batch
        unmasked_and_not_shuffled = (~mask_indices) & (~shuffled_indices) & active_mask_batch
        
        # Masked tokens (weight = 1.0 / p_mask)
        if mask_indices_active.any():
            weights[mask_indices_active] = 1.0 / p_mask[mask_indices_active].clamp(min=1e-4)
        
        # Unmasked & shuffled tokens (weight = λ(1-ϵ))
        if unmasked_and_shuffled.any():
            weights[unmasked_and_shuffled] = lamb * (1 - epsilon)
        
        # Unmasked & not shuffled tokens (weight = λϵ)
        if unmasked_and_not_shuffled.any():
            weights[unmasked_and_not_shuffled] = lamb * epsilon
        
        # Apply weights and calculate final loss (only for active tokens)
        weighted_loss = token_losses * weights
        final_loss = weighted_loss.sum() / active_mask_batch.sum().clamp(min=1.0)
        
        return final_loss
    
    def hybrid_shifted_loss_fn(model, batch):
        """
        Hybrid loss function for the shifted token prediction (next token prediction).
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        
        # Determine masking probability (noise level dependent)
        t = torch.rand((batch_size,), device=device)
        p_mask = (1 - sampling_eps) * t + sampling_eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        # Apply token masking based on p_mask (never mask first token)
        mask_probs = torch.rand((batch_size, seq_len), device=device)
        mask_indices = mask_probs < p_mask
        mask_indices[:, 0] = False  # Never mask the first token (e.g., BOS token)
        
        # Create corrupted batch
        noisy_batch = input_ids.clone()
        
        # Create a tensor to track which tokens were shuffled
        shuffled_indices = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        
        # For all tokens that will not be masked, apply random shuffling with probability epsilon
        unmasked_indices = ~mask_indices
        shuffle_candidates = torch.rand((batch_size, seq_len), device=device) < epsilon
        shuffle_indices = unmasked_indices & shuffle_candidates
        
        if shuffle_indices.any():
            
            # Generate random token IDs for shuffling
            random_tokens = torch.randint(0, mask_index, size=(shuffle_indices.sum().item(),), device=device)
            
            # Apply the shuffling
            noisy_batch[shuffle_indices] = random_tokens
            shuffled_indices[shuffle_indices] = True
        
        # Apply masking after shuffling
        noisy_batch[mask_indices] = mask_index
        
        if train:
            model.train()
        else:
            model.eval()
            
        logits = model(noisy_batch, positions)
        
        # For shifted prediction, we use the logits from positions 0 to N-1
        # to predict tokens at positions 1 to N
        logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
        targets = input_ids[:, 1:]  # (batch_size, seq_len-1)
        
        # We need to identify the context-next pairs where:
        # 1. The context token is visible (not masked)
        # 2. Looking at the original target (regardless of whether it was masked or shuffled)
        current_visible = ~mask_indices[:, :-1]  # Token at position i is visible
        
        # Calculate token-level losses for all pairs where context is visible
        token_losses = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        token_losses = token_losses.view(batch_size, seq_len - 1)
        
        # Apply different weights based on corruption type for the target tokens
        weights = torch.zeros_like(token_losses, device=device)
        
        # Get masks for the target positions (positions 1 to N)
        target_mask_indices = mask_indices[:, 1:]
        target_shuffled_indices = shuffled_indices[:, 1:]
        
        # Only calculate loss where the context token is visible
        valid_positions = current_visible
        
        # Different weighting based on target token corruption:
        # 1. If target was masked: weight = 1.0 / p_mask
        masked_targets = target_mask_indices & valid_positions
        if masked_targets.any():
            weights[masked_targets] = 1.0 / p_mask[:, 1:][masked_targets].clamp(min=1e-4)
        
        # 2. If target was shuffled: weight = λ(1-ϵ)
        shuffled_targets = target_shuffled_indices & valid_positions & (~target_mask_indices)
        if shuffled_targets.any():
            weights[shuffled_targets] = lamb * (1 - epsilon)
        
        # 3. If target was unchanged: weight = λϵ
        unchanged_targets = valid_positions & (~target_mask_indices) & (~target_shuffled_indices)
        if unchanged_targets.any():
            weights[unchanged_targets] = lamb * epsilon
        
        # Apply weights and calculate final loss
        weighted_loss = token_losses * weights
        final_loss = weighted_loss.sum() / (input_ids.shape[0] * (input_ids.shape[1] - 1))
        
        return final_loss
    
    # need to be checked
    def hybrid_shifted_annealing_loss_fn(model, batch):
        """
        Hybrid loss function with annealing schedule for the shifted token prediction.
        """
        device = batch['input_ids'].device
        input_ids = batch['input_ids']
        positions = batch['positions']
        batch_size, seq_len = input_ids.shape
        
        # Get current step and compute masks using metaschedule
        current_step = np.random.randint(1, metaschedule.max_step(seq_len))
        
        t, attn_mask, _, active_mask, _, _ = msutils.compute_t_and_attn_autoregressive(
            batch_size, seq_len, metaschedule, current_step,
            context_attention_type=annealing.attention.context_type, 
            block_attention_type=annealing.attention.block_type,
            device=device, eps=sampling_eps, post_active_context=False)
            
        if not annealing.match_inference:
            # Sample t from a uniform distribution of the target range
            step = 1 / (metaschedule.worthless - 1)
            start_t = t[active_mask]
            end_t = start_t - step
            t[active_mask] = torch.clamp(torch.rand((1, ), device=device) * (start_t - end_t) + end_t, 
                                       min=sampling_eps, max=1 - sampling_eps)
        t[~active_mask] = sampling_eps
        
        # Expand masks to batch size
        active_mask_batch = active_mask.unsqueeze(0).repeat(batch_size, 1)
        p_mask = t.unsqueeze(0).repeat(batch_size, 1)
        
        # Apply token masking based on p_mask and active_mask
        mask_probs = torch.rand((batch_size, seq_len), device=device)
        mask_indices = (mask_probs < p_mask) & active_mask_batch
        
        # Create corrupted batch
        noisy_batch = input_ids.clone()
        
        # Create a tensor to track which tokens were shuffled
        shuffled_indices = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        
        # For all tokens that will not be masked, apply random shuffling with probability epsilon
        unmasked_indices = ~mask_indices & active_mask_batch
        shuffle_candidates = torch.rand((batch_size, seq_len), device=device) < epsilon
        shuffle_indices = unmasked_indices & shuffle_candidates
        
        if shuffle_indices.any():
            
            # Generate random token IDs for shuffling
            random_tokens = torch.randint(0, mask_index, size=shuffle_indices.sum().item(), device=device)
            # Apply the shuffling
            noisy_batch[shuffle_indices] = random_tokens
            shuffled_indices[shuffle_indices] = True
        
        # Apply masking after shuffling
        noisy_batch[mask_indices] = mask_index
        
        # Run the model on corrupted inputs
        if train:
            model.train()
        else:
            model.eval()
            
        logits = model(noisy_batch, positions, attn_mask=attn_mask)
        
        # For shifted prediction, we use logits from positions 0 to N-1 for tokens 1 to N
        logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
        targets = input_ids[:, 1:]  # (batch_size, seq_len-1)
        
        # Calculate token-level losses
        token_losses = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        token_losses = token_losses.view(batch_size, seq_len - 1)
        
        # Get masks for positions 1 to N (targets)
        active_mask_targets = active_mask_batch[:, 1:]
        mask_indices_targets = mask_indices[:, 1:]
        shuffled_indices_targets = shuffled_indices[:, 1:]
        p_mask_targets = p_mask[:, 1:]
        
        # The condition for calculating loss is:
        # 1. The token at position i (context) is visible (not masked)
        # 2. The token at position i+1 (target) is in the active mask
        current_visible = ~mask_indices[:, :-1]
        loss_mask = current_visible & active_mask_targets
        
        # Apply different weights based on corruption type
        weights = torch.zeros_like(token_losses, device=device)
        
        # 1. If target was masked: weight = 1.0 / p_mask
        masked_targets = mask_indices_targets & loss_mask
        if masked_targets.any():
            weights[masked_targets] = 1.0 / p_mask_targets[masked_targets].clamp(min=1e-4)
        
        # 2. If target was shuffled: weight = λ(1-ϵ)
        shuffled_targets = shuffled_indices_targets & loss_mask & (~mask_indices_targets)
        if shuffled_targets.any():
            weights[shuffled_targets] = lamb * (1 - epsilon)
        
        # 3. If target was unchanged: weight = λϵ
        unchanged_targets = loss_mask & (~mask_indices_targets) & (~shuffled_indices_targets)
        if unchanged_targets.any():
            weights[unchanged_targets] = lamb * epsilon
        
        # Apply block weighting from the metaschedule if available
        block_weighting = model_utils.get_block_weighting(seq_len - 1, metaschedule.width, device)
        
        # Apply weights and calculate final loss
        weighted_loss = token_losses * weights * block_weighting
        final_loss = weighted_loss.sum() / (weights * block_weighting).sum().clamp(min=1.0)
        
        return final_loss
    
    # Return the appropriate loss function based on type and metaschedule
    if 'aligned' in type:
        if metaschedule is not None:
            return hybrid_aligned_annealing_loss_fn
        else:
            return hybrid_aligned_loss_fn    
    elif 'shifted' in type:
        if metaschedule is not None:
            return hybrid_shifted_annealing_loss_fn
        else:
            return hybrid_shifted_loss_fn
    else:
        raise ValueError(f'Loss type {type} not supported yet!')


def get_step_fn(config, accelerator, train, metaschedule=None, annealing=None, type='aligned'):    
    # Determine which loss function to use based on config
    if hasattr(config.training, 'loss_type') and config.training.loss_type == 'hybrid':
        # Use hybrid loss with specified epsilon and lambda values
        epsilon = getattr(config.training, 'epsilon', 0.01)  # Default to 0.01 if not specified
        lamb = getattr(config.training, 'lambda', 0.1)  # Default to 0.1 if not specified
        loss_fn = get_loss_fn_hybrid(
            train,
            sampling_eps=config.annealing.sampling_eps,
            epsilon=epsilon,
            lamb=lamb,
            metaschedule=metaschedule,
            annealing=annealing,
            type=type
        )
    else:
        # Use standard loss function
        loss_fn = get_loss_fn(
            train,
            sampling_eps=config.annealing.sampling_eps,
            metaschedule=metaschedule,
            annealing=annealing,
            type=type
        )

    def step_fn(state, batch):
        model = state['model']
        if train:
            optimizer = state['optimizer']
            scheduler = state['scheduler']
            with accelerator.accumulate(model):
                # Zero gradients
                optimizer.zero_grad()

                loss = loss_fn(model, batch)
                loss = loss.mean()
                
                accelerator.backward(loss)
                state['step'] += 1
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"Parameter {name} did not receive a gradient.")
                optimizer.step()
                
                scheduler.step()
                state['ema'].step(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                loss = loss.mean()
                ema.restore(model.parameters())

        return loss.detach()

    return step_fn
