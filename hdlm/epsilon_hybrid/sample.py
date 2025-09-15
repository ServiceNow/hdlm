import torch
import hdlm.metaschedule_utils as msutils


def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()  
def full_diff(model, metaschedule=None, annealing=None, prompt=None, batch_size=1, alg='original', steps=512, temperature=1., cfg_scale=0.,
                context_length=1024, eps=1e-5, dim=50257, device='cuda', repetition_penalty=1.2, eta=0.015):
    """
    Full diffusion sampling - the main reference implementation.
    
    Available algorithms: original, acs, remask, remask_adhoc_shuffle, remdm
    """
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    positions = torch.arange(context_length, device=device).unsqueeze(0).expand(batch_size, -1)
    if prompt is not None:
        x[:, :prompt.shape[1]] = prompt.clone()

    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    for i in range(steps):
        mask_index = (x == dim)
        non_mask_index = ~mask_index
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[:, :prompt.shape[1]] = dim
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_, positions[:, x_.shape[1]])
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits, un_logits = logits, un_logits
            else:
                logits = model(x, positions)

        if cfg_scale > 0.:
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == "original":
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = x.clone()
            transfer_index_t_s_mask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < p_transfer
            logits_with_noise_unmaks = add_gumbel_noise(logits[transfer_index_t_s_mask & mask_index], temperature=temperature)
            x0[transfer_index_t_s_mask & mask_index] = torch.argmax(logits_with_noise_unmaks, dim=-1)
            x = x0.clone()
            if prompt is not None:
                x[:, :prompt.shape[1]] = prompt.clone()


        elif alg == 'acs':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = x.clone()
            transfer_index_t_s_mask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < p_transfer
            transfer_index_t_s_uniform = torch.rand_like(x0, device='cuda', dtype=torch.float64) < eta * (1 - p_transfer)
            logits_with_noise_unmaks = add_gumbel_noise(logits[transfer_index_t_s_mask & mask_index], temperature=temperature)
            logits_with_noise_shuffle = add_gumbel_noise(logits[transfer_index_t_s_uniform & non_mask_index], temperature=temperature)
            x0[transfer_index_t_s_mask & mask_index] = torch.argmax(logits_with_noise_unmaks, dim=-1)
            x0[transfer_index_t_s_uniform & non_mask_index] = torch.argmax(logits_with_noise_shuffle, dim=-1)
            x = x0.clone()
            if prompt is not None:
                x[:, :prompt.shape[1]] = prompt.clone()
        

        elif alg == "remask":
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = x.clone()
            logits_with_noise_unmask = add_gumbel_noise(logits[mask_index], temperature=temperature)
            x0[mask_index] = torch.argmax(logits_with_noise_unmask, dim=-1)
            transfer_index_t_s_remask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < 1 - p_transfer
            x0[transfer_index_t_s_remask] = torch.zeros_like(x0[transfer_index_t_s_remask], device=device, dtype=torch.long) + dim
            x = x0.clone()
            if prompt is not None:
                x[:, :prompt.shape[1]] = prompt.clone()
        elif alg == "remask_adhoc_shuffle":
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = x.clone()
            transfer_index_t_s_uniform = torch.rand_like(x0, device='cuda', dtype=torch.float64) < eta * (1 - p_transfer)
            logits_with_noise_unmask = add_gumbel_noise(logits[mask_index], temperature=temperature)
            logits_with_noise_shuffle = add_gumbel_noise(logits[transfer_index_t_s_uniform & non_mask_index], temperature=temperature)
            x0[mask_index] = torch.argmax(logits_with_noise_unmask, dim=-1)
            x0[transfer_index_t_s_uniform & non_mask_index] = torch.argmax(logits_with_noise_shuffle, dim=-1)
            # remask possible for all tokens
            transfer_index_t_s_remask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < 1 - p_transfer
            x0[transfer_index_t_s_remask] = torch.zeros_like(x0[transfer_index_t_s_remask], device=device, dtype=torch.long) + dim
            x = x0.clone()
            if prompt is not None:
                x[:, :prompt.shape[1]] = prompt.clone()
        elif alg == "remdm":
            # Define schedule parameters from t and s.
            alpha_t = 1 - t.item()
            alpha_s = 1 - s.item()
            # Compute sigma: controlling the amount of probability mass given to the mask token.
            if alpha_t > 0:
                sigma = min(eta, (1 - alpha_s) / alpha_t)
            else:
                sigma = eta
            # Get the model's probability distribution.
            p_x0 = torch.softmax(logits, dim=-1)
            # Mix the distribution:
            #   (1-sigma) portion from the model's prediction,
            #   sigma portion forced on the mask token (index given by dim).
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., dim] = sigma
            new_logits = torch.log(q_xs + 1e-8)
            new_logits = add_gumbel_noise(new_logits, temperature=temperature)
            x0 = x.clone()
            x0[mask_index] = torch.argmax(new_logits[mask_index], dim=-1)
            x = x0.clone()
            if prompt is not None:
                x[:, :prompt.shape[1]] = prompt.clone()
        else:
            raise NotImplementedError(f"Algorithm '{alg}' not supported in full_diff")

    return x



@torch.no_grad()
def semi_diff(model, prompt=None, batch_size=1, alg='original', steps=512, temperature=1., cfg_scale=0.,
                context_length=1024, eps=1e-5, dim=50257, device='cuda', eta=0.01, block_size=16):
    """
    Semi-autoregressive diffusion sampling with block-wise processing.
    
    Available algorithms: original, acs, remask, remask_adhoc_shuffle, remdm
    """
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    positions = torch.arange(context_length, device=device).unsqueeze(0).expand(batch_size, -1)
    if prompt is not None:
        x[:, :prompt.shape[1]] = prompt.clone()
    
    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    for i in range(0, context_length, block_size):
        # we create a mask for the current block
        block_mask = torch.zeros((batch_size, context_length), device=device)
        block_mask[:, i:i+block_size] = 1
        block_mask = block_mask.bool()
        for j in range(steps):
            mask_index = x[:, :i+block_size] == dim
            non_mask_index = ~mask_index
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[:, :prompt.shape[1]] = dim
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_, positions[:, x_.shape[1]])
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits, un_logits = logits, un_logits
                else:
                    logits = model(x[:, :i+block_size], positions[:, :i+block_size])

            if cfg_scale > 0.:
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            t = timesteps[j]
            s = timesteps[j + 1]

            if alg == "original":
                p_transfer = 1 - s / t if j < steps - 1 else 1
                x0 = x[:, :i+block_size].clone()
                transfer_index_t_s_mask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < p_transfer
                logits_with_noise_unmaks = add_gumbel_noise(logits[transfer_index_t_s_mask & mask_index & block_mask[:, :i+block_size]], temperature=temperature)
                x0[transfer_index_t_s_mask & mask_index & block_mask[:, :i+block_size]] = torch.argmax(logits_with_noise_unmaks, dim=-1)
                x[:, :i+block_size] = x0.clone()
                if prompt is not None:
                    x[:, :prompt.shape[1]] = prompt.clone()

            elif alg == 'acs':
                p_transfer = 1 - s / t if j < steps - 1 else 1
                x0 = x[:, :i+block_size].clone()

                transfer_index_t_s_mask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < p_transfer
                transfer_index_t_s_uniform = torch.rand_like(x0, device='cuda', dtype=torch.float64) < eta * (1 - p_transfer)
                logits_with_noise_unmaks = add_gumbel_noise(logits[transfer_index_t_s_mask & mask_index & block_mask[:, :i+block_size]], temperature=temperature)
                logits_with_noise_shuffle = add_gumbel_noise(logits[transfer_index_t_s_uniform & non_mask_index & block_mask[:, :i+block_size]], temperature=temperature)
                x0[transfer_index_t_s_mask & mask_index & block_mask[:, :i+block_size]] = torch.argmax(logits_with_noise_unmaks, dim=-1)
                x0[transfer_index_t_s_uniform & non_mask_index & block_mask[:, :i+block_size]] = torch.argmax(logits_with_noise_shuffle, dim=-1)
                x[:, :i+block_size] = x0.clone()
                if prompt is not None:
                    x[:, :prompt.shape[1]] = prompt.clone()


            elif alg == "remask":
                p_transfer = 1 - s / t if j < steps - 1 else 1
                x0 = x[:, :i+block_size].clone()
                logits_with_noise_unmask = add_gumbel_noise(logits[mask_index], temperature=temperature)
                x0[mask_index] = torch.argmax(logits_with_noise_unmask, dim=-1)
                transfer_index_t_s_remask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < 1 - p_transfer
                x0[transfer_index_t_s_remask] = torch.zeros_like(x0[transfer_index_t_s_remask], device=device, dtype=torch.long) + dim
                x[:, :i+block_size] = x0.clone()
                if prompt is not None:
                    x[:, :prompt.shape[1]] = prompt.clone()
            elif alg == "remask_adhoc_shuffle":
                p_transfer = 1 - s / t if j < steps - 1 else 1
                x0 = x[:, :i+block_size].clone()
                transfer_index_t_s_uniform = torch.rand_like(x0, device='cuda', dtype=torch.float64) < eta * (1 - p_transfer)
                logits_with_noise_unmask = add_gumbel_noise(logits[mask_index], temperature=temperature)
                logits_with_noise_shuffle = add_gumbel_noise(logits[transfer_index_t_s_uniform & non_mask_index & block_mask[:, :i+block_size]], temperature=temperature)
                x0[mask_index] = torch.argmax(logits_with_noise_unmask, dim=-1)
                x0[transfer_index_t_s_uniform & non_mask_index & block_mask[:, :i+block_size]] = torch.argmax(logits_with_noise_shuffle, dim=-1)
                # remask possible for all tokens
                transfer_index_t_s_remask = torch.rand_like(x0, device='cuda', dtype=torch.float64) < 1 - p_transfer
                x0[transfer_index_t_s_remask] = torch.zeros_like(x0[transfer_index_t_s_remask], device=device, dtype=torch.long) + dim
                x[:, :i+block_size] = x0.clone()
                if prompt is not None:
                    x[:, :prompt.shape[1]] = prompt.clone()
            elif alg == "remdm":
                # Define schedule parameters from t and s.
                alpha_t = 1 - t.item()
                alpha_s = 1 - s.item()
                # Compute sigma: controlling the amount of probability mass given to the mask token.
                if alpha_t > 0:
                    sigma = min(eta, (1 - alpha_s) / alpha_t)
                else:
                    sigma = eta
                # Get the model's probability distribution.
                p_x0 = torch.softmax(logits, dim=-1)
                # Mix the distribution:
                #   (1-sigma) portion from the model's prediction,
                #   sigma portion forced on the mask token (index given by dim).
                q_xs = p_x0 * (1 - sigma)
                q_xs[..., dim] = sigma
                new_logits = torch.log(q_xs + 1e-8)
                new_logits = add_gumbel_noise(new_logits, temperature=temperature)
                x0 = x[:, :i+block_size].clone()
                x0[mask_index] = torch.argmax(new_logits[mask_index], dim=-1)
                x[:, :i+block_size] = x0.clone()
                if prompt is not None:
                    x[:, :prompt.shape[1]] = prompt.clone()
            else:
                raise NotImplementedError(f"Algorithm '{alg}' not supported in semi_diff")

    return x




@torch.no_grad()
def semi_diff_shifted(model, metaschedule=None, annealing=None, prompt=None, batch_size=1, alg='original', 
                    context_length=1024, mask_token_id=50257, device='cuda',
                    temperature=1., cfg_scale=0., eps=1e-5, repetition_penalty=1.2):
    """
    Semi-Autoregressive next token sampling using block-wise demasking.
    In this approach, we predict the next token for each masked position.
    """
    # Determine batch size from prompt if provided.
    batch_size = batch_size if prompt is None else prompt.shape[0]
    
    # Initialize the sequence with mask tokens.
    x = torch.full((batch_size, context_length), mask_token_id, dtype=torch.long, device=device)
    # bos 
    x[:, 0] = 50256  # Start with BOS token
    positions = torch.arange(context_length, device=device).unsqueeze(0).expand(batch_size, -1)
    if prompt is not None:
        x[:, :prompt.shape[1]] = prompt.clone()
    
    # Track token frequencies to prevent repetition - ensure we use correct vocab size
    vocab_size = mask_token_id + 1  # Adding 1 to match the logits dimension (50258)
    token_frequencies = torch.zeros(vocab_size, device=device)
    
    total_steps = metaschedule.max_step(context_length)
    
    for i in range(1, total_steps):
        # Get t, attention mask, and active mask for the current step
        t, attn_mask, tokens_to_denoise, active_mask, _, _ = msutils.compute_t_and_attn_autoregressive(
            batch_size, context_length, metaschedule, i,
            annealing.attention.context_type, annealing.attention.block_type, 
            device, eps, post_active_context=False)
        
        # Find positions that need to be unmasked
        mask_index = (x == mask_token_id)
        mask_index = mask_index[:, 1:]
        active_mask = active_mask[1:]
        # Only consider positions within the active block
        mask_index = mask_index & active_mask.unsqueeze(0).expand_as(mask_index)
        
        # Skip if no tokens need to be unmasked
        if not mask_index.any():
            continue
        
        # Calculate transfer probability based on current step and next step
        if i < total_steps - 1:
            # Get next step's t value
            s, _, _, _, _, _ = msutils.compute_t_and_attn_autoregressive(
                batch_size, context_length, metaschedule, i + 1,
                annealing.attention.context_type, annealing.attention.block_type, 
                device, eps, post_active_context=False)
        else:
            # For the last step, use current t value (will result in p_transfer=0)
            s = t
            
        # Calculate base transfer probability
        base_p = 1 - s / t
        base_p = torch.clamp(base_p, min=0.0, max=1.0)
        
        # For tokens that must be denoised, force transfer probability to 1.
        p_transfer = torch.where(tokens_to_denoise, torch.ones_like(base_p), base_p)
        p_transfer_update_batch = p_transfer.unsqueeze(0).repeat(batch_size, 1)
        
        # Run the model to predict next tokens
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Skip if no valid preceding positions
            if not active_mask.any():
                continue
            
            # Handle classifier-free guidance if enabled
            if cfg_scale > 0.:
                # Duplicate x, attn_mask, and positions for classifier-free guidance
                un_x = x.clone()
                if prompt is not None:
                    un_x[:, :prompt.shape[1]] = mask_token_id
                    
                x_cat = torch.cat([x, un_x], dim=0)  # shape: [2*batch_size, context_length]
                positions_cat = positions.repeat(2, 1)  # shape: [2*batch_size, context_length]
                attn_mask_cat = attn_mask.repeat(2, 1, 1)  # Repeat correctly for batch dimension
                
                # Run the model on duplicated inputs
                logits_all = model(x_cat, positions_cat, attn_mask=attn_mask_cat)
                logits, un_logits = torch.chunk(logits_all, 2, dim=0)  # Each: [batch_size, context_length, vocab_dim]
                logits = logits[:, :-1, :]
                un_logits = un_logits[:, :-1, :]
                
                # Get logits for positions preceding masked tokens
                logits = logits[mask_index]
                un_logits = un_logits[mask_index]
                
                # Apply classifier-free guidance
                guided_logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # Get model output with attention mask
                logits = model(x, positions, attn_mask=attn_mask)[:, :-1, :]
                
                # Get logits for positions preceding masked tokens
                guided_logits = logits[mask_index]
        
        # Apply repetition penalty if we have generated tokens
        if i > 1:
            # Get current non-mask tokens
            current_tokens = x[x != mask_token_id]
            if current_tokens.numel() > 0:
                # Update token frequencies
                unique_tokens, counts = torch.unique(current_tokens, return_counts=True)
                token_frequencies[unique_tokens] += counts.float()
                
                # Normalize frequencies to [0, 1] range
                if token_frequencies.max() > 0:
                    norm_frequencies = token_frequencies / token_frequencies.max()
                    
                    # Apply penalty to logits based on token frequency
                    # Make sure penalty has the same size as logits along dimension 1
                    penalty = torch.ones(guided_logits.size(-1), device=device)
                    penalty[:vocab_size] = penalty[:vocab_size] + (repetition_penalty - 1.0) * norm_frequencies
                    
                    # Reshape penalty for broadcasting
                    penalty = penalty.unsqueeze(0).expand(guided_logits.size(0), -1)
                    
                    # Apply penalty by dividing logits (penalizes frequent tokens)
                    guided_logits = guided_logits / penalty
             
        # Expand p_transfer for batch
        p_transfer = p_transfer[1:]
        p_transfer_update_batch = p_transfer.unsqueeze(0).expand(batch_size, -1)
        
        # Sampling logic based on selected algorithm
        if alg == 'original':
            # Initialize tokens to be updated
            active_x = x[:, 1:]
            x0 = torch.zeros_like(active_x[mask_index], device=device, dtype=torch.long) + mask_token_id
            
            # Apply temperature scaling to logits
            logits_temp = guided_logits / max(temperature, 1e-5)
            
            # Determine which tokens to transfer based on probability
            transfer_indices = torch.rand_like(p_transfer_update_batch[mask_index], device=device, dtype=torch.float64) < p_transfer_update_batch[mask_index]
            
            if transfer_indices.any():
                # Add Gumbel noise and sample tokens
                logits_with_noise = add_gumbel_noise(logits_temp[transfer_indices], temperature=temperature)
                x0[transfer_indices] = torch.argmax(logits_with_noise, dim=-1)
            
            # Update sequence
            active_x[mask_index] = x0.clone()            
            x[:, 1:] = active_x
            
            # Update token frequencies with newly generated tokens
            new_tokens = x0[x0 != mask_token_id]
            if new_tokens.numel() > 0:
                unique_new, counts_new = torch.unique(new_tokens, return_counts=True)
                token_frequencies[unique_new] += counts_new.float()

        elif alg == 'acs':
            # ACS (Adaptive Correction Sampler) algorithm for semi-autoregressive shifted sampling
            active_x = x[:, 1:]
            x0 = torch.zeros_like(active_x[mask_index], device=device, dtype=torch.long) + mask_token_id
            
            # Apply temperature scaling to logits
            logits_temp = guided_logits / max(temperature, 1e-5)
            
            # Determine which tokens to transfer vs shuffle
            transfer_indices = torch.rand_like(p_transfer_update_batch[mask_index], device=device, dtype=torch.float64) < p_transfer_update_batch[mask_index]
            
            eta = 0.015
            # Apply adaptive correction shuffling with probability eta * (1 - p_transfer)
            shuffle_prob = eta * (1 - p_transfer_update_batch[mask_index])
            shuffle_indices = torch.rand_like(p_transfer_update_batch[mask_index], device=device, dtype=torch.float64) < shuffle_prob
            
            if transfer_indices.any():
                # Add Gumbel noise and sample tokens for transfer
                logits_with_noise = add_gumbel_noise(logits_temp[transfer_indices], temperature=temperature)
                x0[transfer_indices] = torch.argmax(logits_with_noise, dim=-1)
            
            if shuffle_indices.any():
                # Add Gumbel noise and sample tokens for adaptive correction
                shuffle_logits = add_gumbel_noise(logits_temp[shuffle_indices], temperature=temperature)
                x0[shuffle_indices] = torch.argmax(shuffle_logits, dim=-1)
            
            # Update sequence
            active_x[mask_index] = x0.clone()            
            x[:, 1:] = active_x
            
            # Update token frequencies with newly generated tokens
            new_tokens = x0[x0 != mask_token_id]
            if new_tokens.numel() > 0:
                unique_new, counts_new = torch.unique(new_tokens, return_counts=True)
                token_frequencies[unique_new] += counts_new.float()
                

        elif alg == 'remask':
            # Remask algorithm for semi-autoregressive shifted sampling
            active_x = x[:, 1:]
            x0 = torch.zeros_like(active_x[mask_index], device=device, dtype=torch.long) + mask_token_id
            
            # First, generate for all masked positions
            logits_with_noise = add_gumbel_noise(guided_logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Then remask some positions based on (1 - p_transfer)
            remask_prob = (1 - p_transfer_update_batch[mask_index])
            remask_indices = torch.rand_like(p_transfer_update_batch[mask_index], device=device, dtype=torch.float64) < remask_prob
            x0[remask_indices] = mask_token_id
            
            # Update sequence
            active_x[mask_index] = x0.clone()            
            x[:, 1:] = active_x
            
            # Update token frequencies with newly generated tokens
            new_tokens = x0[x0 != mask_token_id]
            if new_tokens.numel() > 0:
                unique_new, counts_new = torch.unique(new_tokens, return_counts=True)
                token_frequencies[unique_new] += counts_new.float()
                
        elif alg == 'remask_adhoc_shuffle':
            # Remask with adhoc shuffling for semi-autoregressive shifted sampling
            active_x = x[:, 1:]
            x0 = torch.zeros_like(active_x[mask_index], device=device, dtype=torch.long) + mask_token_id
            
            # First, generate for all masked positions
            logits_with_noise = add_gumbel_noise(guided_logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            eta = 0.015
            # Calculate probabilities
            remask_prob = (1 - p_transfer_update_batch[mask_index])
            shuffle_prob = eta * remask_prob
            
            # Apply shuffling to some existing non-mask tokens
            existing_tokens = (active_x != mask_token_id)
            if existing_tokens.any():
                shuffle_existing = torch.rand_like(existing_tokens.float(), device=device) < shuffle_prob.unsqueeze(0).expand_as(existing_tokens).float()
                if shuffle_existing.any():
                    # For simplicity, sample from same logits distribution
                    shuffle_logits = add_gumbel_noise(guided_logits[:shuffle_existing.sum()], temperature=temperature)
                    active_x[shuffle_existing] = torch.argmax(shuffle_logits, dim=-1)
            
            # Then remask some newly generated positions
            remask_indices = torch.rand_like(p_transfer_update_batch[mask_index], device=device, dtype=torch.float64) < remask_prob
            x0[remask_indices] = mask_token_id
            
            # Update sequence
            active_x[mask_index] = x0.clone()            
            x[:, 1:] = active_x
            
            # Update token frequencies with newly generated tokens
            new_tokens = x0[x0 != mask_token_id]
            if new_tokens.numel() > 0:
                unique_new, counts_new = torch.unique(new_tokens, return_counts=True)
                token_frequencies[unique_new] += counts_new.float()
                
        elif alg == 'remdm':
            # ReMDM algorithm for semi-autoregressive shifted sampling
            # Calculate schedule parameters
            base_p_mean = p_transfer.mean()
            alpha_t = 1 - base_p_mean
            alpha_s = alpha_t  # Simplified for this context
            eta = 0.015
            
            # Compute sigma: controlling the amount of probability mass given to the mask token
            if alpha_t > 0:
                sigma = min(eta, (1 - alpha_s) / alpha_t)
            else:
                sigma = eta
                
            # Get the model's probability distribution
            p_x0 = torch.softmax(guided_logits, dim=-1)
            
            # Mix the distribution: (1-sigma) from model, sigma from mask token
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., mask_token_id] = sigma
            
            # Sample from mixed distribution
            new_logits = torch.log(q_xs + 1e-8)
            new_logits = add_gumbel_noise(new_logits, temperature=temperature)
            x0 = torch.argmax(new_logits, dim=-1)
            
            # Update sequence
            active_x = x[:, 1:]
            active_x[mask_index] = x0.clone()            
            x[:, 1:] = active_x
            
            # Update token frequencies with newly generated tokens
            new_tokens = x0[x0 != mask_token_id]
            if new_tokens.numel() > 0:
                unique_new, counts_new = torch.unique(new_tokens, return_counts=True)
                token_frequencies[unique_new] += counts_new.float()
        else:
            raise NotImplementedError(f"Algorithm '{alg}' not supported in semi_diff_shifted")
    
    return x

@torch.no_grad()
def full_diff_shifted(model, metaschedule=None, annealing=None, prompt=None, batch_size=1, alg='original', steps=512, temperature=1., cfg_scale=0.,
                       context_length=1024, eps=1e-5, dim=50257, device='cuda', repetition_penalty=1.2):
    """
    Shifted Diffusion Sampling - predicts the next token instead of the current token.
    Similar to diff_sample but implements a shifted token prediction strategy where
    each token position helps predict the next token.
    """
    batch_size = batch_size if prompt is None else prompt.shape[0]
    
    # Initialize sequence with mask tokens and start with BOS token at position 0
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    x[:, 0] = 50256  # BOS token
    positions = torch.arange(context_length, device=device).unsqueeze(0).expand(batch_size, -1)
    
    if prompt is not None:
        x[:, :prompt.shape[1]] = prompt.clone()

    # Track token frequencies to prevent repetition
    vocab_size = dim + 1  # Adding 1 to match the logits dimension (50258)
    token_frequencies = torch.zeros(vocab_size, device=device)
    
    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    
    for i in range(steps):
        # Create a shifted mask - we only predict tokens after visible tokens
        # We skip the first position (BOS token) in the masking logic
        current_tokens = x[:, :-1]  # All tokens except the last one
        next_tokens = x[:, 1:]  # All tokens except the first one
        
        # Find positions where the current token is not masked and the next token is masked
        # This ensures we only predict next tokens where we have context
        # valid_current = (current_tokens != dim)
        # mask_next = (next_tokens == dim)
        # mask_index = valid_current & mask_next
        mask_index = (next_tokens == dim)
        # Skip if no valid positions to predict
        if not mask_index.any():
            break
            
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if cfg_scale > 0.:
                # Create unconditional version by masking the prompt
                un_x = x.clone()
                if prompt is not None:
                    un_x[:, :prompt.shape[1]] = dim
                x_ = torch.cat([x, un_x], dim=0)
                
                # Get logits from model
                logits = model(x_, positions.repeat(2, 1))
                
                # Split into conditional and unconditional logits
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                
                # Get logits for valid positions (where we have context to predict next token)
                # We need these for the next position, so we use [:, :-1]
                logits = cond_logits[:, :-1][mask_index]
                un_logits = uncond_logits[:, :-1][mask_index]
            else:
                # Get logits for valid positions
                logits = model(x, positions)[:, :-1][mask_index]
                un_logits = None
                
            # Apply classifier-free guidance if enabled
            if cfg_scale > 0.:
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        # Apply repetition penalty to prevent repetitive outputs
        if i > 0:
            current_tokens = x[x != dim]
            if current_tokens.numel() > 0:
                unique_tokens, counts = torch.unique(current_tokens, return_counts=True)
                token_frequencies[unique_tokens] += counts.float()
                
                if token_frequencies.max() > 0:
                    norm_frequencies = token_frequencies / token_frequencies.max()
                    
                    penalty = torch.ones(logits.size(-1), device=device)
                    penalty[:vocab_size] = penalty[:vocab_size] + (repetition_penalty - 1.0) * norm_frequencies
                    
                    penalty = penalty.unsqueeze(0).expand(logits.size(0), -1)
                    logits = logits / penalty

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == 'original':
            # Calculate transfer probability
            p_transfer = 1 - s / t if i < steps - 1 else 1
            
            # Initialize next tokens with mask value
            next_x = torch.zeros_like(next_tokens[mask_index], device=device, dtype=torch.long) + dim
            
            # Determine which tokens to transfer based on probability
            transfer_index_t_s = torch.rand_like(next_x, device='cuda', dtype=torch.float64) < p_transfer
            
            # Add Gumbel noise and generate tokens for selected positions
            if transfer_index_t_s.any():
                logits_with_noise = add_gumbel_noise(logits[transfer_index_t_s], temperature=temperature)
                next_x[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
            
            # Update the sequence with new tokens
            next_tokens[mask_index] = next_x
            x[:, 1:] = next_tokens
            
        elif alg == 'acs':
            # Calculate transfer probability
            p_transfer = 1 - s / t if i < steps - 1 else 1
            eta = 0.015  # eta parameter for ACS (Adaptive Correction Sampler)
            
            # Initialize next tokens with mask value
            next_x = torch.zeros_like(next_tokens[mask_index], device=device, dtype=torch.long) + dim
            
            # Determine which tokens to transfer vs shuffle
            transfer_index_t_s = torch.rand_like(next_x, device='cuda', dtype=torch.float64) < p_transfer
            
            # Add adaptive correction shuffling for non-mask tokens
            if transfer_index_t_s.any():
                logits_with_noise = add_gumbel_noise(logits[transfer_index_t_s], temperature=temperature)
                next_x[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
            
            # Update the sequence with new tokens
            next_tokens[mask_index] = next_x
            x[:, 1:] = next_tokens
            

        elif alg == 'remask':
            # Fill all masked positions then remask some
            next_x = torch.zeros_like(next_tokens[mask_index], device=device, dtype=torch.long) + dim
            
            # Generate for all masked positions
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            next_x = torch.argmax(logits_with_noise, dim=-1)
            
            # Calculate remask probability
            p_transfer = 1 - s / t if i < steps - 1 else 1
            remask_prob = 1 - p_transfer
            
            # Remask some tokens
            remask_indices = torch.rand_like(next_x, device='cuda', dtype=torch.float64) < remask_prob
            next_x[remask_indices] = dim
            
            # Update the sequence
            next_tokens[mask_index] = next_x
            x[:, 1:] = next_tokens
            
        elif alg == 'remask_adhoc_shuffle':
            # Fill all masked positions then remask some with epsilon shuffling
            next_x = torch.zeros_like(next_tokens[mask_index], device=device, dtype=torch.long) + dim
            
            # Generate for all masked positions
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            next_x = torch.argmax(logits_with_noise, dim=-1)
            
            # Calculate remask probability
            p_transfer = 1 - s / t if i < steps - 1 else 1
            eta = 0.015
            remask_prob = 1 - p_transfer
            shuffle_prob = eta * remask_prob
            
            # Apply epsilon shuffling to some existing tokens
            existing_mask = (next_tokens != dim)
            if existing_mask.any():
                shuffle_indices = torch.rand_like(next_tokens, device='cuda', dtype=torch.float64) < shuffle_prob
                shuffle_mask = existing_mask & shuffle_indices
                if shuffle_mask.any():
                    # Get logits for existing positions (need to shift logic for this)
                    # For simplicity, apply same logits with noise
                    shuffle_logits = add_gumbel_noise(logits[:shuffle_mask.sum()], temperature=temperature)
                    next_tokens[shuffle_mask] = torch.argmax(shuffle_logits, dim=-1)
            
            # Remask some newly generated tokens
            remask_indices = torch.rand_like(next_x, device='cuda', dtype=torch.float64) < remask_prob
            next_x[remask_indices] = dim
            
            # Update the sequence
            next_tokens[mask_index] = next_x
            x[:, 1:] = next_tokens
            
        elif alg == 'remdm':
            # ReMDM algorithm implementation
            alpha_t = 1 - t.item()
            alpha_s = 1 - s.item()
            eta = 0.015
            
            # Compute sigma: controlling the amount of probability mass given to the mask token
            if alpha_t > 0:
                sigma = min(eta, (1 - alpha_s) / alpha_t)
            else:
                sigma = eta
                
            # Get the model's probability distribution
            p_x0 = torch.softmax(logits, dim=-1)
            
            # Mix the distribution: (1-sigma) from model, sigma from mask token
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., dim] = sigma
            
            # Sample from mixed distribution
            new_logits = torch.log(q_xs + 1e-8)
            new_logits = add_gumbel_noise(new_logits, temperature=temperature)
            next_x = torch.argmax(new_logits, dim=-1)
            
            # Update the sequence
            next_tokens[mask_index] = next_x
            x[:, 1:] = next_tokens
            
        else:
            raise NotImplementedError(f"Algorithm '{alg}' not supported in full_diff_shifted")

    return x
