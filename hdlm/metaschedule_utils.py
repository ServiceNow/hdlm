# this file contains the utility functions for generating attention masks and work with metaschedules
import torch

def generate_attention_mask(
    batch_size: int,
    seq_len: int,
    settled_end_idx: int,  # inclusive
    active_end_idx: int,   # inclusive
    context_attention_type: str = "causal",  # for settled tokens
    block_attention_type: str = "full",       # for active tokens
    block_size: int = 0,
    causal: bool = False,
    post_active_context: bool = False,  # NEW flag for post-active tokens
):
    """
    Generates a [batch_size, seq_len, seq_len] boolean attention mask 
    (True => masked / cannot attend, False => attendable).

    Layout:
      - Settled tokens: indices [0 .. settled_end_idx]
      - Active tokens:  indices [settled_end_idx+1 .. active_end_idx]
      - Inactive or post-active tokens: indices [active_end_idx+1 .. seq_len-1]
        -> If `post_active_context` is True, this region is treated like the settled region,
           with attention determined by context_attention_type.
    
    Args:
      batch_size: int
      seq_len: int
      settled_end_idx: int (X), inclusive end of settled region
      active_end_idx: int  (Y), inclusive end of active region
      context_attention_type: str in {"full", "causal", "block_causal"}
      block_attention_type:   str in {"full", "causal"}
      post_active_context:    bool, if True then tokens after the active block will use the same 
                              attention logic as the context (settled) tokens.
    
    Returns:
      attn_mask of shape [batch_size, seq_len, seq_len], dtype=bool
        True  => masked (cannot attend)
        False => attendable
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if context_attention_type not in {"full", "causal", "block_causal"}:
        raise ValueError(
            f"Invalid context_attention_type={context_attention_type}. "
            f"Choose from {{'full', 'causal', 'block_causal'}}."
        )
    if block_attention_type not in {"full", "causal"}:
        raise ValueError(
            f"Invalid block_attention_type={block_attention_type}. "
            f"Choose from {{'full', 'causal'}}."
        )

    # Create an all-masked (True) matrix initially
    attn_mask_2d = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    settled_slice  = slice(0, settled_end_idx + 1)        # inclusive of settled_end_idx
    if causal:
        if settled_end_idx >= 0:
            active_slice = slice(settled_end_idx, active_end_idx + 1)
        else:
            active_slice = slice(settled_end_idx + 1, active_end_idx + 1)
    else:
        active_slice   = slice(settled_end_idx + 1, active_end_idx + 1)

    inactive_slice = slice(active_end_idx + 1, seq_len)

    # Region lengths
    num_settled = (settled_end_idx + 1) if settled_end_idx >= 0 else 0
    num_active  = max(0, (active_end_idx + 1) - (settled_end_idx + 1))
    if causal and settled_end_idx >= 0:
        num_active += 1
    # inactive would be seq_len - (active_end_idx + 1)

    # -------------------------------------------------------
    # 1) Mask inactive tokens (fully masked rows & columns)
    # -------------------------------------------------------
    attn_mask_2d[inactive_slice, :] = True
    attn_mask_2d[:, inactive_slice] = True

    # -------------------------------------------------------
    # 2) Handle Settled tokens (context region)
    #    [0 .. settled_end_idx]
    # -------------------------------------------------------
    if num_settled > 0:
        if context_attention_type == "full":
            # Unmask all pairs within the settled region
            attn_mask_2d[settled_slice, settled_slice] = False

        elif context_attention_type == "causal":
            sub_len = num_settled
            sub_causal = torch.triu(
                torch.ones(sub_len, sub_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_mask_2d[settled_slice, settled_slice] = sub_causal

        elif context_attention_type == "block_causal" and num_settled > 0 and block_size > 0:
            sub_len = num_settled
            block_id = torch.arange(sub_len, device=device) // block_size
            sub_mask = attn_mask_2d[:sub_len, :sub_len]
            block_i = block_id.unsqueeze(1)  # shape (sub_len, 1)
            block_j = block_id.unsqueeze(0)  # shape (1, sub_len)
            mask_of_block = (block_j <= block_i)
            sub_mask[mask_of_block] = False

    # -------------------------------------------------------
    # 3) Handle Active tokens (block region)
    #    [settled_end_idx+1 .. active_end_idx]
    # -------------------------------------------------------
    if num_active > 0:
        # (a) Internal logic among active tokens
        if block_attention_type == "full":
            attn_mask_2d[active_slice, active_slice] = False
        else:  # "causal"
            sub_len = num_active
            sub_causal = torch.triu(
                torch.ones(sub_len, sub_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_mask_2d[active_slice, active_slice] = sub_causal
            if causal:
                for i in range(sub_len - 1):  # Exclude the last token since it has no "next"
                    current_idx = active_slice.start + i
                    next_idx = current_idx + 1
                    attn_mask_2d[current_idx, next_idx] = False

        # (b) Typically, active tokens can see settled tokens => unmask columns in settled region
        attn_mask_2d[active_slice, settled_slice] = False

    # -------------------------------------------------------
    # 4) Optionally handle Post-active tokens (treat like context tokens)
    #    [active_end_idx+1 .. seq_len-1]
    # -------------------------------------------------------
    if post_active_context:
        num_post_active = seq_len - (active_end_idx + 1)
        if num_post_active > 0:
            post_active_slice = slice(active_end_idx + 1, seq_len)
            if context_attention_type == "full":
                attn_mask_2d[post_active_slice, post_active_slice] = False

            elif context_attention_type == "causal":
                sub_len = num_post_active
                sub_causal = torch.triu(
                    torch.ones(sub_len, sub_len, dtype=torch.bool, device=device),
                    diagonal=1
                )
                attn_mask_2d[post_active_slice, post_active_slice] = sub_causal

            elif context_attention_type == "block_causal" and block_size > 0:
                sub_len = num_post_active
                block_id = torch.arange(sub_len, device=device) // block_size
                sub_mask = attn_mask_2d[post_active_slice, post_active_slice]
                block_i = block_id.unsqueeze(1)
                block_j = block_id.unsqueeze(0)
                mask_of_block = (block_j <= block_i)
                sub_mask[mask_of_block] = False

    attn_mask = attn_mask_2d.unsqueeze(0).expand(batch_size, -1, -1)
    return attn_mask

def compute_t_and_attn(
    batch_size, 
    seq_len, 
    metaschedule, 
    current_step,
    context_attention_type = "causal",
    block_attention_type = "full", 
    device = "cuda", 
    eps = 1e-4,
    post_active_context: bool = False,
):
    """
    An optimized version that:
      1) Uses generate_attention_mask_faster_strict (contiguous logic)
      2) Extracts settled_end_idx & active_end_idx
      3) Minimizes memory usage & Python overhead
      4) Returns the exact same values and in the same order as before.
    """
    # 1) metaschedule + levels
    ms_step = metaschedule(current_step)
    levels = torch.tensor(ms_step(stop=seq_len), device=device, dtype=torch.long)
    # Could potentially store 'levels' in a smaller dtype if needed, e.g. int16,
    # but that depends on the range of values in metaschedule.

    # 2) Identify contiguous "settled" region [0..num_settled-1]
    num_settled = ms_step.num_settled
    if num_settled > 0:
        settled_end_idx = num_settled - 1
    else:
        settled_end_idx = -1

    # 3) Build a worthless mask (like is_level_num_levels)
    #    worthless means levels == metaschedule.worthless
    is_worthless = (levels == metaschedule.worthless)

    # 4) We don't need 'positions' anymore: "settled" is [0..settled_end_idx].
    #    Let's define is_settled, is_active in contiguous logic:
    is_settled = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if num_settled > 0:
        is_settled[:num_settled] = True  # [0..num_settled-1]

    is_active = ~(is_settled | is_worthless)  # everything else

    # 5) Determine active_end_idx
    active_positions = is_active.nonzero(as_tuple=True)[0]
    if len(active_positions) > 0:
        active_end_idx = active_positions[-1].item()
    else:
        # no active tokens => fallback to settled_end_idx
        active_end_idx = settled_end_idx

    # 6) Initialize t_start/t_end
    t = torch.zeros(seq_len, device=device)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (A) Settled tokens => [0..num_settled-1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if num_settled > 0:
        t[:num_settled] = eps
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (B) Worthless tokens => level == worthless
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_worthless.any():
        t[is_worthless] = 1.0 - eps
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (C) Active tokens => everything else
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    active_eps = 10 * eps
    step = (1 - 2 * active_eps) / (metaschedule.worthless - 1)
    if is_active.any():
        # We only convert a sub-slice to float => less memory overhead
        lv_active = levels[is_active].float()
        t[is_active] = active_eps + lv_active * step
    t.clamp_(min=eps, max=1.0 - eps)

    # 7) tokens_to_denoise
    tokens_to_denoise = t == active_eps

    # 8) Build the attention mask using the new function
    #    "context" = settled [0..settled_end_idx], 
    #    "block"   = active  [settled_end_idx+1..active_end_idx].
    attn_mask = generate_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        settled_end_idx=settled_end_idx,
        active_end_idx=active_end_idx,
        context_attention_type=context_attention_type,  # choices: "causal" / "full" / "block_causal"
        block_attention_type=block_attention_type,       # choices: "full" / "causal"
        block_size = metaschedule.width if context_attention_type == "block_causal" else 0,
        post_active_context = post_active_context,
    ).to(device)

    # 9) Build weight_mask
    weight_mask = torch.zeros(seq_len, device=device, dtype=torch.float)
    # settled => small weight
    if num_settled > 0:
        weight_mask[:num_settled] = 0.05
    # worthless => 0 weight
    weight_mask[is_worthless] = 0.001
    # active => weight=1
    weight_mask[is_active] = 1.0

    return (
        t, 
        attn_mask, 
        tokens_to_denoise, 
        is_active,      
        is_settled,    
        weight_mask
    )


def compute_t_and_attn_autoregressive(
        batch_size, 
        seq_len, 
        metaschedule, 
        current_step,
        context_attention_type = "causal",
        block_attention_type = "full", 
        device = "cuda", 
        eps = 1e-4,
        post_active_context: bool = False,
    ):
    ms_step = metaschedule(current_step)
    levels = ms_step(stop=seq_len - 1)  # [seq_len]
    levels_list = list(levels)
    levels_list.insert(0, 0)
    levels = torch.tensor(levels_list, dtype=torch.long, device=device)
    positions = torch.arange(seq_len, device=device)

    levels_float = levels.float()

    t = torch.zeros(seq_len, device=device)

    is_level_0 = positions < ms_step.num_settled
    is_level_num_levels = levels == metaschedule.worthless
    active_positions_mask = ~is_level_0 & ~is_level_num_levels
    t[is_level_0] = eps

    # 2) Identify contiguous "settled" region [0..num_settled-1]
    num_settled = ms_step.num_settled
    if num_settled > 0:
        settled_end_idx = num_settled - 1
    else:
        settled_end_idx = -1

    active_positions = active_positions_mask.nonzero(as_tuple=True)[0]
    if len(active_positions) > 0:
        active_end_idx = active_positions[-1].item()
    else:
        # no active tokens => fallback to settled_end_idx
        active_end_idx = settled_end_idx

    # Level num_levels (k == num_levels)
    if is_level_num_levels.any():
        t[is_level_num_levels] = 1.0 - eps

    active_eps = 10 * eps
    step = (1 - 2 * active_eps) / (metaschedule.worthless - 1)
    if active_positions_mask.any():
        # We only convert a sub-slice to float => less memory overhead
        lv_active = levels_float[active_positions_mask]
        t[active_positions_mask] = active_eps + lv_active * step
    t.clamp_(min=eps, max=1.0 - eps)
        
    # Ensure first active token has t_start and t_end set to eps
    active_indices = torch.nonzero(active_positions_mask, as_tuple=False).squeeze(1)
    if active_indices.numel() > 0:
        first_active_index = active_indices[0]
        t[first_active_index] = eps

    tokens_to_denoise = t == active_eps
    if settled_end_idx >= 0:
        settled_end_idx += 1
    attn_mask = generate_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        settled_end_idx=settled_end_idx,
        active_end_idx=active_end_idx,
        context_attention_type=context_attention_type,
        block_attention_type=block_attention_type,
        block_size = metaschedule.width + 1 if context_attention_type == "block_causal" else 0,
        causal=True,
        post_active_context=post_active_context 
    ).to(device)

    weight_mask = torch.ones(seq_len, device=device, dtype=torch.float) * 0.001

    weight_mask[active_positions_mask] = 1.0
    
    if active_indices.numel() > 0:
        weight_mask[first_active_index] = 0.001
    weight_mask[is_level_num_levels] = 0.001

    return t, attn_mask, tokens_to_denoise, active_positions_mask, positions <= ms_step.num_settled, weight_mask


def compute_t_and_attn_efficient_training(metaschedule, ms_step, device, eps):
    seq_len = len(ms_step)
    ms_step = torch.tensor(ms_step, dtype=torch.long, device=device)
    t = torch.zeros(2 * seq_len, device=device)
    t[:seq_len] = eps
    active_eps = 10 * eps
    step = (1 - 2 * active_eps) / (metaschedule.worthless - 1)
    t[seq_len:] = active_eps + ms_step.float() * step    
    t = torch.clamp(t, min=eps, max=1 - eps)
    return t


def generate_attention_mask_efficient_training(
    bs: int,
    seq_len: int,
    N: int,
    context_attention_type: str = "causal",
    block_attention_type: str = "full"
):
    """
    Generate an attention mask of shape [bs, 2*seq_len, 2*seq_len].
    
    1) The 'context' (first half: [0 : seq_len]) attention is determined by
       'context_attention_type', which can be:
         - "full"         => full attention among [0..seq_len-1]
         - "causal"       => standard token-level causal attention
         - "block_causal" => block-level causal: the first half is split into N blocks,
                            each block has full attention internally, and can attend
                            all prior blocks in the first half
    
    2) The 'block' (second half: [seq_len : 2*seq_len]) attention is determined by
       'block_attention_type', which can be:
         - "full"   => each bucket has full internal attention
         - "causal" => each bucket is internally causal
    
    3) For the second half, each bucket also attends back to the first half
       in a positional manner: bucket i can attend up to [0 : i * bucket_size] in the first half.

    Returns:
        attn_mask (torch.Tensor): shape [bs, 2*seq_len, 2*seq_len], dtype=torch.bool
            True  => masked (cannot attend)
            False => attendable
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_len = 2 * seq_len

    # Initialize everything to True => "masked"
    attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)

    # ----------------------------------------------------------------------
    # 1) FIRST HALF: context_attention_type
    # ----------------------------------------------------------------------
    if context_attention_type == "full":
        # FULL: unmask all positions in [0:seq_len, 0:seq_len]
        attn_mask[:seq_len, :seq_len] = False

    elif context_attention_type == "causal":
        # CAUSAL: token-level standard causal attention
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        # causal_mask[i,j] = True if j>i => masked
        attn_mask[:seq_len, :seq_len] = causal_mask

    elif context_attention_type == "block_causal":
        # BLOCK-CAUSAL: split the first half into N blocks
        block_size = seq_len // N
        for i in range(N):
            block_start = i * block_size
            block_end   = (i + 1) * block_size  # non-inclusive
            # full attention within the block
            attn_mask[block_start:block_end, block_start:block_end] = False
            # can attend to everything before block_start (previous blocks)
            if block_start > 0:
                attn_mask[block_start:block_end, 0:block_start] = False
    else:
        raise ValueError(
            f"Invalid context_attention_type={context_attention_type}. "
            f"Choose from {{'full', 'causal', 'block_causal'}}."
        )

    # ----------------------------------------------------------------------
    # 2) SECOND HALF: block_attention_type
    # ----------------------------------------------------------------------
    bucket_size = seq_len // N

    for i in range(N):
        bucket_start_pos = seq_len + i * bucket_size
        bucket_end_pos   = seq_len + (i + 1) * bucket_size
        length_of_bucket = bucket_end_pos - bucket_start_pos  # should be bucket_size

        # For each bucket, define internal attention logic
        if block_attention_type == "full":
            # Full attention inside the bucket
            attn_mask[bucket_start_pos:bucket_end_pos, bucket_start_pos:bucket_end_pos] = False

        elif block_attention_type == "causal":
            # Causal attention inside the bucket
            # Create a smaller causal mask for this bucket
            sub_causal_mask = torch.triu(
                torch.ones(length_of_bucket, length_of_bucket, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_mask[
                bucket_start_pos:bucket_end_pos,
                bucket_start_pos:bucket_end_pos
            ] = sub_causal_mask
        else:
            raise ValueError(
                f"Invalid block_attention_type={block_attention_type}. "
                f"Choose from {{'full', 'causal'}}."
            )

        # "Attend back" to the first half up to [0 : true_positions_start]
        true_positions_start = i * bucket_size
        if true_positions_start > 0:
            attn_mask[bucket_start_pos:bucket_end_pos, 0:true_positions_start] = False

    # ----------------------------------------------------------------------
    # 3) Expand to batch dimension
    # ----------------------------------------------------------------------
    attn_mask = attn_mask.unsqueeze(0).expand(bs, -1, -1)

    return attn_mask

def generate_attention_mask_block_efficient_training_autoregressive(
    bs: int,
    seq_len: int,
    N: int,
    context_attention_type: str = "causal",  # {"full", "causal", "block_causal"}
    block_attention_type: str = "full"       # {"full", "causal"}
):
    """
    Generates an autoregressive-style attention mask of shape [bs, 2*seq_len, 2*seq_len].

    * First half = [0..seq_len-1] => 'context', which can be:
        - "full": no masking among these tokens
        - "causal": standard token-level causal
        - "block_causal": chunk-based causal in the context region

    * Second half = [seq_len..2*seq_len-1] => divided into N buckets, each of size bucket_size = seq_len//N
        - "full": tokens in each bucket see each other fully
        - "causal": tokens in each bucket have token-level causal
      Additionally, each bucket i can attend back to [0..(i*bucket_size)-1] in the *first half*.

    Returns:
        torch.Tensor bool mask of shape [bs, 2*seq_len, 2*seq_len], 
            True  => masked (cannot attend)
            False => attendable
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_len = 2 * seq_len

    # ----------------------------------------------------------------------
    # 0) Initialize everything to True => "masked"
    # ----------------------------------------------------------------------
    attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)

    # ----------------------------------------------------------------------
    # 1) FIRST HALF: context_attention_type
    #    Indices => [0..seq_len-1]
    # ----------------------------------------------------------------------
    if context_attention_type == "full":
        # "full" => no mask among [0..seq_len-1]
        attn_mask[:seq_len, :seq_len] = False

    elif context_attention_type == "causal":
        # "causal" => token-level triangular
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        # causal_mask[i,j]=True => masked if j>i
        attn_mask[:seq_len, :seq_len] = causal_mask

    elif context_attention_type == "block_causal":
        # "block_causal" => chunk-based causal in [0..seq_len]
        block_size = seq_len // N
        # We iterate over blocks in the context region
        #   block i => [i*block_size : (i+1)*block_size]
        #   tokens within the block => full attention
        #   tokens in block i can attend [0.. i*block_size)
        #   cannot attend future blocks
        for i in range(N):
            block_start = i * block_size
            block_end   = (i + 1) * block_size
            # clamp the last block
            if block_end > seq_len:
                block_end = seq_len

            if block_start >= seq_len:
                break

            # full attention within the block
            attn_mask[block_start:block_end, block_start:block_end] = False
            # attend to all previous context
            if block_start > 0:
                attn_mask[block_start:block_end, 0:block_start] = False

    else:
        raise ValueError(
            f"Invalid context_attention_type={context_attention_type}. "
            f"Choose from {{'full','causal','block_causal'}}."
        )

    # ----------------------------------------------------------------------
    # 2) SECOND HALF: block_attention_type
    #    Indices => [seq_len..2*seq_len-1]
    #
    #    We partition [seq_len..2*seq_len-1] into N buckets, each of size bucket_size.
    #    For bucket i:
    #      - "full" => fully attend inside [bucket_start:bucket_end]
    #      - "causal" => token-level causal
    #      - can also attend up to [0..(i*bucket_size)-1] in the first half
    # ----------------------------------------------------------------------
    bucket_size = seq_len // N

    for i in range(N):
        # bucket range in the second half
        bucket_start_pos = seq_len + i * bucket_size
        bucket_end_pos   = seq_len + (i + 1) * bucket_size + 1
        # clamp if needed
        if bucket_end_pos > (2*seq_len):
            bucket_end_pos = 2 * seq_len

        if bucket_start_pos >= 2*seq_len:
            break

        length_of_bucket = bucket_end_pos - bucket_start_pos
        if length_of_bucket <= 0:
            continue

        # (A) Internal logic for each bucket
        if block_attention_type == "full":
            # fully unmask the bucket
            attn_mask[bucket_start_pos:bucket_end_pos, bucket_start_pos:bucket_end_pos] = False

        elif block_attention_type == "causal":
            # token-level causal
            sub_causal_mask = torch.triu(
                torch.ones(length_of_bucket, length_of_bucket, dtype=torch.bool, device=device),
                diagonal=1
            )
            # apply sub_causal_mask
            next_token_mask = torch.eye(length_of_bucket, dtype=torch.bool, device=device)
            next_token_mask = torch.roll(next_token_mask, shifts=1, dims=1)  # Shift diagonally
            next_token_mask[:, 0] = False  # Last token doesn't have a "next"
            attn_mask[
                bucket_start_pos:bucket_end_pos,
                bucket_start_pos:bucket_end_pos
            ] = sub_causal_mask & ~next_token_mask


        else:
            raise ValueError(
                f"Invalid block_attention_type={block_attention_type}. "
                f"Choose from {{'full','causal'}}."
            )

        # (B) "Attend back" to the first half => [0.. i*bucket_size)
        true_positions_start = i * bucket_size
        if true_positions_start > 0:
            # unmask columns in [0..true_positions_start]
            attn_mask[bucket_start_pos+1:bucket_end_pos, 0:true_positions_start] = False

        # By default, tokens in bucket i CANNOT see:
        #   - [true_positions_start..seq_len) in the first half
        #   - other buckets in the second half, including future ones

    # ----------------------------------------------------------------------
    # 3) Expand to batch dimension
    # ----------------------------------------------------------------------
    attn_mask = attn_mask.unsqueeze(0).expand(bs, -1, -1)
    return attn_mask


import torch

def generate_attention_mask_simple_efficient_training(
    bs: int,
    seq_len: int,
    steps: list[tuple[int,int]],
    context_attention_type: str = "causal",
    block_attention_type: str = "full"
) -> torch.Tensor:
    """
    Generate an attention mask [bs, 2*seq_len, 2*seq_len] for "simple_annealing efficient" training.
    
    Arguments:
      bs: batch size
      seq_len: length of the context + length for second half
      steps: a list of (start, end) pairs from get_simple_annealing_efficient(...).
             For example, if get_simple_annealing_efficient yields:
                 steps = [(0,6), (6,14), (14,22), (22,30), (30,32)]
             then we have 5 blocks in the second half.
      context_attention_type: one of {"full", "causal", "block_causal"}, controlling
                              how the first half [0..seq_len-1] attends among itself.
      block_attention_type: one of {"full", "causal"}, controlling how each block
                            in the second half attends within itself.
    
    Returns:
      attn_mask: torch.BoolTensor of shape [bs, 2*seq_len, 2*seq_len].
        - True   => masked (cannot attend)
        - False  => attendable
      The first half is set according to 'context_attention_type'.
      The second half is partitioned according to 'steps', placed contiguously,
      with each block's internal attention = block_attention_type, 
      and each block i can attend up to [0.. start_i) in the first half.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_len = 2 * seq_len

    # ----------------------------------------------------------------------
    # 0) Initialize everything to True => "masked"
    # ----------------------------------------------------------------------
    attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)

    # ----------------------------------------------------------------------
    # 1) FIRST HALF [0..seq_len-1]: context_attention_type
    # ----------------------------------------------------------------------
    if context_attention_type == "full":
        # No masking among [0..seq_len-1]
        attn_mask[:seq_len, :seq_len] = False

    elif context_attention_type == "causal":
        # Standard token-level causal
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        # causal_mask[i,j] = True => masked if j>i
        attn_mask[:seq_len, :seq_len] = causal_mask

    elif context_attention_type == "block_causal":
        # --------------------------------------------------------------
        # Use the SAME blocks as in the second half (i.e., 'steps'),
        # but only the portion that lies in [0..seq_len).
        # For each sub-block (start_idx,end_idx), clamp to first-half range
        # and unmask fully among that subrange => tokens [start:end] see each other.
        # --------------------------------------------------------------
        for (start_idx, end_idx) in steps:
            # clamp to the first half
            block_start = start_idx
            block_end   = min(end_idx, seq_len)
            if block_end > block_start >= 0:
                # Fully unmask [block_start : block_end] within itself
                attn_mask[block_start:block_end, :block_end] = False

    else:
        raise ValueError(
            f"Invalid context_attention_type={context_attention_type}. "
            f"Must be one of {{'full','causal','block_causal'}}."
        )

    # ----------------------------------------------------------------------
    # 2) SECOND HALF [seq_len..2*seq_len-1]: use steps to define blocks
    #
    #    We'll place each block contiguously in the second half,
    #    in the same order as 'steps'. For block i with (start, end):
    #       - length = end - start
    #       - place it in [block_start : block_end] in the second half
    #       - internal attention = 'full' or 'causal'
    #       - can attend up to [0.. start) in the first half.
    #
    #    If the sum of all block lengths < seq_len, we leave leftover as masked
    #    (or you can fill worthless tokens?). 
    # ----------------------------------------------------------------------
    current_second_half_pos = seq_len  # where we place the next block
    for (start_idx, end_idx) in steps:
        length = end_idx - start_idx
        if length <= 0:
            continue  # skip empty block
        block_start_pos = current_second_half_pos
        block_end_pos   = block_start_pos + length

        if block_end_pos > 2*seq_len:
            # clamp if it overshoots
            block_end_pos = 2*seq_len

        # (A) Internal logic
        if block_attention_type == "full":
            # unmask inside
            attn_mask[block_start_pos:block_end_pos, block_start_pos:block_end_pos] = False
        elif block_attention_type == "causal":
            # token-level causal
            sub_len = block_end_pos - block_start_pos
            sub_causal = torch.triu(
                torch.ones(sub_len, sub_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_mask[block_start_pos:block_end_pos, block_start_pos:block_end_pos] = sub_causal
        else:
            raise ValueError(
                f"Invalid block_attention_type={block_attention_type}. "
                f"Choose from {{'full','causal'}}."
            )

        # (B) "Attend back" to the first half => [0.. start_idx)
        # i.e. unmask columns [0..start_idx) for rows [block_start_pos..block_end_pos)
        if start_idx > 0:
            attn_mask[block_start_pos:block_end_pos, 0:start_idx] = False

        # Move current_second_half_pos forward
        current_second_half_pos = block_end_pos
        if current_second_half_pos >= 2*seq_len:
            break

    # everything else in the second half (if any leftover space) remains masked

    # ----------------------------------------------------------------------
    # 3) Expand to batch dimension
    # ----------------------------------------------------------------------
    attn_mask = attn_mask.unsqueeze(0).expand(bs, -1, -1)

    return attn_mask


def create_causal_attention(batch_size, sequence_len, end, device):
    """
    Create a causal attention mask for the given sequence length.
    False for positions that are allowed to attend, True otherwise.

    Example:
    create_causal_attention(5, 3) will return:
    tensor([[False,  True,  True,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False, False,  True],
            [False, False, False, False, False]])
    """
    mask = torch.ones(sequence_len, sequence_len, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask | (torch.arange(sequence_len, device=device) > end).unsqueeze(0)
    return mask.unsqueeze(0).expand(batch_size, -1, -1)