from math import ceil
from typing import Sequence, Optional, NamedTuple
from enum import Enum

import math
import numpy as np
from torch import Tensor

class ScheduleType(Enum):
    SIMPLE_ANNEALING = "simple_annealing"
    BLOCK_ANNEALING = "block_annealing"
    HYBRID_ANNEALING = "hybrid_annealing"
    AUTOREGRESSIVE = "autoregressive"
    OTHER = "other"


class MetaScheduleStep(NamedTuple):
    # The part of the step that isn't settled (i.e., 0) nor worthless.
    relevant: tuple[int, ...]
    # Number of (extra) 0 appearing before the relevant part.
    num_settled: int
    # Value indicating "full noise" (after the relevant part)
    worthless: int

    def __call__(
            self, 
            stop: Optional[int]=None, 
            start: int = 0
        ) -> tuple[int, ...]:
        beginning = (0,)*self.num_settled + self.relevant
        if stop is None:
            return beginning[start:]
        elif stop <= len(beginning):
            return beginning[start:stop]
        else:
            all = beginning + (self.worthless,)*(stop - len(beginning))
            return all[start:]


class MetaSchedule():
    def __init__(
        self, 
        steps: Sequence[Sequence[int]],
        # define acceptable types for the type argument below (e.g., "simple_annealing", "block_annealing") create Enum for this
        type: str = ScheduleType.OTHER.value,
        *,
        bootstrap_steps: Optional[Sequence[Sequence[int]]] = None,
        bootstrap_shifts: Optional[Sequence[int]] = None,
        auto_bootstrap: Optional[str] = None,
        width: Optional[int] = None,
    ):
        steps = tuple(tuple(step) for step in steps)
        self.steps = steps
        self.worthless, self.all_tau_present = self._get_max_tau(steps)
        self.shifts = self._get_shifts(steps)
        assert type in ScheduleType._value2member_map_, f"Invalid type '{type}'. Must be one of {[t.value for t in ScheduleType]}"

        # At least one step must yield a clean token on the left
        assert any(step[0] == 0 for step in steps)
        self.type = ScheduleType(type)

        # auto_bootstrap not implemented yet
        if auto_bootstrap is not None:
            raise NotImplementedError("Implement once we have a proof of concept.")
        if bootstrap_steps is None:
            raise NotImplementedError("The `bootstrap_steps` argument is currently mandatory, as no `auto_bootstrap` is implemented yet.")
        
        self.bootstrap_steps = tuple(tuple(step) for step in bootstrap_steps)

        if bootstrap_shifts is None:
            raise NotImplementedError("The `bootstrap_shifts` argument is currently mandatory, as no `auto_bootstrap` is implemented yet.")
        self.bootstrap_shifts = tuple(bootstrap_shifts)
        assert len(bootstrap_shifts) == len(bootstrap_steps) + 1
        # if type is block_annealing, width must be provided
        if type == ScheduleType.BLOCK_ANNEALING.value:
            assert width is not None, "width must be provided for block_annealing type"
        self.width = width

    def __call__(self, t: int, condition_padding: int = 0) -> MetaScheduleStep:
        assert t >= 0
        if t == 0:
            # Only return padding (if any)
            relevant = ()
            shift = condition_padding
        elif t - 1 < len(self.bootstrap_steps):
            # We're still in the bootstrap phase
            relevant = self.bootstrap_steps[t-1]
            shift = condition_padding + self.bootstrap_shifts[t - 1]
        else:
            # We're in the main loopy phase
            phase = (t - 1 - len(self.bootstrap_steps)) % len(self.steps)
            cycle = (t - 1 - len(self.bootstrap_steps)) // len(self.steps)
            relevant = self.steps[phase]
            shift = (
                condition_padding + self.bootstrap_shifts[-1] 
                + self.shifts[phase] + self.shifts[-1]*cycle
            )
        return MetaScheduleStep(
            relevant=relevant, 
            num_settled=shift, 
            worthless=self.worthless
        )

    @staticmethod
    def _get_shifts(steps: Sequence[Sequence[int]]) -> tuple[int]:
        shifts = [0]
        for step in steps:
            # Count the number of zeros at the beginning of `step`
            shifts.append(shifts[-1])
            for x in step:
                if x == 0:
                    shifts[-1] += 1
                else:
                    break
        return tuple(shifts)

    def max_step(self, seq_len: int) -> int:
        # Sample valid current_step (similar to inference)
        if self.type.value == 'simple_annealing':
            # worthless = tau
            tau = self.worthless
            w = self.width

            # Brute-force from 1..some safe upper bound
            # A typical bound is seq_len + tau (that often suffices),
            # but if your partial-slice logic can push concurrency, we add + w for safety.
            # You can adjust bigger if needed, but this usually covers it.
            limit = seq_len + tau + w

            for t in range(1, limit + 1):
                arr = self(t)(stop=seq_len)
                # Check if all tokens are zero
                if all(x == 0 for x in arr):
                    return t + 1

            # Fallback if we never found all-zero (theoretically shouldn't happen).
            return limit
        elif self.type.value == 'block_annealing':
            max_step = self.worthless * math.ceil(seq_len / self.width) + 1
        else:
            raise ValueError(f'Metaschedule type {self.type} not supported for now.')
        return max_step

    @staticmethod
    def _get_max_tau(steps: Sequence[Sequence[int]]) -> tuple[int, bool]:
        seen = frozenset(x for step in steps for x in step)
        assert all(x >= 0 for x in seen)
        max_tau = max(seen) + 1
        all_tau_present = (len(seen) == max_tau)
        return max_tau, all_tau_present

def make_simple_annealing(width: int, tau: int) -> MetaSchedule:
    """
    Produce a 'simple annealing' MetaSchedule in which the relevant block
    (of length `width`) is linearly spaced from 0 up to tau-1.
    The worthless noise level is then automatically determined to be `tau`.
    
    - If tau >= width and tau % width == 0, that implies perfectly integer increments of size tau/width.
    - If tau >= width but not divisible by width, or if tau < width, we still produce an evenly spaced schedule,
      but values may be repeated or “jump” by 1 in some places due to rounding.
    
    Example:
      width=4, tau=4  -> main_step = (0, 1, 2, 3); worthless=4
      width=4, tau=8  -> main_step ≈ (0, 2, 5, 7); worthless=8
      width=4, tau=2  -> main_step ≈ (0, 0, 1, 1); worthless=2
    """
    if width <= 0:
        raise ValueError("`width` must be positive.")
    if tau <= 0:
        raise ValueError("`tau` must be positive.")
    
    # Special case: if width == 1, there's no real 'diagonal'. We'll just produce (tau-1,)
    # so that worthless = tau.
    if width == 1:
        main_step = (tau - 1,)
    else:
        # We want to linearly space from 0 up to (tau-1) inclusive, across `width` slots.
        # i.e. main_step[i] = round( (i / (width-1)) * (tau - 1) )
        main_step = []
        for i in range(width):
            # fraction of the way from 0..(tau - 1)
            val = round((i * (tau - 1)) / (width - 1))
            main_step.append(val)
        main_step = tuple(main_step)
    
    # -- Build the bootstrap steps (partial slices from the end, exactly like your original code) --
    # For example, if main_step = (0,1,2,3), we get bootstrap_steps = ((3,), (2,3), (1,2,3))
    # which yields the “diagonal” progression in the initial steps.
    bootstrap_steps = tuple(
        main_step[-t:]   # tail-slice from the end
        for t in range(1, len(main_step))  # 1..(width-1)
    )
    
    # We'll keep the shifts all zero to match the original approach
    bootstrap_shifts = (0,) * len(main_step)
    
    return MetaSchedule(
        steps=(main_step,),
        bootstrap_steps=bootstrap_steps,
        bootstrap_shifts=bootstrap_shifts,
        type=ScheduleType.SIMPLE_ANNEALING.value,
        width=width
    )


def make_block_annealing(width: int, tau: int) -> MetaSchedule:
    if tau is None:
        tau = width
    steps = [[0] * width] + [[i] * width for i in range(tau - 1, 0, -1)]
    bootstrap_steps = [[i] * width for i in range(tau - 1, 0, -1)]
    bootstrap_shifts = [0] * tau
    ms = MetaSchedule(
        steps,
        bootstrap_steps=bootstrap_steps,
        bootstrap_shifts=bootstrap_shifts,
        type=ScheduleType.BLOCK_ANNEALING.value,
        width=width
    )
    return ms
class HybridMetaScheduleStep(NamedTuple):
    relevant: tuple[int, ...]
    num_settled: int
    worthless: int

    def __call__(self, stop: Optional[int] = None, start: int = 0) -> tuple[int, ...]:
        """
        Return the portion [start:stop] of 'relevant'. 
        'num_settled' and 'worthless' remain the same.
        """
        arr = self.relevant
        if stop is None:
            return arr[start:]
        else:
            return arr[start:stop]


class HybridMetaSchedule:
    """
    A specialized schedule object that:
      - Partitions seq_len into blocks of size at most `width`.
      - For each block i, does (tau + width - 1) partial-diagonal steps 
        that bring that block from (tau-1)->0, left to right.
      - Once a block is fully finished, it remains at 0 in subsequent steps.
      - 'num_settled' is the number of tokens in fully-finished blocks. 
        (i.e. block i is only considered 'settled' after its final step.)
      - 'worthless' = tau for each step.
      - 'relevant' is a full seq_len array of noise values.
    """

    def __init__(
        self, 
        steps: Sequence[Sequence[int]], 
        tau: int, 
        block_sizes: Sequence[int]
    ):
        """
        Args:
          steps: a list of length N, each a seq_len-sized tuple (the schedule at each step).
          tau:   worthless noise level.
          block_sizes: the size (# tokens) of each block. 
                       E.g. [width, width, ..., last_block_len].
                       We'll use this to compute num_settled quickly.
        """
        self.steps = tuple(tuple(s) for s in steps)
        self.tau = tau
        self.worthless = tau
        self.width = block_sizes[0]  # width of the first block
        
        # E.g. if seq_len=16, width=8 => block_sizes=[8,8].
        # If seq_len=18, width=8 => block_sizes=[8,8,2].
        self.block_sizes = tuple(block_sizes)

        # We'll define how many steps each block contributes:
        # block i => (tau + width - 1) steps in the stored schedule
        # (even if the block is smaller, we still produce that many steps 
        #  so the final sub-step sets that partial block to zero).
        self.steps_per_block = tau + max(0, len(block_sizes) > 0 and max(block_sizes)) - 1  
        # Actually, more consistent to keep the same (tau+width-1) for each block 
        # if that matches your generation code. We'll store that explicitly:
        self.steps_per_block = tau + block_sizes[0] - 1  

        # Precompute total steps
        self.n_blocks = len(block_sizes)
        self.total_steps = self.n_blocks * self.steps_per_block

        # For convenience, also build a prefix-sum of block_sizes for quick summation
        self.block_prefix_sum = [0]
        for sz in self.block_sizes:
            self.block_prefix_sum.append(self.block_prefix_sum[-1] + sz)
        # block_prefix_sum[i] = sum of block_sizes[:i]

    def __call__(self, t: int, condition_padding: int=0) -> HybridMetaScheduleStep:
        """
        Return the HybridMetaScheduleStep for step index t (1-based).
        If t is beyond the total, clamp to the final step.
        """
        if t < 1:
            t = 1
        if t > self.total_steps:
            t = self.total_steps

        arr = self.steps[t-1]
        
        # Figure out how many blocks are fully settled at step t.
        # Each block i uses steps in the range:
        #    [ i*(steps_per_block)+1 .. (i+1)*(steps_per_block] 
        # The final sub-step of block i is step (i+1)*steps_per_block,
        # at which point that block is fully 0 => "settled."
        #
        # Let block_in_progress = floor((t-1)/steps_per_block).
        # The final step for block_in_progress is (block_in_progress+1)*steps_per_block.
        # If t == that final step, block_in_progress is settled.
        # Otherwise, only earlier blocks are settled.
        
        steps_per_block = self.steps_per_block
        block_in_progress = (t-1) // steps_per_block
        step_in_block = (t-1) % steps_per_block
        
        # If step_in_block == steps_per_block - 1, 
        # we've just finished block_in_progress => it is settled.
        # So fully settled blocks = block_in_progress if we haven't hit the final sub-step,
        # or block_in_progress+1 if we have.
        if step_in_block == steps_per_block:
            # We just finished that block
            settled_blocks = block_in_progress + 1
        else:
            # We are in the middle of block_in_progress
            settled_blocks = block_in_progress
        
        # clamp the settled_blocks in case we are at or beyond the last block
        if settled_blocks > self.n_blocks:
            settled_blocks = self.n_blocks
        
        # num_settled = sum of block_sizes for all those fully settled blocks
        num_settled = self.block_prefix_sum[settled_blocks]

        return HybridMetaScheduleStep(
            relevant=arr,
            num_settled=num_settled,
            worthless=self.worthless
        )

    def max_step(self, seq_len: int) -> int:
        """
        Return total_steps + 1, so if user asks beyond that, we clamp.
        (Or we could return total_steps exactly.)
        """
        return self.total_steps + 1


def make_hybrid_annealing(width: int, tau: int, seq_len: int) -> HybridMetaSchedule:
    """
    Builds a 'hybrid' schedule by:
      1) Splitting seq_len into blocks of size `width` (last block may be smaller).
      2) For each block, do (tau+width-1) steps that partial-diagonal from (tau-1)->0.
         The earlier blocks stay 0 once done, future blocks remain worthless(tau).
      3) Return a HybridMetaSchedule that also reports 'num_settled' = sum of 
         token-counts in fully-finished blocks.
    """
    if width <= 0:
        raise ValueError("`width` must be positive.")
    if tau <= 0:
        raise ValueError("`tau` must be positive.")
    if seq_len <= 0:
        raise ValueError("`seq_len` must be positive.")

    # Determine block sizes
    n_blocks = math.ceil(seq_len / width)
    block_sizes = []
    for i in range(n_blocks):
        start_idx = i*width
        end_idx = min(start_idx+width, seq_len)
        block_len = end_idx - start_idx
        block_sizes.append(block_len)

    # We'll produce (tau+width-1) steps per block in partial-diagonal style:
    steps = []
    curr_arr = [tau]*seq_len  # start worthless in all tokens

    for block_i, block_len in enumerate(block_sizes):
        start_idx = block_i * width
        # We won't bother to recalc end_idx because block_len is known
        # do partial diagonal for this block
        for s in range(tau + width - 1):
            new_arr = curr_arr[:]
            # partial diagonal update inside [start_idx..start_idx+block_len-1]
            for j in range(block_len):
                offset2 = max(0, s - j)
                val = max(0, (tau - 1) - offset2)
                new_arr[start_idx + j] = val
            steps.append(tuple(new_arr))
            curr_arr = new_arr

    return HybridMetaSchedule(steps, tau=tau, block_sizes=block_sizes)

def get_block_annealing_efficient(ms: MetaSchedule, seq_len: int, sampling=True) -> Tensor:
    tau = ms.worthless # shows the width of the block
    width = ms.width
    min_step, max_step =  1, ms.max_step(seq_len)

    steps = []
    states = []
    counter = 0
    # sample width steps and stitch them together
    # sampling be like: 1st from [1, width], 2nd from [width+1, 2*width], etc. don't forget to sample not just a simple for loop
    if sampling:
        for i in range(min_step, max_step, tau):            
            min_sub_step, max_sub_step = i, min(i+tau, max_step)
            step = np.random.randint(min_sub_step, max_sub_step)
            states.extend(ms(step)(stop=seq_len)[counter * width:(counter + 1) * width])
            steps.append(step)
            counter += 1
    else:
        for i in range(min_step, max_step, tau):
            min_sub_step, max_sub_step = i, min(i+tau, max_step)
            if i == min_step:
                step = np.random.randint(min_sub_step, max_sub_step)
            states.extend(ms(step+counter * tau)(stop=seq_len)[counter * width:(counter + 1) * width])
            steps.append(step)
            counter += 1
    return states, steps


def get_simple_annealing_efficient(ms, seq_len: int):
    """
    An efficient approach for simple_annealing coverage:

    1) Pick one warm-up step in [1..ms.width]. Gather that slice.
    2) Then define consecutive chunks of size 'ms.width' until seq_len.
    3) For each chunk [start..start+width), try to find a single step that
       covers it entirely. If none, pick the first step that partially covers
       from chunk_start onward.  Then proceed to fill the remainder of the chunk
       with subsequent steps.  Repeat until the chunk is fully covered or we
       exhaust all steps.  Then move to the next chunk.

    Returns:
      states: List[int] of concatenated noise values covering [0..seq_len) 
              in disjoint slices, with no skipping.
      steps:  List[Tuple[int,int]] of (start,end) for each slice appended to states.
    """

    if ms.width is None:
        raise ValueError("MetaSchedule must have 'width' set for simple_annealing.")
    width = ms.width

    min_step, max_step = 1, ms.max_step(seq_len)

    # --------------------- (A) Warm-up Step ---------------------
    warmup_step = np.random.randint(1, width + 1)
    w_info = ms(warmup_step)
    w_start = w_info.num_settled
    w_end   = min(w_start + len(w_info.relevant), seq_len)

    steps = [(w_start, w_end)]
    full_arr = w_info(stop=seq_len)
    states = list(full_arr[w_start:w_end])

    coverage = w_end      # coverage so far
    curr_step = warmup_step

    # Helper: build a list of all potential steps for easy access
    all_steps = []
    for s in range(warmup_step+1, max_step+1):
        info_s = ms(s)
        s_start = info_s.num_settled
        s_end   = s_start + len(info_s.relevant)
        all_steps.append( (s, s_start, s_end, info_s) )

    # index in all_steps
    idx = 0

    # ---------------------- (B) Main Chunks of width ----------------------
    while coverage < seq_len:
        chunk_start = coverage
        chunk_end   = min(coverage + width, seq_len)
        if chunk_start >= chunk_end:
            break

        # Try to see if there's a single step that fully covers chunk_start..chunk_end
        single_found = False
        # We'll look ahead in all_steps from idx.. to see if s_start<=chunk_start and s_end>=chunk_end
        for j in range(idx, len(all_steps)):
            s_id, s_start, s_end, s_info = all_steps[j]
            if s_start <= chunk_start and s_end >= chunk_end:
                # Found a single step that covers the entire chunk
                arr_s = s_info(stop=seq_len)
                chunk_vals = arr_s[chunk_start:chunk_end]
                states.extend(chunk_vals)
                steps.append((chunk_start, chunk_end))
                coverage = chunk_end
                curr_step = s_id
                idx = j  # we might reuse or skip steps after j
                single_found = True
                break

        if single_found:
            # move on to next chunk
            continue

        # If we didn't find a single step that fully covers the chunk,
        # we attempt partial coverage in multiple sub-slices:
        chunk_coverage = chunk_start
        while chunk_coverage < chunk_end and idx < len(all_steps):
            # Look at the step all_steps[idx]
            s_id, s_start, s_end, s_info = all_steps[idx]

            # If this step doesn't even begin on or before chunk_coverage,
            # we move to the next step
            if s_start > chunk_coverage:
                idx += 1
                continue

            # If the step ends <= chunk_coverage, it's already behind
            if s_end <= chunk_coverage:
                idx += 1
                continue

            # Now we have s_start <= chunk_coverage < s_end
            # => we can gather some portion from chunk_coverage.. min(s_end, chunk_end)
            partial_end = min(s_end, chunk_end)
            arr_s = s_info(stop=seq_len)
            piece = arr_s[chunk_coverage:partial_end]
            states.extend(piece)
            steps.append((chunk_coverage, partial_end))

            coverage = partial_end
            chunk_coverage = partial_end
            curr_step = s_id

            # If we've filled the chunk up to chunk_end, break
            if chunk_coverage >= chunk_end:
                break

            # Otherwise, try next step in all_steps
            idx += 1

        # If chunk_coverage < chunk_end => we couldn't finish the chunk
        if chunk_coverage < chunk_end:
            # no more steps can fill the rest => stop or fill worthless
            # For now, we'll just stop:
            break

    return states, steps