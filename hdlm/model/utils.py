import torch
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import math

def get_block_weighting(seq_len: int, width: int, device: torch.device = None) -> torch.Tensor:
    """
    Creates a weight matrix of shape [seq_len] where the sequence is divided into blocks of size `width`.
    Each block is assigned a weight proportional to its block index, with the last block having a weight of 1.
    The weights increase linearly across the blocks.

    Args:
        seq_len (int): Length of the sequence.
        width (int): Width of each block.
        device (torch.device, optional): Device on which to create the tensor. Defaults to CPU.

    Returns:
        torch.Tensor: A tensor of shape [seq_len] with the assigned weights.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate the number of blocks (ceiling division)
    num_blocks = math.ceil(seq_len / width)
    
    if num_blocks == 1:
        # If the entire sequence fits within one block, assign weight 1 to all positions
        weights = torch.ones(seq_len, device=device)
        return weights

    # Generate position indices [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device)  # Shape: [seq_len]

    # Determine block index for each position
    block_indices = positions // width  # Shape: [seq_len]

    # Calculate weights: block_index / (num_blocks - 1), ensuring the last block has weight 1
    weights = block_indices.float() / (num_blocks - 1)

    # Clamp weights to a maximum of 1 to handle any potential floating-point inaccuracies
    weights = torch.clamp(weights, max=1.0)

    return weights  # Shape: [seq_len]

@dataclass
class ModelOutput:
    _logits: Optional[torch.FloatTensor] = field(default=None, init=False, repr=False)
    _log_softmax: Optional[torch.FloatTensor] = field(default=None, init=False, repr=False)
    _attentions: Optional[List[torch.FloatTensor]] = field(default=None, init=False, repr=False)

    compute_logits_fn: Optional[callable] = field(default=None, repr=False)
    compute_log_softmax_fn: Optional[callable] = field(default=None, repr=False)

    @property
    def logits(self):
        if self._logits is None and self.compute_logits_fn is not None:
            self._logits = self.compute_logits_fn()
        return self._logits

    @property
    def log_softmax(self):
        if self._log_softmax is None and self.compute_log_softmax_fn is not None:
            self._log_softmax = self.compute_log_softmax_fn()
        return self._log_softmax

def scale_by_sigma(x, sigma):
    """
    Scales the input tensor x by subtracting the log of (exp(sigma) - 1).

    Supports:
        - x: [batch_size, seq_len, vocab_dim], sigma: [batch_size]
        - x: [batch_size, seq_len], sigma: [batch_size, seq_len]

    Args:
        x (torch.Tensor): Input tensor to be scaled.
            - Shape [batch_size, seq_len, vocab_dim] or [batch_size, seq_len]
        sigma (torch.Tensor): Tensor containing sigma values.
            - Shape [batch_size] or [batch_size, seq_len]

    Returns:
        torch.Tensor: Scaled tensor with the same shape as x.
    """
    # Step 1: Compute esigm1 based on sigma using the condition sigma < 0.5
    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),      # Computes exp(sigma) - 1 for sigma < 0.5
        torch.exp(sigma) - 1     # Computes exp(sigma) - 1 for sigma >= 0.5
    )
    
    # Step 2: Take the logarithm of esigm1
    # Ensure numerical stability by adding a small epsilon if necessary
    epsilon = 1e-10
    esigm1 = torch.clamp(esigm1, min=epsilon)  # Prevent log(0)
    esigm1_log = esigm1.log().to(x.dtype)       # Ensure dtype consistency
    
    # Step 3: Dynamically reshape esigm1_log to enable broadcasting
    if x.dim() == 3 and sigma.dim() == 1:
        # Case 1: x = [batch_size, seq_len, vocab_dim], sigma = [batch_size]
        # Reshape esigm1_log to [batch_size, 1, 1]
        esigm1_log = esigm1_log.view(-1, 1, 1)
    elif x.dim() == 3 and sigma.dim() == 2:
        # Case 2: x = [batch_size, seq_len, vocab_dim], sigma = [batch_size, seq_len]
        # Reshape esigm1_log to [batch_size, seq_len, 1]
        esigm1_log = esigm1_log.view(-1, x.size(1), 1)
    elif x.dim() == 2 and sigma.dim() == 2:
        # Case 3: x = [batch_size, seq_len], sigma = [batch_size, seq_len]
        # Reshape esigm1_log to [batch_size, seq_len]
        esigm1_log = esigm1_log.view(-1, 1)
    elif x.dim() == 2 and sigma.dim() == 1:
        # Case 4: x = [batch_size, seq_len], sigma = [batch_size]
        # Reshape esigm1_log to [batch_size, 1]
        esigm1_log = esigm1_log.view(-1, 1)
    else:
        raise ValueError(f"Unsupported shape combination: x.dim()={x.dim()}, sigma.dim()={sigma.dim()}")
    
    # Step 4: Subtract esigm1_log from x with broadcasting
    scaled_x = x - esigm1_log  # Broadcasting occurs here
    
    return scaled_x

def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma, positions=None, attn_mask=None, **kwargs):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma, positions=positions, attn_mask=attn_mask, **kwargs)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)
    if torch.cuda.get_device_properties(0).major >= 8:
        # bfloat16 is only supported on A100 and newer GPUs
        data_type = torch.bfloat16
    else:
        data_type = torch.float16
    with torch.amp.autocast("cuda", dtype=data_type):
        def score_fn(x, sigma, positions=None, attn_mask=None, **kwargs):
            log_score = model_fn(x, sigma, positions=positions, attn_mask=attn_mask, **kwargs)
            return log_score

    return score_fn