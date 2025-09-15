import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import hdlm.model.rotary as rotary
from hdlm.model.fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)   

import hdlm.model.rotary as rotary
import numpy as np
# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale,
    ).view(*x.shape[:-1], dim_out)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar or per-position timesteps into vector representations.
    Supports both [batch_size] and [batch_size, seq_len] shapes for sigma.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU() if silu else nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): Timesteps tensor. Shape: [batch_size] or [batch_size, seq_len].
            dim (int): The dimension of the output.
            max_period (int): Controls the minimum frequency of the embeddings.

        Returns:
            torch.Tensor: Positional embeddings. Shape: [batch_size, dim] or [batch_size, seq_len, dim].
        """
        if t.dim() == 1:
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding
        elif t.dim() == 2:
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t.float()[:, :, None] * freqs[None, None, :]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                zeros = torch.zeros_like(embedding[:, :, :1])
                embedding = torch.cat([embedding, zeros], dim=-1)
            return embedding
        else:
            raise ValueError(f"Unsupported tensor shape for t: {t.shape}")

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dim() == 3:
            batch_size, seq_len, dim = t_freq.shape
            t_freq_flat = t_freq.view(-1, dim)
            t_emb_flat = self.mlp(t_freq_flat)
            t_emb = t_emb_flat.view(batch_size, seq_len, -1)
        else:
            t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        cond_dim,
        mlp_ratio=4,
        dropout=0.1,
        use_sigma=False,
    ):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        if use_sigma:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()
        self.use_sigma = use_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def __forward_without_conditioning(self, x, rotary_cos_sin, attn_mask=None):
        bias_drop_scale_fn = self._get_bias_dropout_scale()
        
        # attention operation
        x_skip = x
        x = self.norm1(x)
        
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
        q, k , v = qkv.unbind(dim=2)
        d_k = q.shape[-1]
        
        attn_scores = torch.einsum('b i h d, b j h d -> b h i j', q, k) / math.sqrt(d_k)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        all_inf_mask = torch.isinf(attn_scores).all(dim=-1, keepdim=True)
        attn_scores = torch.where(all_inf_mask, torch.zeros_like(attn_scores), attn_scores)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout1(attn_weights)
        
        attn_output = torch.einsum('b h i j, b j h d -> b i h d', attn_weights, v)
        
        attn_output = rearrange(attn_output, 'b s h d -> b s (h d)')
        
        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_drop_scale_fn(self.attn_out(attn_output), None, scale, x_skip, self.dropout)
        
        x = bias_drop_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        
        return x

    def forward(self, x, rotary_cos_sin, c, attn_mask=None):
        """
        Minimal change here: we remove the flash_attn usage and do normal scaled-dot product attention.
        """

        if not self.use_sigma:
            return self.__forward_without_conditioning(x, rotary_cos_sin, attn_mask)

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Modulation parameters for conditioning on sigma
        (shift_msa, scale_msa, gate_msa, shift_mlp,
        scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=2)
        
        # Attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        # Project to Q,K,V
        qkv = self.attn_qkv(x)  # [b, s, 3 * dim]
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        # Apply rotary embedding to Q and K
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv =  rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        # qkv shape is now [b, s, 3, n_heads, d_head]
        # Split them out
        q, k, v = qkv.unbind(dim=2)  # each [b, s, n_heads, d_head]

        # Standard scaled dot-product attention
        d_k = q.shape[-1]
        # attn_scores: [b, n_heads, s, s]
        attn_scores = torch.einsum('b s h d, b t h d -> b h s t', q, k) / math.sqrt(d_k)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        all_inf_mask = torch.isinf(attn_scores).all(dim=-1, keepdim=True)  # [batch_size, n_heads, seq_len, 1]
        attn_scores = torch.where(all_inf_mask, torch.zeros_like(attn_scores), attn_scores)

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout1(attn_probs)  # dropout on attention probabilities

        # Weighted sum over V
        # out: [b, s, n_heads, d_head]
        out = torch.einsum('b h s t, b t h d -> b s h d', attn_probs, v)
        # Reshape back
        out = rearrange(out, 'b s h d -> b s (h d)')

        x = bias_dropout_scale_fn(
            self.attn_out(out), None, gate_msa, x_skip, self.dropout
        )
        x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, radd_enabled=False, gamma=1.0):
        """
        Embeds token indices into vector representations with optional RADD-based interpolation.

        Args:
            dim (int): Embedding dimension.
            vocab_dim (int): Vocabulary size.
            radd_enabled (bool): Flag to enable RADD-based interpolation.
            gamma (float): Scaling factor for RADD interpolation.
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        
        self.radd_enabled = radd_enabled
        self.gamma = gamma
        
        if self.radd_enabled:
            # Assuming the last token in the vocabulary is the [MASK] token
            self.mask_token_index = vocab_dim - 1

    def forward(self, x, sigma=None):
        """
        Forward pass for the EmbeddingLayer.

        Args:
            x (torch.Tensor): Token indices tensor of shape [batch_size, seq_len].
            sigma (torch.Tensor, optional): Noise tensor of shape [batch_size, seq_len].
                                            Required if radd_enabled is True.

        Returns:
            torch.Tensor: Embedded tokens tensor of shape [batch_size, seq_len, dim].
        """
        # Ensure all indices are within the valid range
        assert (x >= 0).all() and (x < self.embedding.shape[0]).all(), f"Index out of bounds: x={x}"
        
        # Retrieve embeddings
        embedded = self.embedding[x]  # Shape: [batch_size, seq_len, dim]
        
        if self.radd_enabled:
            assert sigma is not None, "sigma must be provided when radd_enabled is True"
            
            # Shape: [batch_size, seq_len]
            weights = torch.exp(-self.gamma * sigma)  # e^{-gamma * sigma}
            weights = weights.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
            weights_inv = 1 - weights  # Shape: [batch_size, seq_len, 1]
            
            # Shape: [dim]
            mask_emb = self.embedding[self.mask_token_index]  # [dim]
            
            # Shape: [1, 1, dim] -> [batch_size, seq_len, dim]
            mask_emb = mask_emb.unsqueeze(0).unsqueeze(0).expand_as(embedded)
            
            # Interpolate between original and mask embeddings
            embedded = weights * embedded + weights_inv * mask_emb
        
        return embedded  # Shape: [batch_size, seq_len, dim]

class DDitFinalLayerWithSigma(nn.Module):
    def __init__(
        self, hidden_size, out_channels, cond_dim
    ):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        shift = torch.clamp(shift, min=-1e1, max=1e1)
        scale = torch.clamp(scale, min=-1e1, max=1e1)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DDitFinalLayer(nn.Module):
  def __init__(
    self, hidden_size, out_channels, cond_dim
  ):
    super().__init__()

    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

  def forward(self, x, c):
    return self.linear(self.norm_final(x))


class DDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        # Default to True if model_type is epsilon_hybrid
        self.absorb = config.model_type == "epsilon_hybrid"
        # Then apply the previous logic
        if not self.absorb:
            self.absorb = config.graph.type == "absorb" or config.graph.type == "hybrid" or config.graph.type == "QGamma"
        
        try:
            vocab_size = config.tokenizer.tokens + (1 if self.absorb else 0)
        except:
            vocab_size = config.tokens + (1 if self.absorb else 0)
        
        self.vocab_size = vocab_size
        
        if self.config.model_type == "gamma_hybrid":
            embed_gamma = self.config.graph.gamma
            radd_enabled = True if self.config.graph.type in {'hybrid', 'QGamma'} and self.config.model.hybrid_sigma_embedding else False
        else:
            embed_gamma = self.config.training.epsilon
            radd_enabled = False

        self.vocab_embed = EmbeddingLayer(
            dim=config.model.hidden_size,
            vocab_dim=vocab_size,
            radd_enabled=radd_enabled,
            gamma=embed_gamma  
        )

        self.rotary_emb = rotary.Rotary(
            config.model.hidden_size // config.model.n_heads
        )

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                    use_sigma=config.model.transformer_sigma_conditioning
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size,
            vocab_size,
            config.model.cond_dim,
        )
        self.scale_by_sigma = config.model.scale_by_sigma
        self.post_process = config.model.post_process_logits
        if config.model.use_timestep_embedding:
            self.sigma_map = TimestepEmbedder(config.model.cond_dim)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference


# Epsilon-HDLM
class HDLM(DDIT):
    def __init__(self, config):
        super().__init__(config)
        self.mask_index = self.vocab_size - 1

    def post_process_logits(self, logits, indices, settled_tokens=None):
        """
        Post-process the logits by zeroing out:
        - (SAR mode) The current token's logit for unsettled tokens
        - (AR mode)  The *next* token's logit for unsettled tokens
        - For settled tokens, the logit of the [MASK] token (last index) => -1000.0

        Args:
        annealing: config.annealing object with its specified keys
        logits: [batch_size, seq_len, vocab_size] float tensor of raw model outputs
        indices: [batch_size, seq_len] long tensor of token indices
        settled_tokens: [batch_size, seq_len] bool mask (True => token is settled)

        Returns:
        Processed logits: [batch_size, seq_len, vocab_size]
        """
        annealing = self.config.annealing

        # ----------------------------------------------
        # 1) Determine which tokens are unsettled
        # ----------------------------------------------
        if self.training and annealing.efficient and settled_tokens is None:
            # In your design, half of the seq is settled, half unsettled
            seq_len = logits.size(1)
            half_seq_len = seq_len // 2

            # unsettled => second half
            unsettled_mask = torch.zeros_like(indices, dtype=torch.bool)
            unsettled_mask[:, half_seq_len:] = True

            # settled => first half
            settled_tokens = ~unsettled_mask

        elif settled_tokens is not None:
            # invert the settled mask => unsettled
            unsettled_mask = ~settled_tokens
        else:
            # by default, everything is unsettled if no 'settled_tokens' given
            unsettled_mask = torch.ones_like(indices, dtype=torch.bool)

        # ----------------------------------------------
        # 2) "AR"  => zero out *next token* logits
        # ----------------------------------------------
        if annealing.sampling_method in{"AR", "AR_BAD", "AR_SCRATCH"}:
            if self.training and annealing.efficient:
                first_half_next_token_indices = indices[:, :half_seq_len].roll(shifts=-1, dims=1)[:, :-1]
                second_half_next_token_indices = indices[:, half_seq_len:].roll(shifts=-1, dims=1)[:, :-1]
                logits[:, :half_seq_len-1, :] = torch.scatter(logits[:, :half_seq_len-1, :], -1, first_half_next_token_indices[..., None], torch.zeros_like(logits[..., :1]))
                logits[:, half_seq_len:-1, :] = torch.scatter(logits[:, half_seq_len:-1, :], -1, second_half_next_token_indices[..., None], torch.zeros_like(logits[..., :1]))
            else:
                next_token_indices = indices.roll(shifts=-1, dims=1)[:, :-1]
                logits[:, :-1, :] = torch.scatter(logits[:, :-1, :], -1, next_token_indices[..., None], torch.zeros_like(logits[..., :1]))
        
        if settled_tokens is not None:
            # index of [MASK] is last in vocab
            mask_token_index = logits.size(-1) - 1

            # Only apply to positions where settled_tokens[b, t] == True
            next_settled_tokens = settled_tokens.roll(shifts=-1, dims=1)[:,:-1]
            b_idx, t_idx = next_settled_tokens.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                logits[b_idx, t_idx, mask_token_index] = -1000.0

        return logits

    def forward(self, xt, positions, attn_mask=None, settled_tokens=None):
        """Forward pass of the denoising model.

        Args:
        xt: int torch.Tensor with shape
            (batch_size, diffusion_model_input_length), token ids.
        sigma: float torch.Tensor with shape
            (batch_size).

        Returns:
        log probability with shape
            (batch_size, diffusion_model_input_length, vocab_size)
        """
        x = self.vocab_embed(xt, None)
        rotary_cos_sin = self.rotary_emb(x, positions)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, None, attn_mask=attn_mask)
            x = self.output_layer(x, None)

        # post-process logits
        if self.post_process:
            x = self.post_process_logits(x, xt, settled_tokens=settled_tokens)
        
        return x


# Gamma-HDLM
class HDLM_Gamma(DDIT):
    def __init__(self, config):
        super().__init__(config)
        self.mask_index = self.vocab_size - 1

    def post_process_logits(self, logits, indices, settled_tokens=None):
        """
        Post-process the logits by zeroing out:
        - (SAR mode) The current token's logit for unsettled tokens
        - (AR mode)  The *next* token's logit for unsettled tokens
        - For settled tokens, the logit of the [MASK] token (last index) => -1000.0

        Args:
        annealing: config.annealing object with its specified keys
        logits: [batch_size, seq_len, vocab_size] float tensor of raw model outputs
        indices: [batch_size, seq_len] long tensor of token indices
        settled_tokens: [batch_size, seq_len] bool mask (True => token is settled)

        Returns:
        Processed logits: [batch_size, seq_len, vocab_size]
        """
        annealing = self.config.annealing

        # ----------------------------------------------
        # 1) Determine which tokens are unsettled
        # ----------------------------------------------
        if self.training and annealing.efficient and settled_tokens is None:
            # In your design, half of the seq is settled, half unsettled
            seq_len = logits.size(1)
            half_seq_len = seq_len // 2

            # unsettled => second half
            unsettled_mask = torch.zeros_like(indices, dtype=torch.bool)
            unsettled_mask[:, half_seq_len:] = True

            # settled => first half
            settled_tokens = ~unsettled_mask

        elif settled_tokens is not None:
            # invert the settled mask => unsettled
            unsettled_mask = ~settled_tokens
        else:
            # by default, everything is unsettled if no 'settled_tokens' given
            unsettled_mask = torch.ones_like(indices, dtype=torch.bool)

        # ----------------------------------------------
        # 2) "AR"  => zero out *next token* logits
        # ----------------------------------------------
        if annealing.sampling_method in{"AR", "AR_BAD", "AR_SCRATCH"}:
            if self.training and annealing.efficient:
                first_half_next_token_indices = indices[:, :half_seq_len].roll(shifts=-1, dims=1)[:, :-1]
                second_half_next_token_indices = indices[:, half_seq_len:].roll(shifts=-1, dims=1)[:, :-1]
                logits[:, :half_seq_len-1, :] = torch.scatter(logits[:, :half_seq_len-1, :], -1, first_half_next_token_indices[..., None], torch.zeros_like(logits[..., :1]))
                logits[:, half_seq_len:-1, :] = torch.scatter(logits[:, half_seq_len:-1, :], -1, second_half_next_token_indices[..., None], torch.zeros_like(logits[..., :1]))
            else:
                next_token_indices = indices.roll(shifts=-1, dims=1)[:, :-1]
                logits[:, :-1, :] = torch.scatter(logits[:, :-1, :], -1, next_token_indices[..., None], torch.zeros_like(logits[..., :1]))
        
        if settled_tokens is not None:
            # index of [MASK] is last in vocab
            mask_token_index = logits.size(-1) - 1

            # Only apply to positions where settled_tokens[b, t] == True
            next_settled_tokens = settled_tokens.roll(shifts=-1, dims=1)[:,:-1]
            b_idx, t_idx = next_settled_tokens.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                logits[b_idx, t_idx, mask_token_index] = -1000.0

        return logits

    def forward(self, indices, sigma, positions=None, attn_mask=None, settled_tokens=None):
        """
        Args:
            indices: [batch_size, seq_len]
            sigma: [batch_size, seq_len]
            attn_mask: [batch_size, seq_len, seq_len]
        """
        x = self.vocab_embed(indices, sigma)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x, positions=positions)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c, attn_mask=attn_mask)
            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            esigm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1
            ).log().to(x.dtype)  # Shape: [batch_size, seq_len]
            esigm1_log = esigm1_log[:, :, None]  # Shape: [batch_size, seq_len, 1]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)
        
        # post-process logits
        if self.post_process:
            x = self.post_process_logits(x, indices, settled_tokens=settled_tokens)

        return x
    