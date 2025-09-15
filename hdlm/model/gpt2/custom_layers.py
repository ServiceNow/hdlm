import torch
import torch.nn as nn
import torch.nn.functional as F


def bias_dropout_fused_train(x, bias, gate, residual, dropout_prob):
    """
    This is a placeholder if you want to replicate DiT's behavior. Otherwise normal dropout is ok.
    """
    # We'll just do something simpler for demonstration
    if gate is not None:
        x = x * gate
    
    x = F.dropout(x + bias if bias is not None else x, p=dropout_prob, training=True)
    return x + residual

def bias_dropout_fused_inference(x, bias, gate, residual, dropout_prob):
    if gate is not None:
        x = x * gate
    # inference, no dropout
    return x + residual

def get_fused_dropout(training):
    return bias_dropout_fused_train if training else bias_dropout_fused_inference

class DiTLargeConditioningLayer(nn.Module):
    """
    Mimics DiT's approach of chunking the conditioning embedding by 6 and applying it
    twice in the block (MSA portion & MLP portion).
    """
    def __init__(self, hidden_size, conditioning_dim):
        super().__init__()
        self.linear = nn.Linear(conditioning_dim, 6 * hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states, sigma_embeds):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        sigma_embeds:  [batch_size, seq_len, conditioning_dim] or [batch_size, conditioning_dim]
        Returns the chunked parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        which can be used in the GPT2Block forward at two distinct points.
        """
        # If sigma_embeds is [batch, cond_dim], expand to [batch, seq_len, cond_dim]
        if sigma_embeds.dim() == 2:
            batch_size, cond_dim = sigma_embeds.shape
            seq_len = hidden_states.size(1)
            sigma_embeds = sigma_embeds.unsqueeze(1).expand(batch_size, seq_len, cond_dim)

        cond_out = self.linear(sigma_embeds)  # [b, seq_len, 6*hidden_size]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = cond_out.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


# Cross-attention Conditioning Layer
class ConditioningAttentionLayer(nn.Module):
    """
    Attention-Based Conditioning Layer.
    Modulates hidden states based on conditioning information using an attention mechanism.
    """
    def __init__(self, hidden_size, conditioning_dim, num_heads=4):
        super(ConditioningAttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(conditioning_dim, hidden_size)
        self.value = nn.Linear(conditioning_dim, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1)
        self.fc = nn.Linear(hidden_size, hidden_size * 2)  # For scale and bias
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, conditioning_embeds):
        """
        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            conditioning_embeds (torch.Tensor): [batch_size, cond_seq_len, conditioning_dim]
        
        Returns:
            torch.Tensor: Modulated hidden_states
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        cond_batch_size, cond_seq_len, cond_dim = conditioning_embeds.size()

        # Ensure batch sizes match
        assert batch_size == cond_batch_size, "Batch size of hidden_states and conditioning_embeds must match."

        # Project hidden states to query
        query = self.query(hidden_states)  # [batch_size, seq_len, hidden_size]
        query = query.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]

        # Project conditioning embeddings to key and value
        key = self.key(conditioning_embeds)  # [batch_size, cond_seq_len, hidden_size]
        value = self.value(conditioning_embeds)  # [batch_size, cond_seq_len, hidden_size]
        key = key.permute(1, 0, 2)  # [cond_seq_len, batch_size, hidden_size]
        value = value.permute(1, 0, 2)  # [cond_seq_len, batch_size, hidden_size]

        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(query, key, value)  # attn_output: [seq_len, batch_size, hidden_size]

        # Permute back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        # Compute scale and bias
        scale_bias = self.fc(attn_output)  # [batch_size, seq_len, hidden_size * 2]
        scale, bias = scale_bias.chunk(2, dim=-1)  # Each: [batch_size, seq_len, hidden_size]
        scale = torch.sigmoid(scale)  # Normalize scale to [0,1]
        bias = torch.tanh(bias)  # Normalize bias to [-1,1]

        # Apply modulation
        hidden_states = scale * hidden_states + bias  # [batch_size, seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + hidden_states)  # Residual connection and normalization

        return hidden_states

# CaSE Layer
class CaSELayer(nn.Module):
    """
    Contextual Squeeze-and-Excitation (CaSE) Layer.
    Enhances hidden states based on contextual information from sigma embeddings.
    """
    def __init__(self, hidden_size, conditioning_dim, reduction=16):
        super(CaSELayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Squeeze
        self.fc1 = nn.Linear(hidden_size + conditioning_dim, hidden_size // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // reduction, hidden_size * 2)  # Excitation: scale and bias

    def forward(self, hidden_states, sigma_embeds):
        """
        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            sigma_embeds (torch.Tensor): [batch_size, seq_len, conditioning_dim]
        
        Returns:
            torch.Tensor: Modulated hidden_states
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Permute for pooling: [batch_size, hidden_size, seq_len]
        hidden_states_permuted = hidden_states.permute(0, 2, 1)
        # Global average pooling: [batch_size, hidden_size, 1]
        pooled = self.global_avg_pool(hidden_states_permuted).squeeze(-1)  # [batch_size, hidden_size]
        
        # Concatenate with sigma embeddings (average over seq_len if necessary)
        # Assuming sigma_embeds are per position, we can average them
        sigma_pooled = sigma_embeds.mean(dim=1)  # [batch_size, conditioning_dim]
        combined = torch.cat([pooled, sigma_pooled], dim=-1)  # [batch_size, hidden_size + conditioning_dim]
        
        # Excitation
        excitation = self.fc1(combined)  # [batch_size, hidden_size // reduction]
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # [batch_size, hidden_size * 2]
        
        # Split into scale and bias
        scale, bias = excitation.chunk(2, dim=-1)  # Each: [batch_size, hidden_size]
        scale = torch.sigmoid(scale)  # Normalize scale to [0,1]
        # bias can be normalized or left as-is; here we'll apply tanh for stability
        bias = torch.tanh(bias)  # Normalize bias to [-1,1]
        
        # Reshape for broadcasting: [batch_size, 1, hidden_size]
        scale = scale.unsqueeze(1)
        bias = bias.unsqueeze(1)
        
        # Apply modulation
        hidden_states = scale * hidden_states + bias  # [batch_size, seq_len, hidden_size]
        
        return hidden_states
# CLN Layer
class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization (CLN) Layer.
    Adjusts normalization parameters based on conditioning embeddings.
    """
    def __init__(self, hidden_size, conditioning_dim):
        super(ConditionalLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.conditioning_dim = conditioning_dim
        
        # LayerNorm without learnable parameters since gamma and beta are conditional
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Linear layers to generate gamma and beta from conditioning embeddings
        self.gamma_proj = nn.Linear(conditioning_dim, hidden_size)
        self.beta_proj = nn.Linear(conditioning_dim, hidden_size)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.xavier_uniform_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        
    def forward(self, hidden_states, conditioning_embeds):
        """
        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            conditioning_embeds (torch.Tensor): [batch_size, seq_len, conditioning_dim]
        
        Returns:
            torch.Tensor: Normalized and modulated hidden_states
        """
        # Apply LayerNorm
        normalized = self.layer_norm(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Generate gamma and beta
        gamma = self.gamma_proj(conditioning_embeds)  # [batch_size, seq_len, hidden_size]
        beta = self.beta_proj(conditioning_embeds)    # [batch_size, seq_len, hidden_size]
        
        # Apply conditional scaling and shifting
        output = gamma * normalized + beta  # [batch_size, seq_len, hidden_size]
        return output
    
# FiLM Layer
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Applies feature-wise scaling and shifting based on conditioning embeddings.
    """
    def __init__(self, hidden_size, conditioning_dim):
        super(FiLMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.conditioning_dim = conditioning_dim
        
        # Linear layer to generate gamma and beta from conditioning embeddings
        self.film_generator = nn.Linear(conditioning_dim, 2 * hidden_size)
        
        # Initialize the FiLM generator
        nn.init.xavier_uniform_(self.film_generator.weight)
        if self.film_generator.bias is not None:
            nn.init.zeros_(self.film_generator.bias)
    
    def forward(self, hidden_states, sigma_embeds):
        """
        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            sigma_embeds (torch.Tensor): [batch_size, seq_len, conditioning_dim]
        
        Returns:
            torch.Tensor: Modulated hidden_states
        """
        # Generate gamma and beta
        gamma_beta = self.film_generator(sigma_embeds)  # [batch_size, seq_len, 2 * hidden_size]
        gamma, beta = gamma_beta.chunk(2, dim=-1)      # Each: [batch_size, seq_len, hidden_size]
        
        # Apply FiLM modulation
        modulated = gamma * hidden_states + beta     # [batch_size, seq_len, hidden_size]
        
        return modulated
