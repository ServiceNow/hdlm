from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union

from transformers import GPT2Config
from hdlm.model.gpt2.custom_layers import FiLMLayer, CaSELayer, ConditioningAttentionLayer, ConditionalLayerNorm, DiTLargeConditioningLayer, get_fused_dropout



class CustomGPT2Config(GPT2Config):
    def __init__(self, conditioning_method='film', **kwargs):
        super().__init__(**kwargs)
        assert conditioning_method in ['film', 'case', 'attention', 'cln', 'ditadain'], "Invalid conditioning method."
        self.conditioning_method = conditioning_method


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar or per-position timesteps (sigma) into vector representations.
    Supports both [batch_size] and [batch_size, seq_len] shapes for sigma.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
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
            torch.Tensor: Timestep embeddings. Shape: [batch_size, dim] or [batch_size, seq_len, dim].
        """
        if t.dim() == 1:
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding
        elif t.dim() == 2:
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
            args = t.float()[:, :, None] * freqs[None, None, :]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                zeros = torch.zeros_like(embedding[:, :, :1])
                embedding = torch.cat([embedding, zeros], dim=-1)
            return embedding
        else:
            raise ValueError(f"Unsupported tensor shape for t: {t.shape}")

    def forward(self, sigma):
        sigma_emb = self.timestep_embedding(sigma, self.frequency_embedding_size)
        sigma_emb = self.mlp(sigma_emb)
        return sigma_emb

class NewGPT2Attention(GPT2Attention):
    """
    Custom GPT-2 attention with support for non-causal attention masks.
    """
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights


########################################################################################
# Main GPT2Block that merges everything
########################################################################################
class CustomGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx
        
        # Replace GPT2Attention with custom
        self.attn = NewGPT2Attention(config, is_cross_attention=False, layer_idx=layer_idx)

        # We keep the original MLP, ln_1, ln_2, etc.
        # If user picks 'ditadain', we do the chunk-by-6 approach. Otherwise, we do the normal single-layer approach.
        self.conditioning_method = config.conditioning_method.lower()

        if self.conditioning_method == 'film':
            # Suppose we have FiLMLayer (not repeated here)
            self.conditioning_layer = FiLMLayer(hidden_size=config.n_embd, conditioning_dim=config.sigma_embed_dim)
        elif self.conditioning_method == 'case':
            self.conditioning_layer = CaSELayer(hidden_size=config.n_embd, conditioning_dim=config.sigma_embed_dim)
        elif self.conditioning_method == 'attention':
            self.conditioning_layer = ConditioningAttentionLayer(hidden_size=config.n_embd, conditioning_dim=config.sigma_embed_dim)
        elif self.conditioning_method == 'cln':
            self.conditioning_layer = ConditionalLayerNorm(hidden_size=config.n_embd, conditioning_dim=config.sigma_embed_dim)
        elif self.conditioning_method == 'ditadain':
            self.conditioning_layer = DiTLargeConditioningLayer(hidden_size=config.n_embd, conditioning_dim=config.sigma_embed_dim)
        else:
            raise ValueError(f"Unknown conditioning method: {self.conditioning_method}")

        # Using gating approach from DiT if chunk-by-6
        self.dropout_fn = get_fused_dropout(training=False)  # placeholder

    def forward(
        self,
        hidden_states: torch.Tensor,
        sigma_embeds: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # LN1
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        if sigma_embeds is not None:
            if self.conditioning_method == 'ditadain':
                # chunk by 6 approach
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.conditioning_layer(hidden_states, sigma_embeds)
                # MSA portion
                # apply shift_msa, scale_msa, gate_msa
                # hidden_states = hidden_states * (1 + scale_msa) + shift_msa
                # or a more direct approach:
                hidden_states = scale_msa * hidden_states + shift_msa
            else:
                # For the other methods (film, case, attention, cln, adain)
                hidden_states = self.conditioning_layer(hidden_states, sigma_embeds)

        # Self-attention
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # Residual connection after attention
        if self.conditioning_method == 'ditadain' and sigma_embeds is not None:
            # We apply gating to attention output if gate_msa is not None
            # In DiT code, they do something like: x = dropout_bias_scale_fn(...)
            # We'll keep it simpler: gate_msa is optional gating
            if gate_msa is not None:
                attn_output = attn_output * gate_msa
        hidden_states = attn_output + residual

        # Cross-attention (if config.add_cross_attention)
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError("crossattention not configured but encoder_hidden_states is provided!")
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]

        # LN2
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        # If chunk-by-6 in "ditadain" mode: apply second chunk to MLP
        if self.conditioning_method == 'ditadain' and sigma_embeds is not None:
            # shift_mlp, scale_mlp, gate_mlp
            hidden_states = scale_mlp * hidden_states + shift_mlp

        feed_forward_hidden_states = self.mlp(hidden_states)

        if self.conditioning_method == 'ditadain' and sigma_embeds is not None:
            # gating
            if gate_mlp is not None:
                feed_forward_hidden_states = feed_forward_hidden_states * gate_mlp

        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class CustomGPT2Model(GPT2Model):
    """
    GPT-2 model with support for custom attention and sigma embeddings.
    """
    def __init__(self, config):
        super().__init__(config)
        # Replace GPT2Blocks with custom blocks
        self.h =  nn.ModuleList([CustomGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        sigma_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        # Process attention mask
        if attention_mask is not None:
            # Invert the attention mask logic: True = mask, False = attend
            attention_mask = ~attention_mask  # Logical NOT to flip True/False
            attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.to(dtype=self.dtype) # fp16 compatibility
            # attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            attention_mask = (1.0 - attention_mask) * -1e4

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    sigma_embeds,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    sigma_embeds=sigma_embeds,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    """
    GPT-2 model with a custom transformer module for additional conditioning and attention.
    """
    def __init__(self, config, cond_dim=128, scale_by_sigma=True):
        super().__init__(config)
        self.cfg = config
        config.sigma_embed_dim = cond_dim
        self.scale_by_sigma = scale_by_sigma
    
        self.transformer = CustomGPT2Model(config)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * config.n_embd, bias=True)
        nn.init.kaiming_uniform_(self.adaLN_modulation.weight, a=math.sqrt(5))
        if self.adaLN_modulation.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.adaLN_modulation.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.adaLN_modulation.bias, -bound, bound)

        self.sigma_map = TimestepEmbedder(cond_dim)


    @classmethod
    def from_pretrained_with_customizations(cls, pretrained_model_name_or_path, tokenizer=None, from_scratch=False, conditioning_method='film'):
        """
        Loads a pretrained GPT-2 model and applies custom conditioning mechanisms.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model.
            tokenizer (Optional): Tokenizer associated with the model.
            from_scratch (bool): Whether to initialize the model from scratch.
            conditioning_method (str): One of 'film', 'case', 'attention', 'cln'.
        
        Returns:
            MockGPT2LMHeadModel: Customized GPT-2 model.
        """
        cls.trainable = ['sigma_map', 'adaLN_modulation', 'conditioning_layer']
        
        # Initialize custom configuration
        custom_config = CustomGPT2Config(
            conditioning_method=conditioning_method,
            **GPT2Config.from_pretrained(pretrained_model_name_or_path).to_dict()
        )
        
        if not from_scratch:
            # Load the pretrained GPT-2 model
            pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
            pretrained_config = pretrained_model.config

            # Initialize the custom model with the same configuration
            model = cls(custom_config)

            # Transfer weights for the existing transformer layers to the custom transformer
            model.transformer.load_state_dict(pretrained_model.transformer.state_dict(), strict=False)

            # Transfer weights for the language model head
            model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict(), strict=False)

            # Resize token embeddings if the tokenizer size has changed
            if tokenizer is not None and len(tokenizer) != pretrained_config.vocab_size:
                cls.trainable.extend(['transformer.wte', 'lm_head'])  # If applicable
                model.resize_token_embeddings(len(tokenizer))
        else:
            # Initialize from scratch without loading weights
            config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
            custom_config = CustomGPT2Config(
                conditioning_method=conditioning_method,
                **config.to_dict()
            )
            model = cls(custom_config)
            if tokenizer is not None and len(tokenizer) != config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
    
        return model
    
    def freeze_untrainable_layers(self):

        """
        Freezes all layers except those explicitly listed in `trainable`.
        """
        print('Freezing untrainable layers')
        for name, param in self.named_parameters():
            if not any(trainable_layer in name for trainable_layer in self.trainable):
                param.requires_grad = False
            else:
                print(f"Training {name}")
    
    def post_process_logits(self, logits, indices, settled_tokens=None):
        """
        Post-process the logits by zeroing out:
        - Logits corresponding to the current token (unsettled).
        - Logits corresponding to the next token in the input for AR mode.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            indices: [batch_size, seq_len]
            settled_tokens: [batch_size, seq_len], boolean mask indicating which tokens are settled

        Returns:
            Processed logits: [batch_size, seq_len, vocab_size]
        """
        # return logits
    
        if settled_tokens is not None:
            unsettled_mask = ~settled_tokens  # Invert settled tokens
        else:
            unsettled_mask = torch.ones_like(indices, dtype=torch.bool)

        # Ensure logits for the next token input are also zeroed out
        next_token_indices = indices.roll(shifts=-1, dims=1)  # Shift indices left for the next token
        scatter_mask = F.one_hot(next_token_indices, num_classes=logits.size(-1)).bool()
        scatter_mask = scatter_mask & unsettled_mask.unsqueeze(-1)

        # Apply the mask to zero out logits
        logits = logits.masked_fill(scatter_mask, -1e6)

        return logits

    def forward(self, input_ids, sigma, positions=None, attn_mask=None, settled_tokens=None, return_dict=False, **kwargs):
        # Pass positions to the CustomGPT2Model
        c = F.silu(self.sigma_map(sigma))
        transformer_outputs = self.transformer(input_ids=input_ids, sigma_embeds=c, position_ids=positions, attention_mask=attn_mask, **kwargs)
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        adaLN_output = self.adaLN_modulation(c)  # [batch_size, seq_len, 2 * hidden_size]
        shift, scale = adaLN_output.chunk(2, dim=-1)  # Each: [batch_size, seq_len, hidden_size]
        shift = torch.tanh(shift)
        scale = torch.sigmoid(scale)
        hidden_states = scale * hidden_states + shift
        lm_logits = self.lm_head(hidden_states)

        if self.scale_by_sigma:
            esigm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1
            ).log().to(lm_logits.dtype)  # Shape: [batch_size, seq_len]
            esigm1_log = esigm1_log[:, :, None]  # Shape: [batch_size, seq_len, 1]
            lm_logits = lm_logits - esigm1_log - torch.log(torch.tensor(lm_logits.size(-1) - 1, device=lm_logits.device, dtype=lm_logits.dtype))
        
        # TODO: check if this is beneficial to apply the post-processing
        lm_logits = self.post_process_logits(lm_logits, input_ids, settled_tokens=settled_tokens)

        if not return_dict:
            return lm_logits

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )