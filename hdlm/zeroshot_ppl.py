import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

class ZeroShot_calculator:
    def __init__(self, model, device, nll_type='mc', mask_id=50257, cfg=0, mc_num=10, batch_size=8, padding=False, sampling_eps=1e-4, 
                mode='aligned', disable_tqdm=False, metaschedule=None, annealing=None):
        self.nll_type = nll_type
        self.model = model
        self.mask_id = mask_id
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self.mc_num = mc_num
        self.padding = padding
        self.sampling_eps = sampling_eps
        self.mode = mode
        self.disable_tqdm = disable_tqdm
        self.metaschedule = metaschedule
        self.annealing = annealing

    def _forward_process(self, batch):
        """
        Apply forward process to create noisy inputs.
        
        Args:
            batch: Input tensor of shape [batch_size, seq_len]
            mode: Either 'aligned' (mask current token) or 'shifted' (mask next token)
            
        Returns:
            noisy_batch: Batch with tokens masked
            p_mask: Mask probabilities
        """
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)
        
        if self.mode == 'aligned':
            # Standard masking - mask tokens randomly
            mask_indices = torch.rand((b, l), device=batch.device) < p_mask
            noisy_batch = torch.where(mask_indices, self.mask_id, batch)
        else:  # 'shifted'
            noisy_batch = batch.clone()
            
            # 1. Create masking for next tokens (t+1)
            # We'll mask positions 1 to l (shifting the mask right)
            mask_prob_shifted = torch.zeros((b, l), device=batch.device)
            mask_prob_shifted[:, :-1] = p_mask[:, 1:]  # Shift mask probabilities left
            
            # 2. Apply masking
            mask_indices = torch.rand((b, l), device=batch.device) < mask_prob_shifted
            noisy_batch = torch.where(mask_indices, self.mask_id, noisy_batch)
            
            # In shifted mode, we'll return the shifted p_mask for loss calculation
            p_mask = mask_prob_shifted

        return noisy_batch, p_mask
    
    
    @torch.no_grad()
    def evaluate_perplexity_from_dataloader(self, dataloader, nll_type='mc'):
        """
        Computes zero-shot perplexity on an evaluation dataset loaded in batches.
        
        Here we assume each batch is a dict with key "input_ids" (a tensor of shape [batch_size, seq_len]).
        We split each sequence into a prefix and target. In this example, we use the first token as prefix
        and the remainder as target.
        
        Returns:
            perplexity: The overall perplexity computed over the evaluation dataset.
        """
        self.nll_type = nll_type
        total_nll = 0.0
        total_tokens = 0
        
        for batch in tqdm(dataloader, desc="Evaluating zeroshot", disable=self.disable_tqdm):
            # Assume batch["input_ids"] has shape [batch_size, seq_len]
            input_ids = batch["input_ids"].to(self.device)
            batch_size = input_ids.shape[0]
            
            # Split: first token as prefix, remaining as target.
            # (You can adjust this split as needed for your evaluation.)
            prefix = input_ids[:, :1]    # shape: [batch_size, 1]
            target = input_ids[:, 1:]    # shape: [batch_size, seq_len - 1]
            # For each example in the batch, compute the NLL.
            for i in range(batch_size):
                ex_prefix = prefix[i]  # 1D tensor of length 1 (or more, if you choose a larger prefix)
                ex_target = target[i]  # 1D tensor of target tokens
                if ex_target.numel() == 0:
                    continue
                
                if self.nll_type == 'mc':
                    nll = self._eval_target_nll_mc(ex_prefix,  ex_target)
                elif self.nll_type == 'chain_rule':
                    nll = self._eval_target_nll_ar(ex_prefix, ex_target)
                else:
                    raise NotImplementedError(f"nll type {self.nll_type} not supported.")
                
                total_nll += nll.item()
                total_tokens += ex_target.numel()
        
        if total_tokens == 0:
            raise ValueError("No target tokens were processed; please check your data splitting.")
        
        avg_nll = total_nll / total_tokens
        perplexity = math.exp(avg_nll)
        return perplexity
        
    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        if self.padding:
            input_ids = torch.full((batch.size(0), 1024), self.mask_id, device=self.device)
            input_ids[:, :batch.shape[1]] = batch
        else:
            input_ids = batch

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            positions = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            
            if hasattr(self.model, 'module'):
                model_to_call = self.model.module
            else:
                model_to_call = self.model
                
            try:
                logits = model_to_call(input_ids, positions=positions)
            except TypeError:
                try:
                    logits = model_to_call(input_ids)
                except TypeError:
                    sigma = torch.ones(input_ids.shape[0], input_ids.shape[1], device=self.device) * self.sampling_eps
                    logits = model_to_call(input_ids, sigma=sigma, positions=positions)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        '''
        Utilize the chain rule to compute the likelihood
        We need to perform len(target) forward passes in parallel
        '''
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2

        prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        mask_index = torch.triu(mask_index)

        perturbed_[mask_index] = self.mask_id
        perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        temp_index = torch.triu(temp_index, diagonal=1)
        mask_index[temp_index] = False
        logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().float()
        return loss

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        '''
        Employ Monte Carlo estimation to establish a lower bound of the log-likelihood
        
        Args:
            prefix: Tensor containing prefix tokens
            target: Tensor containing target tokens to predict
            mode: Either 'aligned' (mask current token) or 'shifted' (mask next token)
            
        Returns:
            nll: Negative log likelihood estimate
        '''
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq)
            
            if self.mode == 'aligned':
                mask_indices = perturbed_seq == self.mask_id
                
                if not mask_indices.any():
                    continue
                
                logits = self.get_logits(perturbed_seq, prompt_index)
                
                loss = F.cross_entropy(
                    logits[mask_indices].view(-1, logits.size(-1)), 
                    seq[mask_indices], 
                    reduction='none'
                ) / p_mask[mask_indices]
                
            else:  # 'shifted' mode
                has_next_masked = torch.zeros_like(perturbed_seq, dtype=torch.bool)
                has_next_masked[:, :-1] = perturbed_seq[:, 1:] == self.mask_id
                
                if not has_next_masked.any():
                    continue
                
                logits = self.get_logits(perturbed_seq, prompt_index)
                
                current_logits = logits[has_next_masked]
                
                next_pos_indices = torch.nonzero(has_next_masked)
                next_tokens = seq[next_pos_indices[:, 0], next_pos_indices[:, 1] + 1]
                
                loss = F.cross_entropy(
                    current_logits,
                    next_tokens,
                    reduction='none'
                )
                
                p_mask_shifted = torch.zeros_like(p_mask)
                p_mask_shifted[:, :-1] = p_mask[:, 1:]  # Shift p_mask left
                
                p_mask_values = p_mask_shifted[has_next_masked]
                
                loss = loss / p_mask_values
            
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.cpu())

        return sum(loss_acc) / len(loss_acc)
