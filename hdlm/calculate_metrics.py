import torch
from typing import List
import torch.nn.functional as F
import gc
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import DataLoader
import logging
import mauve
import hdlm.utils as utils

logger = logging.getLogger(__name__)


def calculate_gpt2_perplexity(
        samples: torch.Tensor,
        device: str = 'cuda',
        batch_size: int = 8
    ) -> torch.Tensor:
    """
    Calculate GPT-2 perplexity for generated samples
    
    Args:
        samples: Generated token samples (using gpt2 tokenizer) [batch_size, seq_len]
        device: Device to use for computation  
        batch_size: Batch size for evaluation (should match cfg.eval.perplexity_batch_size)
    
    Returns:
        Average perplexity across all samples (as tensor for distributed reduction)
    """
    eval_model = None
    try:
        if isinstance(device, str):
            device = torch.device(device)
        
        # Force memory cleanup before loading model
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        logger.info(f"Loading gpt2-large for perplexity evaluation...")
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
        
        # Reduce batch size if samples are large to save memory
        effective_batch_size = min(batch_size, 4)  # Reduce to max 4 per batch
        
        logger.info(f"Calculating GPT-2 perplexity for {samples.shape[0]} samples...")
        
        # Ensure samples are on the correct device
        samples = samples.to(device)
        
        batches = samples.shape[0] // effective_batch_size
        total_perplexity = 0
        
        with torch.no_grad():
            for i in range(batches):
                s = samples[i * effective_batch_size:(i + 1) * effective_batch_size]
                
                # Process in smaller chunks to reduce memory usage
                loss, logits = eval_model(s, labels=s)[:2]
                logits = logits.transpose(-1, -2)
                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                total_perplexity += perplexity
                
                # Clean up intermediate tensors
                del loss, logits, s
                
                # Force memory cleanup every few batches
                if i % 2 == 0:
                    torch.cuda.empty_cache()
                    
        total_perplexity /= batches
        
        # Comprehensive memory cleanup
        del eval_model
        del samples
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        
        # Return as tensor for distributed reduction
        return total_perplexity
        
    except Exception as e:
        logger.error(f"Error calculating GPT-2 perplexity: {e}")
        # Comprehensive cleanup on error
        if eval_model is not None:
            try:
                del eval_model
            except:
                pass
        try:
            del samples
        except:
            pass
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        return torch.tensor(float('inf'), device=device)


def calculate_mauve_score(
        generated_samples: List[str], 
        human_samples: List[str], 
        max_samples: int = 1000,
        max_text_length: int = 128, 
        batch_size: int = 8
    ) -> torch.Tensor:
    """
    Calculate MAUVE score between generated and human samples.
    
    Args:
        generated_samples: List of generated text samples
        human_samples: List of human-written text samples
        max_samples: Maximum number of samples to use for evaluation
        max_text_length: Maximum text length to consider
        batch_size: Batch size for processing
        
    Returns:
        MAUVE score (0-1, higher is better)
    """
    try:
        # Limit samples to avoid memory issues
        if len(generated_samples) > max_samples:
            generated_samples = generated_samples[:max_samples]
        if len(human_samples) > max_samples:
            human_samples = human_samples[:max_samples]
        
        # Further limit if we have too few human samples
        min_samples = min(len(generated_samples), len(human_samples))
        generated_samples = generated_samples[:min_samples]
        human_samples = human_samples[:min_samples]
        
        logger.info(f"Computing MAUVE score with {len(generated_samples)} generated and {len(human_samples)} human samples...")
        
        # Force memory cleanup before MAUVE computation
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate MAUVE score
        mauve_results = mauve.compute_mauve(
            p_text=human_samples,
            q_text=generated_samples,
            device_id=0 if torch.cuda.is_available() else -1,
            max_text_length=max_text_length,
            verbose=False,
            batch_size=batch_size,
            use_float64=False  # Use float32 for memory efficiency
        )
        
        mauve_score = mauve_results.mauve if mauve_results.mauve is not None else 0.0
        
        # Clean up after MAUVE computation
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        
        return torch.tensor(mauve_score, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Error while calculating MAUVE score: {e}")
        # Clean up on error
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        return torch.tensor(0.0)


def extract_human_samples(dataloader: DataLoader, tokenizer, num_samples: int = 32) -> List[str]:
    """
    Extract human-written samples from the dataloader.
    
    Args:
        dataloader: DataLoader containing human-written text
        tokenizer: Tokenizer for decoding
        num_samples: Number of samples to extract
    
    Returns:
        List of human-written text samples
    """
    try:
        logger.info(f"Extracting {num_samples} human-written samples from dataloader...")
        
        human_samples = []
        for batch in dataloader:
            if isinstance(batch, dict):
                tokens = batch.get('input_ids', batch.get('tokens', None))
            else:
                tokens = batch
            
            if tokens is not None:
                # Decode tokens to text
                texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                human_samples.extend(texts)
                
                if len(human_samples) >= num_samples:
                    break
        
        # Limit to requested number
        human_samples = human_samples[:num_samples]
        
        if len(human_samples) < num_samples:
            logger.warning(f"Only found {len(human_samples)} human samples, expected {num_samples}")
        
        return human_samples
        
    except Exception as e:
        logger.error(f"Error extracting human samples: {e}")
        return []


def calculate_generation_metrics(
        eval_dataloader,
        generated_samples: torch.Tensor,
        tokenizer,
        device: str = 'cuda'
    ) -> dict:
    """
    Calculate all generation metrics: GPT-2 perplexity, text entropy, and MAUVE score.
    This function is standalone and doesn't handle distributed reduction - handle externally.
    
    Args:
        eval_dataloader: Reference dataloader for human text
        generated_samples: Generated samples tensor [num_samples, seq_length]
        tokenizer: Tokenizer for text processing
        device: Device for computation
    
    Returns:
        Dictionary containing all generation metrics as raw values (not distributed)
    """
    
    metrics = {}
    
    print("=== Calculating Generation Metrics ===")
    
    # Force initial memory cleanup
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Get samples as strings
    generated_texts = tokenizer.batch_decode(generated_samples, skip_special_tokens=True)

    # GPT-2 Perplexity
    try:
        assert isinstance(tokenizer, GPT2TokenizerFast), (
            f"Not implemented: re-tokenize `generated_texts` with GPT2TokenizerFast tokenizer instead of {type(tokenizer).__name__}."
        )
        # TODO: Convert assertion above to `if` and re-tokenize for GPT2 when needed.
        gpt2_ppl = calculate_gpt2_perplexity(generated_samples, device=device)
        metrics['gpt2_perplexity'] = gpt2_ppl
        
        # Clean up after GPT-2 calculation
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        
    except Exception as e:
        import traceback
        print(f"Error calculating GPT-2 perplexity:")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print(f"Full Traceback:\n{traceback.format_exc()}")
        metrics['gpt2_perplexity'] = None
        
        # Clean up on error
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
    
    # Text Entropy
    try:
        generated_texts = tokenizer.batch_decode(generated_samples, skip_special_tokens=True)
        entropy = utils.compute_entropy(generated_texts, tokenizer, device=device)
        metrics['text_entropy_mean'] = entropy.mean()
        metrics['text_entropy_std'] = entropy.std()
        
        # Clean up after entropy calculation
        del entropy
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        
    except Exception as e:
        import traceback
        print(f"Error calculating text entropy:")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print(f"Full Traceback:\n{traceback.format_exc()}")
        metrics['text_entropy_mean'] = None
        metrics['text_entropy_std'] = None
        
        # Clean up on error
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()

    # MAUVE Score
    try:
        # Limit number of samples for memory efficiency
        max_samples_for_mauve = min(len(generated_texts), 128)
        generated_texts = generated_texts[:max_samples_for_mauve]
        
        human_samples = extract_human_samples(eval_dataloader, tokenizer, num_samples=max_samples_for_mauve)
        mauve_results = calculate_mauve_score(generated_texts, human_samples)
        metrics['mauve_score'] = mauve_results
        
        # Clean up after MAUVE calculation
        del generated_texts, human_samples
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        
    except Exception as e:
        import traceback
        print(f"Error calculating MAUVE score:")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print(f"Full Traceback:\n{traceback.format_exc()}")
        metrics['mauve_score'] = None
        
        # Clean up on error
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
    
    # Final cleanup
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    
    print("=== Generation Metrics Summary ===")
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.item():.4f}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: Failed to calculate")
    
    return metrics

