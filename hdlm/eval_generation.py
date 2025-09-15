import argparse
import datetime
import os

from hdlm.utils import cfg_to_family
from hdlm.hf_utils import load_model_and_config

import torch
from transformers import GPT2TokenizerFast
import json

import hdlm.data as data
import hdlm.calculate_metrics as metrics

# Specific to epsilon
import hdlm.epsilon_hybrid.sample as sample_utils

# Specific to gamma
import hdlm.gamma_hybrid.sampling as sampling
import hdlm.gamma_hybrid.sampling_sar as sampling_sar
import hdlm.gamma_hybrid.noise_lib as noise_lib
import hdlm.gamma_hybrid.graph_lib as graph_lib


def generate_samples_epsilon(model, cfg, num_samples, device, metaschedule=None, sampling_method='diff_sample_epsilon', algorithm='original', steps=128, eta=0.01):
    print(f"Generating {num_samples} samples using {sampling_method}...")
    
    batch_size = min(16, num_samples)
    all_samples = []
    print(f"Sampling method: {sampling_method}")
    print(f"Algorithm: {algorithm}")
    print(f"Steps: {steps}")
    print(f"Eta: {eta}")
    print(f"Metaschedule: {metaschedule}")
    print(f"Annealing: {cfg.annealing}")
    print(f"Batch size: {batch_size}")
    if sampling_method == 'full_diff':
        sampling_fn = sample_utils.full_diff
    elif sampling_method == 'semi_diff':
        sampling_fn = sample_utils.semi_diff  
    elif sampling_method == 'full_diff_shifted':
        sampling_fn = sample_utils.full_diff_shifted
    elif sampling_method == 'semi_diff_shifted':
        sampling_fn = sample_utils.semi_diff_shifted
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}")
    
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        with torch.no_grad():
            if sampling_method == 'semi_ar_sample':
                samples = sampling_fn(
                    model=model,
                    metaschedule=metaschedule,
                    annealing=cfg.annealing,
                    batch_size=current_batch_size,
                    context_length=cfg.model.length,
                    alg=algorithm,
                    temperature=1.0,
                    mask_token_id=50257,
                    device=device
                )
            elif sampling_method == 'semi_ar_sample_epsilon':
                samples = sampling_fn(
                    model=model,
                    batch_size=current_batch_size,
                    context_length=cfg.model.length,
                    alg=algorithm,
                    steps=steps,
                    temperature=1.0,
                    device=device,
                    eta=eta
                )
            else:
                samples = sampling_fn(
                    model=model,
                    metaschedule=metaschedule,
                    annealing=cfg.annealing,
                    batch_size=current_batch_size,
                    context_length=cfg.model.length,
                    alg=algorithm,
                    steps=steps,
                    temperature=1.0,
                    device=device,
                    eta=eta
                )
        
        all_samples.append(samples)
        print(f"Generated {i} samples")
    
    generated_samples = torch.cat(all_samples, dim=0)
    print(f"Successfully generated {generated_samples.shape[0]} samples")
    
    return generated_samples


def generate_samples_gamma(model, cfg, num_samples, device, metaschedule=None, sampling_method='NAR',  steps=128):
    print(f"Generating {num_samples} samples using {sampling_method}...")
    
    batch_size = min(16, num_samples)
    all_samples = []
    print(f"Sampling method: {sampling_method}")
    print(f"Steps: {steps}")
    print(f"Metaschedule: {metaschedule}")
    print(f"Annealing: {cfg.annealing}")
    print(f"Batch size: {batch_size}")

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg)
    
    sampling_eps = cfg.annealing.sampling_eps
    
    if sampling_method == "NAR":
        sampling_fn = sampling.get_sa_sampling_fn(cfg, graph, noise, metaschedule, (batch_size, cfg.model.length), sampling_eps, device)
    elif sampling_method == "SAR":
        sampling_fn = sampling_sar.get_sa_sampling_fn(cfg, graph, noise, metaschedule, (batch_size, cfg.model.length), sampling_eps, device)
    else:
        raise ValueError(f"Sampling method should be either NAR(non-autoregressive) or SAR(semi-autoregressive).")
    
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        with torch.no_grad():
            samples = sampling_fn(model)
        
        all_samples.append(samples)
        print(f"Generated {i + current_batch_size} samples")
    
    generated_samples = torch.cat(all_samples, dim=0)
    print(f"Successfully generated {generated_samples.shape[0]} samples")
    
    return generated_samples


def evaluate_on_dataset(model, cfg, tokenizer, dataset_name, num_samples, device, args):
    """
    Evaluate generation on a specific dataset.
    
    Args:
        model: Trained model
        cfg: Configuration
        tokenizer: Tokenizer
        dataset_name: Name of dataset to evaluate on
        num_samples: Number of samples to generate
        device: Device to run on
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation results
    """
    print(f"\n=== Evaluating on {dataset_name} ===")
    
    original_dataset = cfg.data.valid
    cfg.data.valid = dataset_name
    
    try:
        _, eval_ds = data.get_dataloaders(cfg, distributed=False)
        
        if args.family == "epsilon":
            generated_samples = generate_samples_epsilon(
                model=model,
                cfg=cfg,
                num_samples=num_samples,
                device=device,
                metaschedule=args.metaschedule,
                steps=args.steps,
                sampling_method=args.sampling_method,
                algorithm=args.algorithm,
                eta=args.eta
            )
        elif args.family == "gamma":
            generated_samples = generate_samples_gamma(
                model=model,
                cfg=cfg,
                num_samples=num_samples,
                device=device,
                metaschedule=args.metaschedule,
                steps=args.steps,
                sampling_method=args.sampling_method,
            )
        else:
            raise NotImplementedError(f"Unknown family: {args.family}")

        generation_metrics = metrics.calculate_generation_metrics(
            eval_dataloader=eval_ds,
            generated_samples=generated_samples,
            tokenizer=tokenizer,
            device=device
        )
        
        if args.save_samples:
            save_samples_to_file(generated_samples, tokenizer, dataset_name, args.sampling_method, args.checkpoint_path)
        
        return {
            'dataset': dataset_name,
            'num_samples': num_samples,
            'metrics': generation_metrics
        }
        
    except Exception as e:
        import traceback
        error_details = {
            'exception_type': type(e).__name__,
            'exception_message': str(e),
            'traceback': traceback.format_exc()
        }
        
        print(f"Error evaluating on {dataset_name}:")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print(f"Full Traceback:\n{traceback.format_exc()}")
        
        return {
            'dataset': dataset_name,
            'num_samples': num_samples,
            'metrics': None,
            'error': error_details
        }
    finally:
        cfg.data.valid = original_dataset


def save_samples_to_file(generated_samples, tokenizer, dataset_name, sampling_method, checkpoint_path):
    """Save generated samples to a predefined samples directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(script_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '').replace('.pt', '')
    
    run_dir = os.path.join(samples_dir, f"{checkpoint_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    generated_texts = tokenizer.batch_decode(generated_samples, skip_special_tokens=True)
    
    output_file = os.path.join(run_dir, f"{dataset_name}_{sampling_method}_{len(generated_texts)}samples.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Generated Samples - {dataset_name.upper()}\n")
        f.write(f"Checkpoint: {checkpoint_name}\n")
        f.write(f"Sampling Method: {sampling_method}\n")
        f.write(f"Number of Samples: {len(generated_texts)}\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, text in enumerate(generated_texts):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(text)
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Saved {len(generated_texts)} samples to {output_file}")
    return output_file


def save_results(results, output_dir):
    """Save evaluation results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'results': results
    }
    
    results_file = os.path.join(output_dir, "generation_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    summary_file = os.path.join(output_dir, "generation_evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Generation Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Samples: {result['num_samples']}\n")
            
            if result['metrics'] is not None:
                metrics_dict = result['metrics']
                f.write("Metrics:\n")
                for metric_name, value in metrics_dict.items():
                    if value is not None:
                        f.write(f"  {metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric_name}: Failed\n")
            else:
                error_info = result.get('error', 'Unknown error')
                if isinstance(error_info, dict):
                    f.write(f"Error Type: {error_info.get('exception_type', 'Unknown')}\n")
                    f.write(f"Error Message: {error_info.get('exception_message', 'Unknown')}\n")
                    f.write(f"Full Traceback:\n{error_info.get('traceback', 'Not available')}\n")
                else:
                    f.write(f"Error: {error_info}\n")
            
            f.write("\n" + "-" * 30 + "\n\n")
    
    print(f"Results saved to {results_file}")
    print(f"Summary saved to {summary_file}")


def start_parser():
    parser = argparse.ArgumentParser(add_help=False, description="Generation Evaluation for Hybrid Models")
    parser.add_argument("-h", "--help", action="store_true", 
                        help="Show this help message and exit. Specify --family for full argument help.")
    parser.add_argument("--family", type=str, required=False,
                        choices=["epsilon", "gamma"],
                        help="Which family of HDLM model to use. When used in conjunction with --help, provides additional command help specific to this family. If unspecified, inferred by attempting to load the model (not for help though).")
    return parser


def add_basic_arguments_to_parser(parser):
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=False,
                        help="Path to model configuration directory (optional for Hugging Face models)")
    
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate for each dataset")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of steps to use for sampling")
    parser.add_argument("--datasets", type=str, nargs='+', 
                        default=["ptb", "wikitext103", "lambada"],
                        help="Datasets to evaluate on")
    
    parser.add_argument("--output_dir", type=str, default="./generation_evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save generated samples to text files in ./samples directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")


def finalize_parser(family, parser):
    if family == "epsilon":
        parser.add_argument("--sampling_method", type=str, default="full_diff",
                            choices=["full_diff", "semi_diff", "full_diff_shifted", "semi_diff_shifted"],
                            help="Sampling method to use ")
        parser.add_argument("--algorithm", type=str, default="original", 
                            choices=["original", "acs",  "remask", "remask_adhoc_shuffle", "remdm"],
                            help="Algorithm variant to use within the sampling method (cleaned algorithm names)")
        
        parser.add_argument("--eta", type=float, default=0.01,
                            help="Eta parameter for epsilon algorithms")
    elif family == "gamma":
        parser.add_argument("--sampling_method", type=str, default="NAR",
                            choices=["NAR", "SAR"],
                            help="Sampling method to use (NAR: non-autoregressive, SAR: semi-autoregressive)")
    else:
        raise NotImplementedError("Unknown family: {family}")


def main():
    """Main evaluation function."""
    # Parser solely used for the sake of --help
    help_parser = start_parser()
    # The "real" parser, completed later on by calling finalize_parser
    parser = start_parser()
    add_basic_arguments_to_parser(parser)
    
    # Handle help manually
    help_args, _ = help_parser.parse_known_args()
    family = help_args.family  # This is also used in the normal flow below
    if help_args.help:
        # Show help using the real parser
        if family:
            finalize_parser(family, parser)
        parser.print_help()
        return
    # We're done with help: normal execution beyond this point

    print("=== Hybrid Generation Evaluation ===")

    # Parse what we can parse
    if family:
        # Finalize the parser and parse for realz
        print(f"Explicitly-specified family: {family}")
        finalize_parser(family, parser)
        args = parser.parse_args()
    else:
        # Parse what we can
        args, _ = parser.parse_known_args()
    
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.config_path:
        print(f"Config: {args.config_path}")
    else:
        print("Config: Auto-detected (Hugging Face model or inferred from checkpoint)")
    print(f"Datasets: {args.datasets}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    if args.save_samples:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        samples_dir = os.path.join(script_dir, "samples")
        print(f"Generated samples will be saved to: {samples_dir}")

    print("\nLoading model and configuration...")
    model, cfg, device, accelerator, metaschedule = load_model_and_config(args.checkpoint_path, args.config_path)

    # Resolve family issues
    if family:
        # Just make sure the model wasn't switched at birth        
        assert args.family == cfg_to_family(cfg)
        # Paranoia
        assert family == args.family
    else:
        family = cfg_to_family(cfg)
        print(f"Inferred family: {family}")
        finalize_parser(family, parser)
        # Parse arguments for real (some errors may show here)
        args = parser.parse_args()
        args.family = family
    
    print(f"Sampling method: {args.sampling_method}")
    args.metaschedule = metaschedule
    # `args` should be complete: no `family` issues beyond this point
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    all_results = []
    
    for dataset_name in args.datasets:
        result = evaluate_on_dataset(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            num_samples=args.num_samples,
            device=device,
            args=args
        )
        all_results.append(result)
    
    save_results(all_results, args.output_dir)
    
    print("\n=== Final Summary ===")
    for result in all_results:
        print(f"\n{result['dataset']}:")
        if result['metrics'] is not None:
            metrics_dict = result['metrics']
            for metric_name, value in metrics_dict.items():
                if value is not None:
                    print(f"  {metric_name}: {value:.4f}")
        else:
            error_info = result.get('error', 'Unknown error')
            if isinstance(error_info, dict):
                print(f"  Error Type: {error_info.get('exception_type', 'Unknown')}")
                print(f"  Error Message: {error_info.get('exception_message', 'Unknown')}")
                print(f"  See full traceback in results file for details")
            else:
                print(f"  Error: {error_info}")
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
