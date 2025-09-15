import argparse
import datetime
import os

from hdlm.hf_utils import load_model_and_config

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import hdlm.data as data
import hdlm.utils as utils
import hdlm.zeroshot_ppl as zeroshot_ppl


class WrapperDDP(nn.parallel.DistributedDataParallel):
    """
    Wrapper for DistributedDataParallel that allows access to module attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _run(rank, args):
    """
    Main evaluation function for multi process.
    
    Args:
        rank: Process rank for distributed evaluation
        args: Command line arguments
    """
    if rank == 0:
        logger = utils.get_logger(os.path.join(args.work_dir, f"zero_shot_on_{args.valid_dataset}_logs"))

    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    mprint("================================")

    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    mprint("================================")

    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        mprint(f"{arg_name}: {arg_value}")

    model, cfg, device, _, _ = load_model_and_config(args.checkpoint_path, args.config_path)
    cfg.data.valid = args.valid_dataset
    
    token_dim = model.config.tokenizer.tokens + 1 if hasattr(model.config, 'tokenizer') else 50258
    
    model = WrapperDDP(model, device_ids=[rank], static_graph=True)

    _, eval_ds = data.get_dataloaders(cfg, distributed=False)    

    # Check if we have metaschedule and annealing available
    metaschedule = None
    annealing = None
    if hasattr(cfg, 'annealing') and hasattr(cfg, 'metaschedule'):
        from hdlm.metaschedule import make_simple_annealing, make_block_annealing, make_hybrid_annealing
        
        if cfg.annealing.type == "simple":
            metaschedule = make_simple_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
        elif cfg.annealing.type == "block":
            metaschedule = make_block_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
        elif cfg.annealing.type == "hybrid":
            metaschedule = make_hybrid_annealing(cfg.annealing.width, cfg.annealing.eval_tau, cfg.model.length)
        
        annealing = cfg.annealing

    zero_shot_eval = zeroshot_ppl.ZeroShot_calculator(
        model=model,
        device=device,
        nll_type='mc',
        mask_id=token_dim - 1,
        mc_num=args.monte_carlo_timesteps,
        batch_size=args.batch_size,
        sampling_eps=1e-3,
        mode='aligned',
        metaschedule=metaschedule,
        annealing=annealing
    )

    mprint(f"Starting evaluation on {args.valid_dataset} dataset...")
    mprint(f"Using {args.monte_carlo_timesteps} Monte Carlo timesteps")
    if metaschedule is not None and annealing is not None:
        mprint(f"Using annealing-based evaluation with {cfg.annealing.type} metaschedule")

    with torch.no_grad():
        ppl = zero_shot_eval.evaluate_perplexity_from_dataloader(eval_ds, nll_type='mc')
        
        # For distributed evaluation, we need to gather results from all processes
        ppl_tensor = torch.tensor(ppl, device=device)
        dist.all_reduce(ppl_tensor)
        ppl = ppl_tensor.item() / args.ngpus
        
        if ppl == float('inf'):
            print(f"Warning: Perplexity is infinity. This typically indicates very poor model performance.")

    mprint("================================")
    mprint(f"Evaluation Results:")
    mprint(f"  Dataset: {args.valid_dataset}")
    mprint(f"  Perplexity: {ppl:.4f}")
    mprint(f"  Sequence Length: {args.length}")
    mprint(f"  Monte Carlo Timesteps: {args.monte_carlo_timesteps}")
    mprint("================================\n\n")

    # Save results to file if main process
    if rank == 0:
        results = {
            'dataset': args.valid_dataset,
            'perplexity': ppl,
            'sequence_length': args.length,
            'monte_carlo_timesteps': args.monte_carlo_timesteps,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        results_file = os.path.join(args.work_dir, f"perplexity_results_{args.valid_dataset}.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        mprint(f"Results saved to {results_file}")


def run(rank, args, port):
    """
    Wrapper function for distributed evaluation setup and cleanup.
    
    Args:
        rank: Process rank
        args: Command line arguments  
        port: Port for distributed communication
    """
    def setup(rank, world_size, port):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        # Initialize the process group
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
        )

    def cleanup():
        dist.destroy_process_group()

    try:
        setup(rank, args.ngpus, port)
        _run(rank, args)
    finally:
        cleanup()


def evaluate_single_dataset(args, dataset_name):
    """
    Evaluate on a single dataset without distributed processing.
    
    Args:
        args: Command line arguments
        dataset_name: Name of dataset to evaluate
    """
    print(f"=== Evaluating on {dataset_name} (Single GPU) ===")
    
    model, cfg, device, _, _ = load_model_and_config(args.checkpoint_path, args.config_path)
    cfg.data.valid = dataset_name
    
    _, eval_ds = data.get_dataloaders(cfg, distributed=False)
    
    # Check if we have metaschedule and annealing available
    metaschedule = None
    annealing = None
    if hasattr(cfg, 'annealing') and hasattr(cfg, 'metaschedule'):
        from hdlm.metaschedule import make_simple_annealing, make_block_annealing, make_hybrid_annealing
        
        if cfg.annealing.type == "simple":
            metaschedule = make_simple_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
        elif cfg.annealing.type == "block":
            metaschedule = make_block_annealing(cfg.annealing.width, cfg.annealing.eval_tau)
        elif cfg.annealing.type == "hybrid":
            metaschedule = make_hybrid_annealing(cfg.annealing.width, cfg.annealing.eval_tau, cfg.model.length)
        
        annealing = cfg.annealing

    zero_shot_eval = zeroshot_ppl.ZeroShot_calculator(
        model=model,
        device=device,
        nll_type='mc',
        mask_id=50257,  # Default GPT-2 mask token
        mc_num=args.monte_carlo_timesteps,
        batch_size=args.batch_size,
        sampling_eps=1e-3,
        mode='aligned',
        metaschedule=metaschedule,
        annealing=annealing
    )
    
    ppl = zero_shot_eval.evaluate_perplexity_from_dataloader(eval_ds, nll_type='mc')
    
    print(f"Dataset: {dataset_name}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"Sequence Length: {args.length}")
    print(f"Monte Carlo Timesteps: {args.monte_carlo_timesteps}")
    print("="*50)
    
    return {
        'dataset': dataset_name,
        'perplexity': ppl,
        'sequence_length': args.length,
        'monte_carlo_timesteps': args.monte_carlo_timesteps,
        'timestamp': datetime.datetime.now().isoformat()
    }


def main():
    """
    Main function with argument parsing and evaluation orchestration.
    """
    parser = argparse.ArgumentParser(description="Zeroshot Perplexity Evaluation for Hybrid Models")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=False,
                        help="Path to model configuration directory (optional for Hugging Face models)")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--length", type=int, default=1024,
                        help="Sequence length for perplexity calculation")
    parser.add_argument("--monte_carlo_timesteps", type=int, default=100,
                        help="Number of Monte Carlo timesteps")
    
    parser.add_argument("--valid_dataset", type=str, default="ptb",
                        help="Validation dataset name")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help="Multiple datasets to evaluate (overrides --valid_dataset)")
    parser.add_argument("--ngpus", type=int, default=1,
                        help="Number of GPUs for distributed evaluation")
    
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Working directory for logs and results")
    
    parser.add_argument("--single_gpu", action="store_true",
                        help="Use single GPU mode (simpler, no distributed)")
    
    args = parser.parse_args()
    
    print("=== HDLM Zeroshot Perplexity Evaluation ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.config_path:
        print(f"Config: {args.config_path}")
    else:
        print("Config: Auto-detected (Hugging Face model or inferred from checkpoint)")
    print(f"Monte Carlo Timesteps: {args.monte_carlo_timesteps}")
    print(f"Sequence Length: {args.length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Working Directory: {args.work_dir}")
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    if args.datasets:
        datasets = args.datasets
        print(f"Evaluating on datasets: {datasets}")
    else:
        datasets = [args.valid_dataset]
        print(f"Evaluating on dataset: {args.valid_dataset}")
    
    all_results = []
    
    if args.single_gpu or args.ngpus == 1:
        print("\nUsing single GPU evaluation mode...")
        for dataset in datasets:
            try:
                result = evaluate_single_dataset(args, dataset)
                all_results.append(result)
            except Exception as e:
                import traceback
                error_details = {
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc()
                }
                
                print(f"Error evaluating {dataset}:")
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Message: {str(e)}")
                print(f"Full Traceback:\n{traceback.format_exc()}")
                
                all_results.append({
                    'dataset': dataset,
                    'error': error_details,
                    'timestamp': datetime.datetime.now().isoformat()
                })
    else:
        print(f"\nUsing distributed evaluation with {args.ngpus} GPUs...")
        logger = utils.get_logger(os.path.join(args.work_dir, "evaluation_logs"))
        
        for dataset in datasets:
            args.valid_dataset = dataset
            port = int(np.random.randint(10000, 20000))
            
            try:
                mp.set_start_method("forkserver", force=True)
                mp.spawn(run, args=(args, port), nprocs=args.ngpus, join=True)
                
                # Load results
                results_file = os.path.join(args.work_dir, f"perplexity_results_{dataset}.json")
                if os.path.exists(results_file):
                    import json
                    with open(results_file, 'r') as f:
                        result = json.load(f)
                    all_results.append(result)
                
            except Exception as e:
                import traceback
                error_details = {
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc()
                }
                
                logger.critical(f"Error evaluating {dataset}:")
                logger.critical(f"Exception Type: {type(e).__name__}")
                logger.critical(f"Exception Message: {str(e)}")
                logger.critical(f"Full Traceback:\n{traceback.format_exc()}")
                
                all_results.append({
                    'dataset': dataset,
                    'error': error_details,
                    'timestamp': datetime.datetime.now().isoformat()
                })
    
    summary_file = os.path.join(args.work_dir, "evaluation_summary.json")
    import json
    with open(summary_file, 'w') as f:
        json.dump({
            'evaluation_summary': all_results,
            'config': vars(args),
            'timestamp': datetime.datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    
    for result in all_results:
        if 'error' not in result:
            print(f"{result['dataset']:>15}: PPL = {result['perplexity']:>8.4f}")
        else:
            error_info = result['error']
            if isinstance(error_info, dict):
                print(f"{result['dataset']:>15}: ERROR - {error_info.get('exception_type', 'Unknown')}: {error_info.get('exception_message', 'Unknown')}")
            else:
                print(f"{result['dataset']:>15}: ERROR - {error_info}")
    
    print(f"\nResults saved to: {summary_file}")
    print("="*60)


if __name__ == "__main__":
    main()
