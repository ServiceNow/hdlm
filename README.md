# Hybrid Diffusion Language Models (HDLM)

[![Website](https://img.shields.io/badge/Website-green)](https://hdlm-colm.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2504.06416)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/collections/hdlm-group)

This repository contains the official implementation of **Hybrid Diffusion Language Models (HDLM)**, a novel approach that unifies autoregressive and diffusion-based sequence generation. Our models achieve state-of-the-art performance on text generation tasks while maintaining the benefits of both autoregressive and diffusion paradigms.

## Overview

HDLM introduces a unified framework that combines the strengths of autoregressive and diffusion-based language models.

- **Hyperschedules**: Position-dependent noise schedules for flexible generation.
- **Hybrid noising processes**: Interpolation between absorbing and uniform processes, improving error correction in mask-based models.
- **Adaptive Correction Sampler (ACS)**: Novel inference algorithm for error correction.

Our implementation provides two main model families, one for each of our hybrid noising processes. Simply put, these models differ in *how* they "interpolate" between absorbing and uniform processes.
- **Epsilon-Hybrid**: interpolate the *evolution* operators, yielding models conceptually closer to MDLM (Sahoo et al. 2024).
- **Gamma-Hybrid**: interpolate the *transition* operators, yielding models conceptually closer to SEDD (Lou et al. 2024).

The names "epsilon" and "gamma" stem from the parameters ε and γ in [the paper](https://openreview.net/forum?id=rgq9BFXSFl). Crudely speaking, these parameters specify "how much uniform is blended with the absorbing process" in the corresponding hybrid family.

We provide model weights under the naming convention `hdlm-group/hdlm-base-{family}-{value}`, where `family` is either `epsilon` or `gamma`, and `value` is the value taken by the corresponding parameter. See https://huggingface.co/hdlm-group for all available combinations.

## Quick Start

We recommend creating a [uv](https://docs.astral.sh/uv/) environment with all our requirements.
```bash
# Clone the repository
git clone https://github.com/ServiceNow/hdlm.git
cd hdlm

# Create and activate the uv environment
bash create_uv_venv.sh
source .venv/bin/activate
```

You may generate samples with our provided model weights by passing the `--save_samples` option to our `eval_generation.py` script.
```bash
python hdlm/eval_generation.py \
    --checkpoint_path hdlm-group/hdlm-base-epsilon-0.01 \
    --sampling_method full_diff \
    --algorithm acs \
    --save_samples
```

A different script is used for perplexity.
```bash
# Perplexity evaluation
python hdlm/eval_modeling.py \
    --checkpoint_path hdlm-group/hdlm-base-epsilon-0.01 \
    --work_dir "./logs/eval_modeling_epsilon_`date +%s`" \
    --dataset ptb
```

Both these scripts can be pointed at a local checkpoint.
```bash
python hdlm/eval_modeling.py \
    --checkpoint_path exp_local/openwebtext-train/hdlm-experiments/epsilon-hdlm-1/checkpoints/checkpoint_123456.pth \
    --config_path exp_local/openwebtext-train/hdlm-experiments/epsilon-hdlm-1 \
    --work_dir "./logs/eval_modeling_epsilon_`date +%s`" \
    --dataset ptb
```

We provide example configurations in the `configs` folder. For example, you may train an epsilon-hybrid model on two 80GB GPUs.
```bash
accelerate launch --config-file configs/accelerate/multi2.yaml hdlm/train.py --config-name epsilon_hdlm
```

A different script and configuration file template is provided for finetuning our provided model weights.
```bash
accelerate launch --config-file configs/accelerate/multi2.yaml hdlm/finetune.py --config-name epsilon_finetune_from_hf.yaml
```

## Licenses

Our own contributions are distributed under the [MIT license](LICENSE).

This project is built off [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), which is also distributed under the [MIT license](LICENSE).

Additional code was adapted from [MDLM](https://github.com/kuleshov-group/mdlm), distributed under the [Apache 2.0](LICENSE-APACHE.md) license.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{fathi2025unifying,
  title={Unifying autoregressive and diffusion-based sequence generation},
  author={Fathi, Nima and Scholak, Torsten and No{\"e}l, Pierre-Andr{\'e}},
  journal={arXiv preprint arXiv:2504.06416},
  year={2025}
}
```

