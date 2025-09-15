#!/bin/bash
export UV_TORCH_BACKEND=cu124 
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy
uv venv --python 3.11
uv pip install --upgrade --no-build-isolation setuptools
uv pip install --no-build-isolation ninja cmake
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
uv pip install --no-build-isolation flash-attn==2.7.4.post1
uv pip install -r requirements.txt
uv pip install -e .
