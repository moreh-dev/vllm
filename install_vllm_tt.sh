#!/bin/bash

source /home/jungwook/tt-metal_moreh/python_env/bin/activate

export VLLM_TARGET_DEVICE="tt"

pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu