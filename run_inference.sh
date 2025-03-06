#!/bin/bash
# Environment setup for DeepSpeed with ZeRO-3
export TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ensure we're using DeepSpeed ZeRO-3
export ACCELERATE_USE_DEEPSPEED=true
# Disable HF Hub downloads since there's no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load modules
source /etc/profile
module load anaconda/Python-ML-2025a

# Install package (if needed)
pip install -e scribe_agent

# Print system info
echo "=== Running cross-modal web agent inference with DeepSpeed ZeRO-3 ==="
echo "Working directory: $(pwd)"
echo "GPU information:"
nvidia-smi

# Run inference with accelerate using DeepSpeed
# Using local DeepSpeed config file
accelerate launch \
    --config_file default_config.yaml \
    --main_process_port 29501 \
    run_inference.py \
    --model_path "checkpoints/best_model" \
    --pickled_dataset "dataset.pkl" \
    --num_examples 10 \
    --use_multi_gpu \
    --output_dir "inference_results"

echo "=== Inference completed ==="
echo "Results saved to: inference_results"