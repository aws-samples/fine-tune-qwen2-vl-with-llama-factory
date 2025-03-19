#!/bin/bash
#SBATCH --job-name=finetune_qwen2vl
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o finetune_output.log

export GPUS_PER_NODE=1

export OMP_NUM_THREADS=1
# If you are finetuning Qwen2.5 VL, you might need to use transformers>4.49.0
# see issue: https://github.com/huggingface/transformers/pull/36188
# export DISABLE_VERSION_CHECK=1


# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate llamafactory

export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL llamafactory-cli train ./train_configs/train/qwen2_vl_7b_pissa_qlora_128_fintabnet_en.yaml
