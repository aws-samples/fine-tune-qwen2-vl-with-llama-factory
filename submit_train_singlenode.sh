#!/bin/bash
#SBATCH --job-name=finetune_qwen2vl
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o finetune_output.log

export GPUS_PER_NODE=1

export OMP_NUM_THREADS=1


# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate llamafactory

export FORCE_TORCHRUN=1

srun --export=ALL llamafactory-cli train ./train_configs/qwen2_vl_7b_sft_cfg.yaml
