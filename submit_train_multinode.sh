#!/bin/bash
#SBATCH --job-name=finetune_qwen2vl
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -o finetune_output_multinode.log

export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1


#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

# AWS specific (if needed)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens
export NCCL_IGNORE_DISABLED_P2P=1

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate llamafactory

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export NNODES=$SLURM_NNODES


export FORCE_TORCHRUN=1


# Run the command on each node
srun --ntasks=$NNODES --ntasks-per-node=1 \
    bash -c "FORCE_TORCHRUN=1 NNODES=$NNODES RANK=\$SLURM_PROCID MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    llamafactory-cli train ./train_configs/qwen2_vl_7b_sft_cfg.yaml"

echo "END TIME: $(date)"
