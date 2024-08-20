#!/bin/bash

#SBATCH --job-name=fmengine
#SBATCH --container-writable
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --output=logs/slurm-%j.out

cd /fmengine
export PYTHONPATH=/fmengine

echo "START TIME: $(date)"
set -eo pipefail

# [debug] logging script's variables/commands
set -x

GPUS_PER_NODE=4

# Network
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

COMMAND="
cd /fmengine
export PYTHONPATH=/fmengine
export WANDB_API_KEY=$WANDB_API_KEY

# printing this here again for debugging purposes
echo \"### Node Init ###\"
echo \"### Job ID: \$SLURM_JOB_ID\"
echo \"### Node Name: \$SLURMD_NODENAME\"
echo \"### Node Rank: \$SLURM_PROCID\"
echo \"### Node Local Rank: \$SLURM_LOCALID\"
echo \"### Number of Nodes: \$SLURM_JOB_NUM_NODES\"
echo \"### Number of Tasks: \$SLURM_NTASKS\"
echo \"### CPUs per Task: \$SLURM_CPUS_PER_TASK\"
echo \"### GPUs per Node: $GPUS_PER_NODE\"
echo \"### Partition: \$SLURM_JOB_PARTITION\"
echo \"### Master Node: $MASTER_ADDR\"
echo \"### Master Port: $MASTER_PORT\"
echo \"### Log Dir: \$PWD/logs\"
echo \"\"

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    fmengine/cli/main.py train \
    --config /configs/tinyllama_dp.yaml
"

echo $COMMAND

# submit the job to slurm
NAME="run"

srun \
    --unbuffered \
    --output=$PWD/logs/fmengine_${NAME}_%j.log \
    --environment /store/swissai/a09/xyao/containers/fmengine.toml \
    --container-writable \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    bash -c "$COMMAND"

echo "END TIME: $(date)"