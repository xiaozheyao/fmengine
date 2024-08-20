#!/bin/bash

#SBATCH --job-name=fmengine
#SBATCH --container-writable
#SBATCH --environment=fmengine
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2

cd /fmengine
export PYTHONPATH=/fmengine

echo "START TIME: $(date)"
