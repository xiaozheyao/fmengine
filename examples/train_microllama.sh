CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=8 torchrun --nproc-per-node=2 fmengine/cli/main.py train --config=examples/microllama.yaml