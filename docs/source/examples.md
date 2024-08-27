# Examples

## Training

```bash
torch run --nproc_per_node 4 fmengine/cli/main.py train --config={config}.yaml
```

## Helpful Scripts

Convert Distributed Checkpoint to PyTorch Checkpoint:

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch /path/to/ckpts/step-{X} .local/ckpts/ckpt.pt
```

