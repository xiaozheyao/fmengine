import os
import torch
from typing import Union
from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from .logging import logger


def _warn_overwrite_env(env, val):
    if env in os.environ:
        logger.warning(f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config")
    os.environ[env] = val


def dist_max(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.MAX.name, group=mesh).item()


def dist_mean(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.AVG.name, group=mesh).item()


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.hidden_size // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, becuase
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issueing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}")
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier()
    torch.cuda.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
