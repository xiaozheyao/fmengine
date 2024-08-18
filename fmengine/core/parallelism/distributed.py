import os
from datetime import timedelta

import torch

from fmengine.utilities import _warn_overwrite_env

TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"


def init_distributed(dump_folder, trace_buf_size=20000, init_timeout_seconds=300):
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)
    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(trace_buf_size))
    if trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = f"{dump_folder}/comm_trace"
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/rank_")

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=init_timeout_seconds))

    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
