import os
import sys

from loguru import logger as logger

config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "enqueue": True,
            "format": "<level>[Rank {extra[rank]}/{extra[world]}] [{time:YYYY-MM-DD HH:mm:ss}] [{level}]:</> {message}",
        },
    ],
    "extra": {
        "rank": os.environ.get("LOCAL_RANK", 0),
        "world": os.environ.get("WORLD_SIZE", 1),
    },
}


logger.configure(**config)
rank0_logger = logger.bind()