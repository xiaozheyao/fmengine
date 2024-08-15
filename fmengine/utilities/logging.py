import os
from loguru import logger as _logger


class _Logger:
    def __init__(self) -> None:
        self.rank = os.environ.get("LOCAL_RANK", 0)
        self.world = os.environ.get("WORLD_SIZE", 1)

    def info(self, msg):
        _logger.info(f"[Rank {self.rank}/{self.world}] {msg}")

    def warning(self, msg):
        _logger.warning(f"[Rank {self.rank}/{self.world}] {msg}")

    def error(self, msg):
        _logger.error(f"[Rank {self.rank}/{self.world}] {msg}")

logger = _Logger()