from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities import rank_zero_only


class RayEnvironment(LightningEnvironment):
    """Environment for PTL training on a Ray cluster."""

    def __init__(self, world_size):
        self.set_world_size(world_size)
        self._global_rank = 0
        self._is_remote = False

    def set_remote_execution(self, is_remote: bool) -> None:
        self._is_remote = is_remote

    def is_remote(self) -> bool:
        return self._is_remote

    def creates_processes_externally(self) -> bool:
        return False

