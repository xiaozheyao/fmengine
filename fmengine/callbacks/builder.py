from typing import TYPE_CHECKING, List, Optional
from fmengine.utilities import get_component_from_path

if TYPE_CHECKING:
    from fmengine.core.configs.train_config import CallbackConfig


def build_callbacks(callback_configs: Optional[List["CallbackConfig"]]):
    callbacks = []
    if callback_configs is not None:
        for callback_config in callback_configs:
            cb = get_component_from_path(callback_config.callback_class)(**callback_config.callback_args)
            print(cb)
            callbacks.append(cb)
    return callbacks
