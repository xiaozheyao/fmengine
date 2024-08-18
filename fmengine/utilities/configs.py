from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Union

from omegaconf import DictConfig, OmegaConf

from .errors import InstantiationError


def has_component(node: Union[Dict[str, Any], DictConfig]) -> bool:
    return (OmegaConf.is_dict(node) or isinstance(node, dict)) and "_component_" in node


def get_component_from_path(path: str) -> Any:
    """
    Return an object by name or dotted path, importing as necessary.
    The base functionality relies on ``getattr()`` and handles all
    possible exceptions accordingly.

    Based on Hydra's `_locate` from Facebook Research:
    https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/utils.py#L614

    Args:
        path (str): Dotted path of the object

    Returns:
        Any: The object

    Raises:
        InstantiationError: If there is an exception loading the
            object from the provided path
        ValueError: If a relative or invalid dotpath is passed in
    """
    if path == "":
        raise ValueError("Empty path")

    parts = [part for part in path.split(".")]
    for part in parts:
        # If a relative path is passed in, the first part will be empty
        if not len(part):
            raise ValueError(f"Error loading '{path}': invalid dotstring." + "\nRelative imports are not supported.")
    # First module requires trying to import to validate
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except ImportError as exc_import:
        raise InstantiationError(
            f"Error loading '{path}':\n{repr(exc_import)}" + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    # Subsequent components can be checked via getattr() on first module
    # It can either be an attribute that we can return or a submodule that we
    # can import and continue searching
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        # If getattr fails, check to see if it's a module we can import and
        # continue down the path
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise InstantiationError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                # Any other error trying to import module can be raised as
                # InstantiationError
                except Exception as exc_import:
                    raise InstantiationError(f"Error loading '{path}':\n{repr(exc_import)}") from exc_import
            # If the component is not an attribute nor a module, it doesn't exist
            raise InstantiationError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj
