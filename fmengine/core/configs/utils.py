from omegaconf import OmegaConf

def dict_to_config(d: dict):
    """
    Convert a dictionary to a config object.
    """
    return OmegaConf.structured()