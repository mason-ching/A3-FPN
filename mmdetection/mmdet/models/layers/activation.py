import torch.nn as nn
from typing import Optional, Union


def get_activation(act: Union[str, nn.Module, bool] = 'GELU', inpace: bool = True):
    """
    get activation
    """
    if isinstance(act, bool):
        return act

    if isinstance(act, str):
        act = act.lower()
        if len(act) == 0:
            return nn.Identity()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m
