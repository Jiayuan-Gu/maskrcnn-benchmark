import torch
from torch.nn.parallel._functions import Scatter, Gather


def scatter(obj, device):
    if isinstance(obj, torch.Tensor):
        # return Scatter.apply([device], None, 0, obj)[0]
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (tuple, list)):
        return type(obj)(scatter(x, device) for x in obj)
    if isinstance(obj, dict):
        return type(obj)((k, scatter(v, device)) for k, v in obj.items())
    if hasattr(obj, "to"):
        return obj.to(device, non_blocking=True)
    return obj


def gather(obj, device):
    if isinstance(obj, torch.Tensor):
        return Gather.apply(device.index, 0, obj)
    if isinstance(obj, dict):
        return type(obj)((k, gather(v, device))
                         for k, v in obj.items())
    if isinstance(obj, (tuple, list)):
        return type(obj)(gather(o, device)
                         for o in obj)
    if hasattr(obj, "to"):
        return obj.to(device, non_blocking=True)
    return obj
