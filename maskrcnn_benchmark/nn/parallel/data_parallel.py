import torch
from torch.cuda._utils import _get_device_index
from torch.nn.parallel.data_parallel import _check_balance, replicate, parallel_apply

from .scatter_gather import scatter, gather


def _get_device(device_id):
    if isinstance(device_id, torch.device):
        return device_id
    else:
        return torch.device("cuda:%d" % device_id)


# TODO: add unittest
class MutableDataParallel(torch.nn.Module):
    def __init__(self, module, device_ids=None, output_device=None):
        super(MutableDataParallel, self).__init__()

        # ---------- Copied from torch.nn.parallel.DataParallel ----------
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device(output_device)  # modified

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])
        # ---------- Copied from torch.nn.parallel.DataParallel ----------

        if self.device_ids:
            self.devices = [_get_device(device_id) for device_id in self.device_ids]
        else:
            self.devices = [torch.device("cpu")]
        self.num_devices = len(self.devices)

    def forward(self, data_dicts, **kwargs):
        assert all(isinstance(d, dict) for d in data_dicts)
        if not self.device_ids:
            return [self.module(**data_dicts[0], **kwargs)]

        data_dicts, kwargs_tup = self.scatter(data_dicts, kwargs, self.devices)
        if self.num_devices == 1:
            return [self.module(**data_dicts[0], **kwargs_tup[0])]
        replicas = replicate(self.module, self.device_ids[:len(data_dicts)])
        outputs = self.parallel_apply(replicas, data_dicts, kwargs_tup)
        # return self.gather(outputs, self.output_device)
        return outputs

    def scatter(self, data_dicts, kwargs, devices):
        # scatter mini batches to devices
        data_dicts = [scatter(data_dict, device)
                      for data_dict, device in zip(data_dicts, devices)]
        # replicate kwargs for each mini batch
        kwargs_tup = [kwargs.copy() for _ in data_dicts]
        return data_dicts, kwargs_tup

    def parallel_apply(self, replicas, data_dicts, kwargs_tup):
        inputs = tuple(() for _ in data_dicts)
        kwargs_tup = tuple(dict(**data_dict, **kwargs)
                           for data_dict, kwargs in zip(data_dicts, kwargs_tup))
        return parallel_apply(replicas, inputs, kwargs_tup, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return [gather(output, output_device) for output in outputs]
