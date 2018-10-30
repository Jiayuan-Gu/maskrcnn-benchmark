# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


def merge_dicts(dicts):
    keys = dicts[0].keys()
    return {k: [d[k] for d in dicts] for k in keys}


def split_list(l, num_slice):
    assert len(l) >= num_slice
    step = len(l) // num_slice
    slices = [l[i * step:(i + 1) * step] for i in range(num_slice)]
    return slices


def _split_and_merge(batch, num_slice, update_keys=True):
    """ Split a batch into several slices,
        and merge each slice into one dictionary

    Args:
        batch (list[dict]): a list of samples from the dataset
        num_slice (int): the number of slices of a batch.
        update_keys (bool): whether to update keys to plural

    Returns:
        data_dicts (list[dict])

    """
    assert isinstance(batch[0], dict)
    slices = split_list(batch, num_slice)
    data_dicts = [merge_dicts(l) for l in slices]
    if update_keys:
        data_dicts = [{(k + 's'): v for k, v in data_dict.items()}
                      for data_dict in data_dicts]
    return data_dicts


# TODO: add unittest
class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, num_slice, size_divisible=0):
        """

        Args:
            num_slice (int): the number of slices of a batch.
                Equal to the number of devices
            size_divisible (int): the size that an image should be divisible by.
        """
        assert num_slice >= 1
        self.num_slice = num_slice
        self.size_divisible = size_divisible

    def __call__(self, batch):
        data_dicts = _split_and_merge(batch, self.num_slice, update_keys=True)
        img_ids = []
        for data_dict in data_dicts:
            images = to_image_list(data_dict['images'], self.size_divisible)
            data_dict['images'] = images
            img_ids.append(data_dict.pop("img_ids"))
        return data_dicts, img_ids
