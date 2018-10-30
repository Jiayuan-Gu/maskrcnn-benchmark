# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.nn.parallel.scatter_gather import Gather


def parse_output_dicts(output_dicts):
    # TODO: add metric dict
    loss_dict = dict()
    result_dict = dict()

    for k in output_dicts[0].keys():
        values = tuple(d[k] for d in output_dicts)
        if k.startswith("loss"):
            loss_dict[k] = values
        else:
            result_dict[k] = values
    return loss_dict, result_dict


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    arguments,
    checkpoint_period,
):
    """ Training interface

    .. note::
        It is a little tricky when doing training in parallel.
        1. Single gpu training
            Straightforward pipeline. output = model(*inputs, **kwargs)
            output is just what model.forward return, and allocated on current device.
        2. Multi gpus training (data parallel)
            torch.nn.DataParallel will do four steps to enable data parallel:
            (1) Scatter the inputs from data_loader.
                It assumes all tensors in inputs are already collated,
                and split them evenly with structure remained to all devices.
            (2) Replicate module to all devices (differentiable)
            (3) Forward model in parallel
            (4) Gather all the outputs to the first device.
                The output of module should be tensor or dict.
                It stack scalars or concat tensors along 0-axis
            Thus, we use MutableDataParallel to make things easier
        3. Distributed training (single gpu per process)
            For each process, model is identical to that in single gpu training.
            losses could be backward on each process,
            unless there are some losses calculated on the full batch.
            Only master process will have valid logger, meters, checkpoint

    Args:
        model (torch.nn.Module):
        data_loader (torch.utils.data.DataLoader):
            When doing distributed training,
            data_loader will provide corresponding data for each process.
        optimizer (torch.optim.Optimizer):
        scheduler (torch.optim.lr_scheduler._LRScheduler):
        checkpointer (DetectronCheckpointer):
        arguments (dict): arguments to save in the checkpoint
        checkpoint_period (int): period to save checkpoint

    Returns:

    """
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (data_dicts, img_ids) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()

        output_dicts = model(data_dicts)

        loss_dict, result_dict = parse_output_dicts(output_dicts)
        # average loss across all gpus
        loss_dict = {k: Gather.apply(0, 0, *v).mean() for k, v in loss_dict.items()}

        losses = sum(loss for loss in loss_dict.values())

        # with torch.no_grad():
        meters.update(loss=losses, **loss_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == (max_iter - 1):
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
