"""
Utility functions for distributed metric computation.

Adapted from: https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/green.py#L30
"""

import pickle
from tqdm import tqdm
import torch
import torch.distributed as dist
import os
import sys
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def tqdm_on_main(*args, **kwargs):
    if is_main_process():
        return tqdm(*args, **kwargs)
    else:
        return kwargs.get("iterable", None)


def gather_processes(all_tensors_list):
    """
    Gathers objects from all processes to all processes.
    Works with arbitrary Python objects including lists of strings.
    """
    if not dist.is_available() or not dist.is_initialized():
        return all_tensors_list

    world_size = dist.get_world_size()
    gathered_data = [None for _ in range(world_size)]

    # Use all_gather_object which handles Python objects directly
    dist.all_gather_object(gathered_data, all_tensors_list)

    # Flatten the list of lists
    result = []
    for part in gathered_data:
        if part is not None:
            result.extend(part)

    return result


def destroy_process_group_if_necessary():
    local_rank = int(os.environ.get("RANK", "0"))
    if local_rank != 0:
        dist.destroy_process_group()  # Clean up the distributed processing group
        sys.exit()  # Exit the process


def create_distributed_dataloader_if_needed(dataset, batch_size, shuffle):
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        print("Distributed dataloader created on rank: ", int(os.environ["RANK"]))
    else:
        # For single GPU/CPU, use regular DataLoader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12
        )
    return dataloader
