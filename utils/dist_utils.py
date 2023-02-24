"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def setup_dist(args):
    def set_function(main_worker):
        """
        Setup a distributed process group.
        """
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.fastest = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        os.environ["MASTER_ADDR"] = "127.0.0.1"  #
        os.environ["MASTER_PORT"] = "8888"  #
        world_size = 1
        port_id = 10002 + np.random.randint(0, 1000) + int(args.cuda_devices[0])
        dist_url = "tcp://127.0.0.1:" + str(port_id)
        ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node * world_size
        print("multiprocessing_distributed")
        torch.multiprocessing.set_start_method("spawn")
        mp.spawn(  # Left 2: softmax weight=1 Right 2: softmax weight=2
            main_worker, nprocs=ngpus_per_node, args=(args,ngpus_per_node, world_size, dist_url)
        )
    return set_function




def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
