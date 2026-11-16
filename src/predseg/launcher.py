"""Launch a single process, local DDP workers, or workers managed by torchrun."""

import os
import random


def _worker(rank, world_size, config, stage):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if "LOCAL_RANK" not in os.environ or not os.environ.get("TORCHELASTIC_RUN_ID"):
        os.environ["LOCAL_RANK"] = str(rank)
    import numpy as np
    import torch
    from predseg.utils.distributed import init_distributed

    seed = config["meta"].get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    init_distributed(port=config["meta"].get("port", 29500))
    try:
        if stage == "pretrain":
            from predseg.pretraining.train import main
        else:
            from predseg.segmentation.train import train_main as main
        # Legacy modules seed at import; apply the requested experiment seed afterward.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        main(config)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def launch(config, stage, devices=None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if devices:
            raise ValueError("Let torchrun choose devices; omit --devices under torchrun")
        _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), config, stage)
        return
    devices = devices or ["cuda:0"]
    if len(devices) != len(set(devices)):
        raise ValueError("Devices must be unique")
    if devices == ["cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif any(not name.startswith("cuda:") or not name[5:].isdigit() for name in devices):
        raise ValueError("Use one cpu device or a list such as cuda:0 cuda:1")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(name[5:] for name in devices)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(config["meta"].get("port", 29500)))
    import torch

    if devices != ["cpu"] and (
        not torch.cuda.is_available() or torch.cuda.device_count() < len(devices)
    ):
        raise RuntimeError(
            "Requested CUDA devices are unavailable; use --devices cpu for CPU execution"
        )
    if len(devices) == 1:
        _worker(0, 1, config, stage)
    else:
        torch.multiprocessing.spawn(
            _worker, args=(len(devices), config, stage), nprocs=len(devices), join=True
        )
