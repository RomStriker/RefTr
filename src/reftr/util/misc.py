"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import datetime
import logging
import os
import pickle
import subprocess
from pathlib import Path
import numpy as np
from argparse import Namespace
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from src.reftr.util.eval_utils import (
    add_edge_lengths,
    get_resampled_graph,
    rename_node_names
)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_stats(input_dict):
    """
    Args:
        input_dict (dict): all the values will be gathered
    Gather the values in the dictionary from all processes. Returns a dict with the same fields as
    input_dict, after gathering.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        index = 0
        indices = [index]
        names = []
        values = []
        shapes = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            shapes.append(input_dict[k].shape)
            index += len(input_dict[k].flatten())
            indices.append(index)
            values.append(input_dict[k].flatten())
        values = torch.cat(values, dim=0)
        values_list = [torch.empty(values.shape).to('cuda') for _ in range(world_size)]
        dist.all_gather(values_list, values)
        gathered_dict = {}
        for i, name in enumerate(names):
            value = []
            for j in range(world_size):
                value.append(values_list[j][indices[i]: indices[i + 1]].reshape(shapes[i]))
            stacked = torch.stack(value, dim=1)
            interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)
            gathered_dict[name] = interleaved

    return gathered_dict


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
        # slurm process but not interactive
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(seconds=3600))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        min_values = values.min()
        max_values = values.max()
        m = min(cost_matrix.shape)

        positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
        if max_inf:
            place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
        elif min_inf:
            place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)


def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace


def nested_dict_to_device(dictionary, device):
    output = {}
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            output[key] = nested_dict_to_device(value, device)
        return output
    return dictionary.to(device)


def restore_config(args):
    checkpoint = torch.load(args.resume, map_location='cpu')
    args_n = checkpoint['args']

    # add eval args
    if args.eval_only:
        args_n.resume = args.resume
        args_n.output_dir = args.output_dir
        args_n.dataset = args.dataset
        args_n.data_dir = args.data_dir
        args_n.eval_only = args.eval_only
        args_n.mask = args.mask

    # add possible missing args
    for k in args.__dict__:
        if k not in args_n.__dict__.keys():
            args_n.__dict__[k] = args.__dict__[k]

    return args_n


def get_lr_scheduler(args, optimizer, len_dataloader):
    total_steps = int(args.epochs) * len_dataloader * int(args.traj_train_len)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=total_steps,  # Maximum number of iterations
                                                           eta_min=args.min_lr)  # Minimum learning rate.

    return scheduler


def setup_logger(save_path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(save_path / 'output.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def process_preds(preds, args):
    resample_dict = {'atm22': None, 'parse2022': None,
                     'parse2022_lr': 0.8, 'synthetic': None, 'synthetic_1.2': 1.2}
    dataset_name = Path(args.data_dir).name
    scale = resample_dict[dataset_name]

    if scale is not None:
        pred = preds[0]
        old_spacing = scale
        new_spacing = 1.0 if 'syntrx' in dataset_name else 0.5
        for node in pred.nodes:
            pred.nodes[node]['position'] = (np.array(pred.nodes[node]['position']) * old_spacing) / new_spacing
            pred.nodes[node]['radius'] = (pred.nodes[node]['radius'] * old_spacing) / new_spacing
        pred = add_edge_lengths(pred)
        pred = get_resampled_graph(pred, points_dist=1.0, smooth=False)
        pred = rename_node_names(pred, "0-0")
        preds = [pred]

    return preds

