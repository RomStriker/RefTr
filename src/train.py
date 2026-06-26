import os
import random
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
import sacred
import torch
import yaml
import monai
from monai.data import ThreadDataLoader, set_track_meta
from reftr.engine import Engine
import reftr.util.eval_utils as eval_utils
import reftr.util.misc as utils
import reftr.datasets.cache_dataset_tra as cd
from reftr.models import build_model
from reftr.util.misc import nested_dict_to_namespace, restore_config
from reftr.util.eval_utils import get_score_nx_ta_mft
from src.reftr.util.misc import process_preds

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

ex = sacred.Experiment('train', save_git_info=False)
ex.add_config('./cfgs/train.yaml')
ex.add_named_config('eval', './cfgs/eval.yaml')


def reload_args(args):
    if args.resume:
        args = restore_config(args)
    return args


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    monai.utils.set_determinism(seed=seed, additional_settings=None)


def build_dataloaders(args):
    data_loader_train = None
    data_loader_val = None
    data_loader_val_sv = None
    data_loader_test = None

    set_track_meta(False)
    if not args.eval_only:
        dataset_train = cd.build_dataset(args, 'training')
        data_loader_train = ThreadDataLoader(dataset_train, batch_size=args.batch_size,
                                             collate_fn=cd.train_collate_fn, num_workers=0, shuffle=True)
    if args.volume_eval:
        dataset_val = cd.build_dataset(args, 'validation')
        data_loader_val = ThreadDataLoader(dataset_val, batch_size=args.batch_size_val,
                                           collate_fn=cd.val_collate_fn, num_workers=0)
    if args.sub_volume_eval:
        dataset_val_sv = cd.build_dataset(args, 'validation_sv')
        data_loader_val_sv = ThreadDataLoader(dataset_val_sv, batch_size=args.batch_size,
                                              collate_fn=cd.val_sv_collate_fn, num_workers=0)
    if args.test_eval:
        dataset_test = cd.build_dataset(args, 'test')
        data_loader_test = ThreadDataLoader(dataset_test, batch_size=args.batch_size_val,
                                            collate_fn=cd.val_collate_fn, num_workers=0)

    return data_loader_train, data_loader_val, data_loader_val_sv, data_loader_test


def create_optim_and_lr_sched(args, param_dicts, data_loader, model, checkpoint):
    # create the optimizer and lr scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = utils.get_lr_scheduler(args, optimizer, len(data_loader))
    # load optimizer and lr scheduler state dict if provided
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    return optimizer, lr_scheduler


def test_evaluation(args, engine_tra, model):
    if len(args.test_sample):
        sav_dir = Path(args.resume).parent / 'test_result'
        sav_dir.mkdir(parents=True, exist_ok=True)
        set_track_meta(False)
        args.batch_size_per_sample = args.batch_size
        test_sample_list = args.test_sample

        for sample in test_sample_list:
            preds, targets, sample_ids, elapsed_time = engine_tra.evaluate_sinsam(model, sample)
            preds = process_preds(preds)

            # apply Tree NMS
            fil_preds = [eval_utils.merge_duplicate_branches(pred, position_threshold=0.3, min_voxel_threshold=1.5,
                                                             duplicate_ratio=0.2, passes=1) for pred in preds]

            pred_dict = {'preds': preds, 'target_ids': sample_ids, 'elapsed_time': elapsed_time,
                         'filtered_preds': fil_preds, 'resampled': True}

            torch.save(pred_dict, (sav_dir / (sample + '.pkl')))

        return


def print_log_training_metrics(logger, epoch, metrics, st):
    # get the end time
    et = time.time()
    elapsed_time = et - st
    if utils.get_rank() == 0:
        logger.info(f"Epoch: {epoch} \t | \t loss: {metrics['scaled_losses']} \t\t | \t time taken: {elapsed_time}")


def train(args: Namespace) -> None:
    args = reload_args(args)

    # Configure logging to save both to console and a log file
    logger = utils.setup_logger(Path(args.resume).parent)

    if args.distributed:
        utils.init_distributed_mode(args)
        logger.info("git:\n  {}\n".format(utils.get_sha()))

    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        val_res_dir = output_dir / 'val_results'
        val_sv_res_dir = output_dir / 'val_sv_results'
        val_res_dir.mkdir(parents=True, exist_ok=True)
        val_sv_res_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(
            vars(args),
            open(output_dir / 'config.yaml', 'w'), allow_unicode=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.determinism:
        seed = args.seed + utils.get_rank()
        fix_seed(seed)

    # build model and criterion
    model, criterion = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # load previous checkpoint if provided
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
    else:
        checkpoint = {}

    # load model state dict
    if 'model' in checkpoint:
        model_without_ddp.load_state_dict(checkpoint['model'])

    param_dicts = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad], "lr": args.lr}
    ]

    # build dataloaders
    if not len(args.test_sample):
        data_loader_train, data_loader_val, data_loader_val_sv, data_loader_test = build_dataloaders(args)
    else:
        data_loader_train, data_loader_val, data_loader_val_sv, data_loader_test = None, None, None, None

    if not args.eval_only:
        # create the optimizer and lr scheduler
        optimizer, lr_scheduler = create_optim_and_lr_sched(args, param_dicts, data_loader_train,
                                                            model_without_ddp, checkpoint)

        # load previous indicators
        if args.resume:
            if not args.eval_only:
                start_epoch = checkpoint['epoch']
                best_score = checkpoint['best_score'] if 'best_score' in checkpoint else 0
                new_best = False
        else:
            start_epoch = 0
            best_score = 0
            new_best = True

    torch.set_float32_matmul_precision('high')
    engine = Engine(args, logger, device)

    # run evaluation if eval_only is set and exit
    if args.eval_only:
        test_evaluation(args, engine, model)
        return

    if utils.get_rank() == 0:
        logger.info("Start training")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    for epoch in range(start_epoch, args.epochs):
        epoch_scaled = epoch * args.traj_train_len
        if args.distributed:
            torch.distributed.barrier()

        # while resuming we skip the first step of training
        if not (args.resume and epoch == start_epoch):
            # get the start time
            st = time.time()
            metrics = engine.train_one_epoch(model, criterion, data_loader_train,
                                             optimizer, lr_scheduler, scaler, epoch)

            if utils.get_rank() == 0:
                print_log_training_metrics(logger, epoch, metrics, st)

            if args.distributed:
                torch.distributed.barrier()

            if utils.get_rank() == 0 and args.output_dir and not epoch % args.save_checkpoint:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

                if args.save_model_interval and not epoch % args.save_model_interval:
                    checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch_scaled}.pth")

                for checkpoint_path in checkpoint_paths:
                    save_dict = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'best_score': best_score}
                    utils.save_on_master(save_dict, checkpoint_path)

        if (args.sub_volume_eval and (not epoch % args.val_interval_sv)
                or epoch == (args.epochs - 1) and args.val_interval_sv):
            (pro_preds, pro_targets, sample_ids,
            samples, masks, elapsed_time) = engine.evaluate_sv(model, data_loader_val_sv)
            stats_reduced_sv = get_score_nx_ta_mft(pro_preds, pro_targets, elapsed_time, args.distributed)

            if utils.get_rank() == 0:
                logger.info("Sub-volume Eval:")
                stats_message = eval_utils.get_stats_message_ta_mft(stats_reduced_sv, averages_only=True)
                print(stats_message)

        if args.volume_eval and (not epoch % args.val_interval) or epoch == (args.epochs - 1):
            if not epoch:
                args.batch_size_per_sample = args.batch_size
                logger.info("sub_vol batch_size_per_sample: ", args.batch_size_per_sample)

            preds, targets, sample_ids, samples, masks, elapsed_time = engine.evaluate(model, data_loader_val)
            fil_preds = [eval_utils.merge_duplicate_branches(pred, position_threshold=0.3, min_voxel_threshold=1.5,
                                                             duplicate_ratio=0.2, passes=1) for pred in preds]

            # Compute Acc, Recall and F1
            stats_reduced_fil_mft = get_score_nx_ta_mft(fil_preds, targets, elapsed_time, args.distributed,
                                                        thresholds=args.f1_thresholds)

            pred_dict = {'preds': preds, 'filtered_preds': fil_preds, 'targets': targets, 'target_ids': sample_ids,
                         'elapsed_time': elapsed_time, 'stats_reduced': stats_reduced_fil_mft}
            torch.save(pred_dict, (val_res_dir / f"val_res_ep_{epoch_scaled}.pkl"))

            if utils.get_rank() == 0:
                logger.info("Full Volume Eval Radius Aware Metrics (Filtered):")
                stats_message_fil_mft = eval_utils.get_stats_message_ta_mft(stats_reduced_fil_mft, averages_only=False)
                print(stats_message_fil_mft)

                if stats_reduced_fil_mft['avg_scores'][3] > best_score:
                    best_score = stats_reduced_fil_mft['avg_scores'][3]
                    new_best = True

        if utils.get_rank() == 0 and args.output_dir:
            checkpoint_paths = []
            if args.volume_eval:
                if new_best:
                    checkpoint_paths += [output_dir / f"checkpoint_best.pth"]
                    new_best = False

            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_score': best_score}
                utils.save_on_master(save_dict, checkpoint_path)

    logger.info("Training Finished!")

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    train(args)
