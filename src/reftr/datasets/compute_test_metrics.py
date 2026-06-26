import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.reftr.util.eval_utils import (
                                             get_score_nx_ta_mft,
                                             get_stats_message_ta_mft,
                                             get_stats_message_branch_mft,
                                             get_score_nx_ta_branch_mft,)
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(pred_file, gt_dir, filtered):
    sample_id = os.path.basename(pred_file).split('.')[0]
    target_file = Path(gt_dir) / (sample_id + '.pickle')

    with open(target_file, 'rb') as handle:
        target = pickle.load(handle)['networkx'][0]
    output = torch.load(pred_file)
    if filtered:
        pred = output['filtered_preds'][0]
    else:
        pred = output['preds'][0]

    return sample_id, pred, target


def load_graphs(pred_dir, gt_dir, dataset='atm22', filtered=False):
    pred_files = sorted(list(pred_dir.glob("*.pickle")) + list(pred_dir.glob("*.pkl")))
    dataset_sizes = {'atm22': 60, 'parse2022': 20, 'syntrx': 100}

    assert len(pred_files) == dataset_sizes[dataset], (f"Expected {dataset_sizes[dataset]} "
                                                       f"samples but found {len(pred_files)}")
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, str(pf), str(gt_dir), filtered): pf
            for pf in pred_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading graphs (parallel)"):
            sample_id, pred, target = future.result()
            results.append((sample_id, pred, target))

    # Sort results to maintain order (optional)
    results.sort(key=lambda x: x[0])  # Assuming sample_ids are consistent

    sample_ids = [r[0] for r in results]
    preds = [r[1] for r in results]
    targets = [r[2] for r in results]

    return preds, targets, sample_ids


if __name__ == '__main__':
    filtered = True
    dataset = 'atm22'
    pred_dirs = 'PATH/TO/PREDICTION/DIR/LIST'
    mask_dir = Path(f"/data/{dataset}/masks_test")
    gt_dir = Path(f"/data/{dataset}/annots_test")

    f1_mft = True
    branch_f1_mft = True
    if f1_mft:
        rad_thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        prec_list = []
        recall_list = []
        f1_list = []
        radii_list = []
        for pred_dir in pred_dirs:
            preds, targets, sample_ids = load_graphs(pred_dir, gt_dir, dataset, filtered)
            # compute precision, recall and f1 score
            stats = get_score_nx_ta_mft(preds, targets, None, False,
                                        thresholds=rad_thresholds)
            prec_list.append(stats['avg']['avg_scores'][0])
            recall_list.append(stats['avg']['avg_scores'][1])
            f1_list.append(stats['avg']['avg_scores'][2])
            radii_list.append(stats['avg']['avg_scores'][3])
            print(f"precision: {stats['avg']['avg_scores'][0] * 100:.2f}, "
                  f"recall: {stats['avg']['avg_scores'][1] * 100:.2f}, "
                  f"f1: {stats['avg']['avg_scores'][2] * 100:.2f}")
            stats_message = get_stats_message_ta_mft(stats, averages_only=False, thresholds=['avg'],
                                                     ids=sample_ids)
            print(stats_message)

        prec_list = np.array(prec_list)
        mean_prec = np.mean(prec_list)
        std_prec = np.std(prec_list, ddof=1)
        recall_list = np.array(recall_list)
        mean_recall = np.mean(recall_list)
        std_recall = np.std(recall_list, ddof=1)
        f1_list = np.array(f1_list)
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list, ddof=1)
        radii_list = np.array(radii_list)
        mean_radii = np.mean(radii_list)
        std_radii = np.std(radii_list, ddof=1)

        print(f"Precision: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Radii: {mean_radii:.4f} ± {std_radii:.4f}")

    if branch_f1_mft:
        rad_thresholds = [0.5]
        br_cov_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        prec_list = []
        recall_list = []
        f1_list = []
        for pred_dir in pred_dirs:
            preds, targets, sample_ids = load_graphs(pred_dir, gt_dir, dataset, filtered)
            # compute precision, recall and f1 score
            stats = get_score_nx_ta_branch_mft(preds, targets, False, rad_thresholds=rad_thresholds,
                                               br_cov_thresholds=br_cov_thresholds)
            prec_list.append(stats['avg']['avg_scores'][0])
            recall_list.append(stats['avg']['avg_scores'][1])
            f1_list.append(stats['avg']['avg_scores'][2])
            stats_message = get_stats_message_branch_mft(stats, averages_only=False,
                                                               thresholds=['avg'], ids=sample_ids)

        prec_list = np.array(prec_list)
        mean_prec = np.mean(prec_list)
        std_prec = np.std(prec_list, ddof=1)
        recall_list = np.array(recall_list)
        mean_recall = np.mean(recall_list)
        std_recall = np.std(recall_list, ddof=1)
        f1_list = np.array(f1_list)
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list, ddof=1)

        print(f"Precision: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
