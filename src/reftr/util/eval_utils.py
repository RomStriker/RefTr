import numpy as np
import torch
import src.reftr.util.misc as utils
from scipy.spatial import cKDTree
import networkx as nx
from itertools import groupby
from collections import defaultdict, deque
from scipy.spatial import KDTree
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from concurrent.futures import ProcessPoolExecutor


def calculate_scores(gt_points, pred_points, gt_radii, pred_radii, threshold=0.3, min_threshold=1.5):
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    dis_gt2pred, dis_gt2pred_idxs = pred_tree.query(gt_points, k=1)
    dis_pred2gt, dis_pred2gt_idxs = gt_tree.query(pred_points, k=1)

    min_distances = {}
    for i in range(len(dis_gt2pred_idxs)):
        tgt_idx = i
        pred_idx = dis_gt2pred_idxs[i]
        distance = dis_gt2pred[i]
        if pred_idx not in min_distances or distance < min_distances[pred_idx][0]:
            min_distances[pred_idx] = (distance, tgt_idx)

    filtered_dis_gt2pred_pred_idxs = [idx for idx, _ in min_distances.items()]
    filtered_dis_gt2pred_tgt_idxs = [min_distances[idx][1] for idx in filtered_dis_gt2pred_pred_idxs]
    filtered_dis_gt2pred = [min_distances[idx][0] for idx in filtered_dis_gt2pred_pred_idxs]

    filtered_dis_gt2pred_pred_idxs = np.array(filtered_dis_gt2pred_pred_idxs)
    filtered_dis_gt2pred_tgt_idxs = np.array(filtered_dis_gt2pred_tgt_idxs)
    filtered_dis_gt2pred = np.array(filtered_dis_gt2pred)

    # radius mae
    pred_radii = np.array(pred_radii)
    gt_radii = np.array(gt_radii)
    matched_pred_radii = pred_radii[filtered_dis_gt2pred_pred_idxs]
    matched_tgt_radii = gt_radii[filtered_dis_gt2pred_tgt_idxs]
    radius_mae = np.mean(np.abs(matched_pred_radii - matched_tgt_radii))

    # precision, recall and f1-score:
    filtered_true_positives = [x for x_i, x in enumerate(filtered_dis_gt2pred)
                               if x < max((threshold * matched_tgt_radii[x_i]), min_threshold)]
    recall_nd = len(filtered_true_positives) / len(dis_gt2pred)  # Num true positive / num all ground truths
    prec_nd = len(filtered_true_positives) / len(dis_pred2gt)  # Num true positives / num all predictions
    r_f_nd = 0
    if recall_nd * prec_nd:
        r_f_nd = 2 * recall_nd * prec_nd / (prec_nd + recall_nd)

    return prec_nd, recall_nd, r_f_nd, radius_mae


def group_and_sort_nodes_by_branch(G):
    parsed = sorted(
        ((node.split('-')[0], int(node.split('-')[1]), node) for node in G.nodes),
        key=lambda x: (int(x[0]), x[1])  # sort by branchID numerically, then pointID numerically
    )
    grouped = {
        branch: [n for _, _, n in group]
        for branch, group in groupby(parsed, key=lambda x: x[0])
    }

    return grouped


def get_branch_prec_recall(gt_points, pred_points, gt_radii, threshold=0.3,
                           min_threshold=1.5, output='precision'):
    pred_tree = cKDTree(pred_points)
    dis_gt2pred, dis_gt2pred_idxs = pred_tree.query(gt_points, k=1)

    min_distances = {}
    for i in range(len(dis_gt2pred_idxs)):
        tgt_idx = i
        pred_idx = dis_gt2pred_idxs[i]
        distance = dis_gt2pred[i]
        if pred_idx not in min_distances or distance < min_distances[pred_idx][0]:
            min_distances[pred_idx] = (distance, tgt_idx)

    filtered_dis_gt2pred_pred_idxs = [idx for idx, _ in min_distances.items()]
    filtered_dis_gt2pred_tgt_idxs = [min_distances[idx][1] for idx in filtered_dis_gt2pred_pred_idxs]
    filtered_dis_gt2pred = [min_distances[idx][0] for idx in filtered_dis_gt2pred_pred_idxs]
    filtered_dis_gt2pred_pred_idxs = np.array(filtered_dis_gt2pred_pred_idxs)
    filtered_dis_gt2pred_tgt_idxs = np.array(filtered_dis_gt2pred_tgt_idxs)
    filtered_dis_gt2pred = np.array(filtered_dis_gt2pred)
    gt_radii = np.array(gt_radii)
    matched_tgt_radii = gt_radii[filtered_dis_gt2pred_tgt_idxs]

    tp_mask = filtered_dis_gt2pred < np.maximum(threshold * matched_tgt_radii, min_threshold)
    tp_pred_idxs = filtered_dis_gt2pred_pred_idxs[tp_mask]
    tp_gt_idxs = filtered_dis_gt2pred_tgt_idxs[tp_mask]
    if output == 'precision':
        metric_nd = len(tp_pred_idxs) / len(pred_points)  # Num true positive / num all predictions
    elif output == 'recall':
        metric_nd = len(tp_gt_idxs) / len(gt_points)

    return metric_nd, tp_pred_idxs, tp_gt_idxs


def trajectories_to_depth(G, start, max_depth):
    paths = []
    queue = deque([(start, [start])])  # (current_node, path)
    while queue:
        node, path = queue.popleft()
        depth = len(path) - 1
        if depth == max_depth or len(list(G.successors(node))) == 0:
            paths.append(path)
            continue
        for child in G.successors(node):
            queue.append((child, path + [child]))

    return paths


def get_prec_recall_f1_branches_ft(target, target_positions_list, target_radii_list, target_ids, pred,
                                      pred_positions_list, pred_radii_list, pred_ids, threshold=0.25, min_threshold=2.0,
                                      br_cov_thresholds=(0.5, 0.6, 0.7, 0.8)):

    prec_recall_f1_dict = {}
    for br_cov_th in br_cov_thresholds:
        # Compute Precision
        TPs = set()
        FPs = set()
        target_positions_array = np.concatenate(target_positions_list, axis=0)
        target_radii_array = np.concatenate(target_radii_list, axis=0)
        target_ids_array = np.concatenate(target_ids, axis=0)
        gt_tree = cKDTree(target_positions_array)

        for pred_pos, pred_radii, pred_id in zip(pred_positions_list, pred_radii_list, pred_ids):
            # find the closest three points to the start
            start_pos = pred_pos[0]
            start_radius = pred_radii[0]
            start_radius = 4 if start_radius < 4 else start_radius
            k_pred = min(3, len(target_ids_array))
            distances, dis_pred2gt_idxs = gt_tree.query(start_pos, k=k_pred)
            dis_pred2gt_idxs = dis_pred2gt_idxs[distances <= start_radius]
            valid_idxs = dis_pred2gt_idxs[dis_pred2gt_idxs < len(target_ids_array)]
            closest_gt_ids = target_ids_array[valid_idxs]
            if not len(closest_gt_ids):
                FPs.add(pred_id[0].split('-')[0])
                continue

            # Group pointIDs by branchID
            groups = defaultdict(list)
            for id_ in closest_gt_ids:
                branch, point = id_.split('-')
                groups[branch].append(int(point))
            lowest_ids = [f"{branch}-{min(points)}" for branch, points in groups.items()]

            candidate_trajectories = []
            for id_ in lowest_ids:
                candidate_trajectories += trajectories_to_depth(target, id_, len(pred_id) * 1)

            precision_nds = []
            matched_preds_list = []
            matched_gts_list = []
            for traj in candidate_trajectories:
                traj = np.array(traj)
                indices = np.where(np.isin(target_ids_array, traj))[0]
                tgt_pos = target_positions_array[indices]
                tgt_rad = target_radii_array[indices]
                precision_nd, matched_preds, matched_gts = get_branch_prec_recall(tgt_pos, pred_pos, tgt_rad,
                                                                                  threshold, min_threshold, 'precision')
                precision_nds.append(precision_nd)
                matched_preds_list.append(pred_id[matched_preds])
                matched_gts_list.append(traj[matched_gts])

            max_prec_nd_idx = precision_nds.index(max(precision_nds))
            max_prec_nd = precision_nds[max_prec_nd_idx]
            if max_prec_nd > br_cov_th:
                TPs.add(pred_id[0].split('-')[0])
                max_matched_gts = matched_gts_list[max_prec_nd_idx]
                fil_indices = np.where(~np.isin(target_ids_array, max_matched_gts))[0]
                target_positions_array = target_positions_array[fil_indices]
                target_radii_array = target_radii_array[fil_indices]
                target_ids_array = target_ids_array[fil_indices]
                gt_tree = cKDTree(target_positions_array)
            else:
                FPs.add(pred_id[0].split('-')[0])
        precision = len(TPs) / (len(TPs) + len(FPs))

        # Compute Recall
        TPs = set()
        FNs = set()
        pred_positions_array = np.concatenate(pred_positions_list, axis=0)
        pred_radii_array = np.concatenate(pred_radii_list, axis=0)
        pred_ids_array = np.concatenate(pred_ids, axis=0)
        pred_tree = cKDTree(pred_positions_array)
        for tgt_pos, tgt_rad, tgt_id in zip(target_positions_list, target_radii_list, target_ids):
            # find the closest 3 points to the start
            start_pos = tgt_pos[0]
            start_radius = tgt_rad[0]
            start_radius = 4 if start_radius < 4 else start_radius
            k_gt = min(3, len(pred_ids_array))
            distances, dis_gt2pred_idxs = pred_tree.query(start_pos, k=k_gt)
            dis_gt2pred_idxs = dis_gt2pred_idxs[distances <= start_radius]
            valid_idxs = dis_gt2pred_idxs[dis_gt2pred_idxs < len(pred_ids_array)]
            closest_pred_ids = pred_ids_array[valid_idxs]
            if not len(closest_pred_ids):
                FPs.add(pred_id[0].split('-')[0])
                continue

            # Group pointIDs by branchID
            groups = defaultdict(list)
            for id_ in closest_pred_ids:
                branch, point = id_.split('-')
                groups[branch].append(int(point))
            lowest_ids = [f"{branch}-{min(points)}" for branch, points in groups.items()]

            # get all trajectories starting from this point and going to depth 2 times the length of pred_pos
            candidate_trajectories = []
            for id_ in lowest_ids:
                candidate_trajectories += trajectories_to_depth(pred, id_, len(tgt_id) * 1)

            # compute recall for each candidate trajectory and return the matched points for each trajectory
            recall_nds = []
            matched_preds_list = []
            matched_gts_list = []
            for traj in candidate_trajectories:
                traj = np.array(traj)
                indices = np.where(np.isin(pred_ids_array, traj))[0]
                pred_pos = pred_positions_array[indices]
                recall_nd, matched_preds, matched_gts = get_branch_prec_recall(tgt_pos, pred_pos, tgt_rad,
                                                                                  threshold, min_threshold, 'recall')
                recall_nds.append(recall_nd)
                matched_preds_list.append(traj[matched_preds])
                matched_gts_list.append(tgt_id[matched_gts])

            # get the index of max precision_nd
            max_recall_nd_idx = recall_nds.index(max(recall_nds))
            max_recall_nd = recall_nds[max_recall_nd_idx]
            if max_recall_nd > br_cov_th:
                TPs.add(tgt_id[0].split('-')[0])
                max_matched_preds = matched_preds_list[max_recall_nd_idx]
                fil_indices = np.where(~np.isin(pred_ids_array, max_matched_preds))[0]
                pred_positions_array = pred_positions_array[fil_indices]
                pred_radii_array = pred_radii_array[fil_indices]
                pred_ids_array = pred_ids_array[fil_indices]
                pred_tree = cKDTree(pred_positions_array)
            else:
                FNs.add(tgt_id[0].split('-')[0])
        recall = len(TPs) / (len(TPs) + len(FNs))
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        prec_recall_f1_dict[br_cov_th] = {'recall': recall, 'precision': precision, 'f1': f1}

    recalls = [v['recall'] for v in prec_recall_f1_dict.values()]
    precisions = [v['precision'] for v in prec_recall_f1_dict.values()]
    f1s = [v['f1'] for v in prec_recall_f1_dict.values()]
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_f1 = sum(f1s) / len(f1s)
    prec_recall_f1_dict['avg'] = {'recall': avg_recall, 'precision': avg_precision, 'f1': avg_f1}

    return prec_recall_f1_dict


def calculate_scores_branch_ft(target, target_branches, pred, pred_branches,
                               threshold=0.25, min_threshold=1.5, br_cov_thresholds=(0.5, 0.6, 0.7, 0.8)):
    target_positions_list = [np.array([target.nodes[node]['position'] for node in branch]) for branch in target_branches.values()]
    target_radii_list = [np.array([target.nodes[node]['radius'] for node in branch]) for branch in target_branches.values()]
    target_ids = [np.array([node for node in branch]) for branch in target_branches.values()]
    pred_positions_list = [np.array([pred.nodes[node]['position'] for node in branch]) for branch in pred_branches.values()]
    pred_radii_list = [np.array([pred.nodes[node]['radius'] for node in branch]) for branch in pred_branches.values()]
    pred_ids = [np.array([node for node in branch]) for branch in pred_branches.values()]

    prec_recall_f1_dict = get_prec_recall_f1_branches_ft(target, target_positions_list, target_radii_list,
                                                         target_ids, pred, pred_positions_list, pred_radii_list, pred_ids,
                                                         threshold, min_threshold, br_cov_thresholds)

    avg_prec = prec_recall_f1_dict['avg']['precision']
    avg_recall = prec_recall_f1_dict['avg']['recall']
    avg_f1 = prec_recall_f1_dict['avg']['f1']

    return avg_prec, avg_recall, avg_f1


def compute_score_for_pair_ft(pred, target, threshold, br_cov_thresholds):
    pred_branches = group_and_sort_nodes_by_branch(pred)
    target_branches = group_and_sort_nodes_by_branch(target)
    if len(pred_branches) and len(target_branches):
        return list(calculate_scores_branch_ft(target, target_branches, pred, pred_branches, threshold,
                                               br_cov_thresholds=br_cov_thresholds)) + [len(pred_branches),
                                                                                              len(target_branches)]
    return [0, 0, 0, len(pred_branches), len(target_branches)]


def get_score_nx_branch_ft(preds, targets, dist=False, threshold=0.25,
                           br_cov_thresholds=(0.5, 0.6, 0.7, 0.8)):
    scores_dict = {'total_branches': [],
                   'num_branches': [],
                   'scores': []}

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, (pred, target) in enumerate(zip(preds, targets)):
            future = executor.submit(compute_score_for_pair_ft, pred, target,
                                     threshold, br_cov_thresholds)
            futures.append((i, future))

        for i, future in tqdm(futures, desc="Overall Progress", unit="task", leave=False):
            try:
                result = future.result()
                scores_dict['scores'].append(result[:3])
                scores_dict['num_branches'].append(result[3])
                scores_dict['total_branches'].append(result[4])
            except Exception as e:
                print(f"Error processing future for index {i}: {e}")
                scores_dict['scores'].append([0, 0, 0])
                scores_dict['num_branches'].append(0)
                scores_dict['total_branches'].append(0)

    stats_t = {'num_branches': torch.FloatTensor(scores_dict['num_branches'])}
    stats_t['total_branches'] = torch.FloatTensor(scores_dict['total_branches'])
    stats_t['scores'] = torch.FloatTensor(scores_dict['scores'])

    if dist:
        for key in stats_t:
            stats_t[key] = stats_t[key].to('cuda')
        torch.cuda.synchronize()
        torch.distributed.barrier()
        stats_reduced = utils.gather_stats(stats_t)
    else:
        stats_reduced = stats_t

    stats_reduced['avg_num_branches'] = torch.mean(stats_reduced['num_branches'], dim=0)
    stats_reduced['avg_total_branches'] = torch.mean(stats_reduced['total_branches'], dim=0)
    stats_reduced['avg_scores'] = torch.mean(stats_reduced['scores'], dim=0)

    return stats_reduced


def get_score_nx_ta_branch_mft(preds, targets, dist=False, rad_thresholds=(0.3, 0.4, 0.5, 0.6),
                               br_cov_thresholds=(0.5, 0.6, 0.7, 0.8)):
    # multiple fraction thresholds
    stats_reduced_frac = {}
    for threshold in tqdm(rad_thresholds, desc='Computing scores'):
        stats_reduced_frac[threshold] = get_score_nx_branch_ft(preds, targets, dist, threshold, br_cov_thresholds)

    # compute average of the thresholds
    stats_reduced_frac['avg'] = {}
    for metric in stats_reduced_frac[rad_thresholds[0]]:
        stats_reduced_frac['avg'][metric] = np.mean(np.stack([stats_reduced_frac[th][metric] for th in rad_thresholds]), axis=0)

    return stats_reduced_frac


def get_score_nx_ta_ft(preds, targets, elapsed_time, dist=False, threshold=1.5):
    # extract points list
    data_dict = {'preds_list': [],
                 'targets_list': [],
                 'pred_rad_list': [],
                 'targets_rad_list': []}

    for i, (pred, target) in enumerate(zip(preds, targets)):
        for item in data_dict:
            data_dict[item].append([])

        # fill data_dict with pred info
        data_dict['preds_list'][i] = list(nx.get_node_attributes(pred, 'position').values())
        data_dict['pred_rad_list'][i] = list(nx.get_node_attributes(pred, 'radius').values())

        # fill data_dict with target info
        data_dict['targets_list'][i] = list(nx.get_node_attributes(target, 'position').values())
        data_dict['targets_rad_list'][i] = list(nx.get_node_attributes(target, 'radius').values())

    scores_dict = {'total_points': [],
                   'num_points': [],
                   'scores': [], }

    # calculate_scores
    for i in range(len(data_dict['preds_list'])):
        pred_points = data_dict['preds_list'][i]
        target_points = data_dict['targets_list'][i]
        pred_rad = data_dict['pred_rad_list'][i]
        target_rad = data_dict['targets_rad_list'][i]
        # if we have both target and pred points, we compute the score normally
        if len(pred_points) and len(target_points):
            scores_dict['scores'].append(list(calculate_scores(target_points, pred_points,
                                                               target_rad, pred_rad, threshold=threshold)))

        scores_dict['num_points'].append(len(pred_points))
        scores_dict['total_points'].append(len(target_points))

    # tensor dict for dist training (new)
    stats_t = {'num_points': torch.FloatTensor(scores_dict['num_points'])}
    stats_t['scores'] = torch.FloatTensor(scores_dict['scores'])
    stats_t['total_points'] = torch.FloatTensor(scores_dict['total_points'])
    if elapsed_time is not None:
        stats_t['elapsed_time'] = torch.FloatTensor(elapsed_time)
    else:
        stats_t['elapsed_time'] = torch.zeros(len(preds), dtype=torch.float32)

    if dist:
        for key in stats_t:
            stats_t[key] = stats_t[key].to('cuda')
        torch.cuda.synchronize()
        torch.distributed.barrier()
        stats_reduced = utils.gather_stats(stats_t)
    else:
        stats_reduced = stats_t

    stats_reduced['avg_num_points'] = torch.mean(stats_reduced['num_points'], dim=0)
    stats_reduced['avg_scores'] = torch.mean(stats_reduced['scores'], dim=0)
    stats_reduced['avg_total_points'] = torch.mean(stats_reduced['total_points'], dim=0)
    stats_reduced['avg_elapsed_time'] = torch.mean(stats_reduced['elapsed_time'], dim=0)

    for key in stats_reduced:
        stats_reduced[key] = stats_reduced[key].cpu().numpy()

    return stats_reduced


def get_score_nx_ta_mft(preds, targets, elapsed_time, dist=False, thresholds=(0.3, 0.4, 0.5, 0.6)):
    # multiple fraction thresholds
    stats_reduced_frac = {}
    for threshold in tqdm(thresholds, desc='Computing scores'):
        stats_reduced_frac[threshold] = get_score_nx_ta_ft(preds, targets, elapsed_time, dist, threshold)

    # compute average of the thresholds
    stats_reduced_frac['avg'] = {}
    for metric in stats_reduced_frac[thresholds[0]]:
        stats_reduced_frac['avg'][metric] = np.mean(np.stack([stats_reduced_frac[th][metric] for th in thresholds]), axis=0)

    return stats_reduced_frac


def get_stats_message_ta_mft(stats_all, averages_only=False, thresholds=['avg'], ids=None):
    message = ""
    for key in stats_all:
        if key not in thresholds:
            continue
        stats = stats_all[key]
        message += "\n Threshold: " + str(key) + "\n"
        message += "\nAll points\n"
        if not averages_only:
            for i, score in enumerate(stats['scores']):
                id = ids[i] if ids is not None else i
                message += (f"ID: {id} \t | \t Precision: {score[0].item():.3f} \t "
                            f"| \t Recall: {score[1].item():.3f} \t "
                            f"| \t F1: {score[2].item():.3f} \n "
                            f"| \t Radius MAE: {score[3].item():.3f} \n "
                            f"| \t Time taken: {stats['elapsed_time'][i].item():.3f}"
                            f"| \t Num Points: {stats['num_points'][i].item():.3f} \t "
                            f"| \t Tot Points: {int(stats['total_points'][i].item())} \t "
                            f"| \t Time taken: {stats['elapsed_time'][i].item():.3f}\n")

        message += "Average\n"
        message += (f"| \t Precision: {stats['avg_scores'][0]:.3f} \t "
                    f"| \t Recall: {stats['avg_scores'][1]:.3f} \t "
                    f"| \t F1: {stats['avg_scores'][2]:.3f} \n "
                    f"| \t Radius MAE: {stats['avg_scores'][3]:.3f} \n "
                    f"| \t Avg Points: {int(stats['avg_num_points'].item())} \t "
                    f"| \t Tot Points: {int(stats['avg_total_points'].item())} \t "
                    f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

    return message


def get_stats_message_branch_mft(stats_all, averages_only=False, thresholds=['avg'], ids=None):
    message = ""
    for key in stats_all:
        if key not in thresholds:
            continue
        stats = stats_all[key]
        message += "\n Threshold: " + str(key) + "\n"
        message += "\nAll points\n"
        if not averages_only:
            for i, score in enumerate(stats['scores']):
                id = ids[i] if ids is not None else i
                message += (f"ID: {id} \t | \t Precision: {score[0].item():.3f} \t "
                            f"| \t Recall: {score[1].item():.3f} \t "
                            f"| \t F1: {score[2].item():.3f} \n ")
                message += (f"| \t Num Branches: {stats['num_branches'][i].item():.3f} \t "
                            f"| \t Tot Branches: {int(stats['total_branches'][i].item())}\n")

        message += "Average\n"
        message += (f"Precision: {stats['avg_scores'][0]:.3f} \t "
                    f"| \t Recall: {stats['avg_scores'][1]:.3f} \t "
                    f"| \t F1: {stats['avg_scores'][2]:.3f} \n ")
        message += (f"| \t Avg Branches: {int(stats['avg_num_branches'].item())} \t "
                    f"| \t Tot Branches: {int(stats['avg_total_branches'].item())}\n")

    return message


def merge_tuples_to_sets(tuples):
    G = nx.Graph()
    G.add_edges_from(tuples)
    return list(nx.connected_components(G))


def weighted_mean_by_depth(positions, radii, depths, gamma=2.0, minimum=False):
    """
    Compute weighted mean where shallow depths are favored sharply.

    Args:
        positions (List[np.array]): List of 2D or 3D coordinates.
        radii (List[float]): Corresponding radii values.
        depths (List[float]): Corresponding depth values.
        gamma (float): Exponent controlling sharpness of weight drop-off.
        minimum (bool): If True, return the minimum depth position instead of weighted mean.
    Returns:
        np.array: Weighted average position.
    """
    if minimum:
        min_depth_idx = np.argmin(depths)
        position = positions[min_depth_idx]
        radius = radii[min_depth_idx]
        return position, radius

    norm_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths) + 1e-6)
    weights = (1.0 - norm_depths) ** gamma
    weights /= np.sum(weights)
    weighted_mean_position = np.average(positions, axis=0, weights=weights)
    weighted_mean_radius = np.average(radii, weights=weights)

    return weighted_mean_position, weighted_mean_radius


def merge_sets_of_nodes(G, sets_to_merge):
    """
    Merge groups of nodes in a directed graph.
    Args:
        G (nx.DiGraph): The input graph.
        sets_to_merge (list[set]): List of node sets to merge (e.g., [{1, 2}, {3, 4, 5}]).
    Returns:
        nx.DiGraph: Graph with merged nodes.
    """
    if 'global_depth' not in G.nodes['0-0']:
        G = add_depth_to_nodes(G)

    for nodes in sets_to_merge:
        nodes = list(nodes)
        depths = np.array([G.nodes[node]['global_depth'] for node in nodes])
        target_idx = np.argmin(depths)
        target = nodes[target_idx]
        other_nodes = [val for i, val in enumerate(nodes) if i != target_idx]
        all_preds = set()
        all_succs = set()
        for node in other_nodes:
            all_preds.update(G.predecessors(node))
            all_succs.update(G.successors(node))

        # Reconnect predecessors and successors
        for pred in all_preds:
            G.add_edge(pred, target)
        for succ in all_succs:
            G.add_edge(target, succ)

        positions = np.array([G.nodes[node]['position'] for node in nodes])
        radii = np.array([G.nodes[node]['radius'] for node in nodes])
        weighted_mean_position, weighted_mean_radius = weighted_mean_by_depth(positions, radii,
                                                                              depths, gamma=2.0, minimum=False)
        G.nodes[target]['position'] = weighted_mean_position
        G.nodes[target]['radius'] = weighted_mean_radius

        G.remove_nodes_from(other_nodes)

    return G


def add_depth_to_nodes(G):
    """
    Add depth attribute to each node in the graph.
    Depth is defined as the number of edges from the root node.
    """
    for i, level in enumerate(nx.bfs_layers(G, '0-0')):
        for node in level:
            G.nodes[node]['global_depth'] = i

    return G


def rename_node_names(G, root):
    renamed_tree = nx.DiGraph()
    st_br_id = 0
    old_node_id_to_new = {}
    for level in nx.bfs_layers(G, root):
        for node in level:
            if node == root:
                node_name = str(st_br_id) + "-" + str(0)
                renamed_tree.add_node(node_name, **G.nodes[node])
                old_node_id_to_new[node] = node_name
            else:
                parent = list(G.predecessors(node))[0]
                parent_name = old_node_id_to_new[parent]
                if G.out_degree(parent) > 1:
                    st_br_id += 1
                    node_name = str(st_br_id) + "-" + str(0)
                    renamed_tree.add_node(node_name, **G.nodes[node])
                    old_node_id_to_new[node] = node_name
                else:
                    parent_br = int(parent_name.split("-")[0])
                    parent_pt = int(parent_name.split("-")[1])
                    node_name = str(parent_br) + "-" + str(parent_pt + 1)
                    renamed_tree.add_node(node_name, **G.nodes[node])
                    old_node_id_to_new[node] = node_name
                renamed_tree.add_edge(parent_name, node_name)

    return renamed_tree


def get_branches(G):
    nodes = list(G.nodes)
    branch_names = list(set([node.split("-")[0] for node in nodes]))
    branch_start_end_points = {}
    for branch in branch_names:
        start_node_name = branch + "-0"
        parent_node = list(G.predecessors(start_node_name))
        if len(parent_node) == 0:
            parent_node = branch + "-0"
        else:
            parent_node = parent_node[0]
        branch_nodes = [node for node in nodes if node.split('-')[0] == branch]
        max_point_id = max([int(node.split("-")[1]) for node in branch_nodes])
        end_node_name = branch + "-" + str(max_point_id)
        branch_start_end_points[branch] = (parent_node, end_node_name)
    return branch_start_end_points


def smooth_edges(G, sigma=1):
    branches = get_branches(G)
    branch_segments = [nx.shortest_path(G, branch[0], branch[1]) for branch in branches.values()]
    for segment in branch_segments:
        path_pos = np.array([G.nodes[node]["position"] for node in segment])
        path_pos_smooth = gaussian_filter1d(path_pos, sigma=sigma, axis=0, mode="nearest")
        for node, pos in zip(segment[1:-1], path_pos_smooth[1:-1]):
            G.nodes[node]["position"] = pos
    return G


def add_edge_lengths(graph):
    """
    Add the length of each edge to the graph.

    Parameters:
    - graph: networkx.DiGraph with 'position' attribute for each node.

    Returns:
    - networkx.DiGraph with 'length' attribute for each edge.
    """
    for u, v in graph.edges:
        pos_u = np.array(graph.nodes[u]['position'])
        pos_v = np.array(graph.nodes[v]['position'])
        graph.nodes[v]['edge_length'] = np.linalg.norm(pos_u - pos_v)

    if 'edge_length' not in graph.nodes['0-0']:
        graph.nodes['0-0']['edge_length'] = 0

    return graph


def sample_equidistant_points(positions, radii, n_points=None, points_dist=None,
                              smooth=True, smooth_override=True):
    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    avg_segment_distance = np.mean(segment_distances)
    if (segment_distances[0] > 2 * avg_segment_distance
            or segment_distances[-1] > 2 * avg_segment_distance) and smooth_override:
        smooth = False

    if len(positions) > 2 and smooth:
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_distances)])
        total_length = cumulative_distances[-1]
        if points_dist is not None:
            n_points = max(round(total_length / points_dist) + 1, 2)
        assert n_points is not None, "Either n_points or points_dist should be provided"
        target_distances = np.linspace(0, total_length, n_points)
        splines_positions = [CubicSpline(cumulative_distances, positions[:, i]) for i in range(3)]
        positions = np.column_stack([spline(target_distances) for spline in splines_positions])
        radii = np.interp(target_distances, cumulative_distances, radii)

    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_distances)])
    total_length = cumulative_distances[-1]
    if points_dist is not None:
        n_points = max(round(total_length / points_dist) + 1, 2)
    assert n_points is not None, "Either n_points or points_dist should be provided"
    target_distances = np.linspace(0, total_length, n_points)

    sampled_positions = np.empty((n_points, 3))
    for i in range(3):
        sampled_positions[:, i] = np.interp(target_distances, cumulative_distances, positions[:, i])
    sampled_radii = np.interp(target_distances, cumulative_distances, radii)

    return sampled_positions, sampled_radii


def get_resampled_graph(G, n_points=None, points_dist=None, smooth=True, remove_small=True):
    """
    Resample the graph to have equidistant points
    """
    G_resampled = nx.DiGraph()
    if remove_small:
        G = merge_remove_small_branches(G, min_length=0.5, root='0-0')
    branches = get_branches(G)

    for branch in branches:
        start_node, end_node = branches[branch]
        if start_node == end_node:
            assert start_node == end_node == "0-0", "There should be no self loops in the graph"
            G_resampled.add_node("0-0", position=G.nodes["0-0"]["position"],
                                 radius=G.nodes["0-0"]["radius"])
            continue

        shortest_path = nx.shortest_path(G, start_node, end_node)
        positions = np.array([G.nodes[node]["position"] for node in shortest_path])
        radii = np.array([G.nodes[node]["radius"] for node in shortest_path])
        if len(positions) > 1:
            resampled_positions, resampled_radii = sample_equidistant_points(positions, radii,
                                                                             n_points=n_points, points_dist=points_dist, smooth=smooth)
        else:
            raise ValueError("Branch has less than 2 points!")

        if branch == '0':
            resampled_positions = np.concatenate([np.array([[0.0, 0.0, 0.0]]), resampled_positions])
            resampled_radii = np.concatenate([np.array([0]), resampled_radii])

        point_id = 0
        G_resampled.add_node(branch + '-' + str(point_id),
                             position=resampled_positions[1],
                             radius=resampled_radii[1])
        for (position, radius) in zip(resampled_positions[2:], resampled_radii[2:]):
            point_id += 1
            node_name = branch + "-" + str(point_id)
            parent_pos = G_resampled.nodes[branch + "-" + str(point_id - 1)]["position"]
            edge_length = np.linalg.norm(position - parent_pos)
            G_resampled.add_node(node_name, position=position, radius=radius,
                                 edge_length=edge_length)
            G_resampled.add_edge(branch + "-" + str(point_id - 1), node_name)

    new_branches = get_branches(G_resampled)
    for branch in new_branches:
        if branch == '0':
            continue
        parent_br = branches[branch][0].split("-")[0]
        parent_end = new_branches[parent_br][1]
        branch_start = branch + "-0"
        parent_pos = G_resampled.nodes[parent_end]["position"]
        branch_start_pos = G_resampled.nodes[branch_start]["position"]
        edge_length = np.linalg.norm(branch_start_pos - parent_pos)
        G_resampled.add_edge(parent_end, branch_start)
        G_resampled.nodes[branch_start]["edge_length"] = edge_length

    G_resampled.nodes['0-0']["edge_length"] = 0
    if remove_small:
        G_resampled = merge_remove_small_edges(G_resampled, min_length=0.5)

    return G_resampled


def merge_remove_small_edges(G, min_length=0.5):
    """
    Merge small edges in the graph G based on the specified minimum length.
    """
    init_num_nodes = len(G.nodes)
    edges = list(G.edges)

    for u, v in edges:
        if u not in G.nodes or v not in G.nodes:
            continue

        edge_length = G.nodes[v]["edge_length"]

        if edge_length >= min_length:
            continue

        grandparent_nodes = list(G.predecessors(u))
        children = list(G.successors(v))

        if len(grandparent_nodes):
            grandparent = grandparent_nodes[0]
            parent_edge_length = G.nodes[u]["edge_length"]

            if edge_length + parent_edge_length < 1.5:
                G.add_edge(grandparent, v)
                G.nodes[v]["edge_length"] = edge_length + parent_edge_length
                G.remove_node(u)
                print(f"Merged edge {u} -> {v} with length {edge_length}")
                continue

        if len(children):
            children_edge_lengths = np.array([G.nodes[child]["edge_length"] for child in children])
            merged_edge_lengths = children_edge_lengths + edge_length

            if (merged_edge_lengths < 1.5).all():
                for c_i, child in enumerate(children):
                    G.add_edge(u, child)
                    G.nodes[child]["edge_length"] = merged_edge_lengths[c_i]
                G.remove_node(v)
                print(f"Merged edge {u} -> {v} with length {edge_length}")
                continue

        print(f"Edge {u} -> {v} with length {edge_length} could not be merged")
    final_num_nodes = len(G.nodes)
    if init_num_nodes != final_num_nodes:
        G = merge_remove_small_edges(G, min_length)

    return G


def merge_remove_small_branches(G, min_length=0.5, root='0-0'):
    """
    Merge small intermediate branches to the parent branch
    """
    init_num_nodes = len(G.nodes)
    branches = get_branches(G)
    for branch in branches:
        start_node, end_node = branches[branch]
        if start_node not in G.nodes or end_node not in G.nodes:
            continue

        if start_node == end_node:
            assert start_node == end_node == root, "There should be no self loops in the graph"
            continue

        shortest_path = nx.shortest_path(G, start_node, end_node)
        branch_length = sum([G.nodes[node]["edge_length"] for node in shortest_path[1:]])
        if branch_length < min_length:
            if G.out_degree(end_node) == 0:
                G.remove_nodes_from(shortest_path[1:])
                continue
            else:
                successors = list(G.successors(end_node))
                for succ in successors:
                    succ_pos = G.nodes[succ]["position"]
                    start_pos = G.nodes[start_node]["position"]
                    edge_length = np.linalg.norm(succ_pos - start_pos)
                    G.add_edge(succ, start_node)
                    G.nodes[succ]["edge_length"] = edge_length
                G.remove_nodes_from(shortest_path[1:])
                continue
    G = rename_node_names(G, root)
    final_num_nodes = len(G.nodes)
    if init_num_nodes != final_num_nodes:
        G = merge_remove_small_branches(G, min_length, root)

    return G


def remove_small_branches(G, min_length=4):
    """
    Remove small branches from the graph
    """
    removed = False
    out_degrees = dict(G.out_degree())
    end_nodes = [node for node in G.nodes if out_degrees[node] == 0]
    nodes_to_remove = []
    skip = False
    for end_node in end_nodes:
        parent_node = end_node
        while out_degrees[parent_node] <= 1:
            current_node = parent_node
            parent_node = list(G.predecessors(parent_node))
            if len(parent_node) == 0:
                skip = True
                break
            else:
                parent_node = parent_node[0]
        if skip:
            skip = False
            continue
        parent_node = current_node
        branch_nodes = nx.shortest_path(G, parent_node, end_node)
        num_points = len(branch_nodes)
        if num_points < min_length:
            nodes_to_remove.extend(nx.shortest_path(G, parent_node, end_node))
            removed = True
    G.remove_nodes_from(nodes_to_remove)

    if removed:
        G = remove_small_branches(G, min_length)

    return G


def clean_tree(G, root_node):
    G = rename_node_names(G, root_node)
    G = smooth_edges(G, sigma=2)
    G = add_edge_lengths(G)
    G = get_resampled_graph(G, points_dist=1)
    G = rename_node_names(G, root_node)
    G = remove_small_branches(G, min_length=5)
    G = rename_node_names(G, root_node)

    return G


def merge_duplicate_branches(G, position_threshold=0.6, min_voxel_threshold=1.5,
                             duplicate_ratio=0.4, passes=1):
    """
    Merge duplicate branches in the tree
    Modifies the graph in-place
    """
    for _ in range(passes):
        if G.number_of_nodes() <= 1 or len([node for node in G.nodes if G.out_degree(node) > 1]) == 0:
            return G
        G = G.copy()
        root_node = '0-0'
        visited_nodes = []
        visited_pos = []

        branch_stats = defaultdict(lambda: {'total': [], 'flagged': []})
        num_nodes = len(G.nodes)
        curr_branch_id = '0'
        curr_branch_nodes = [root_node]
        curr_branch_positions = [np.array(G.nodes[root_node]['position'])]
        kdtree = None

        # traverse the graph in Pre-order
        for node in nx.dfs_preorder_nodes(G, source='0-0'):
            branch_id = node.split('-')[0]
            if branch_id != curr_branch_id:
                curr_branch_id = branch_id
                visited_nodes.extend(curr_branch_nodes)
                visited_pos.extend(curr_branch_positions)
                curr_branch_nodes = []
                curr_branch_positions = []
                kdtree = KDTree(visited_pos)

            print(f"\rProcessing node ({len(visited_nodes) + len (curr_branch_nodes)}/{num_nodes}): {node} ", end='')
            branch_stats[branch_id]['total'].append(node)
            position = np.array(G.nodes[node]['position'])
            curr_branch_nodes.append(node)
            curr_branch_positions.append(position)

            if kdtree:
                radius = np.array(G.nodes[node]['radius'])
                threshold = max(min_voxel_threshold, position_threshold * radius)
                distance, index = kdtree.query(
                    position,
                    k=1,
                    distance_upper_bound=threshold,
                )
                if distance != np.inf:
                    neighbor_node = visited_nodes[index]
                    branch_stats[branch_id]['flagged'].append((node, neighbor_node))

        # Identify branches to merge based on duplicate ratio
        branches_to_merge = [
            branch_id for branch_id, stats in branch_stats.items()
            if len(stats['flagged']) / len(stats['total']) >= duplicate_ratio
        ]

        # from branches to merge, merge duplicate nodes
        all_flagged_nodes = [node for branch in branches_to_merge
                             for node in branch_stats[branch]['flagged']]
        nodes_to_merge = merge_tuples_to_sets(all_flagged_nodes)
        G = merge_sets_of_nodes(G, nodes_to_merge)

        # find nodes with in-degree > 1 (cycles)
        cycle_nodes = [node for node in G.nodes if G.in_degree(node) > 1]

        # for each cycle node, find the cycle
        for i, node in enumerate(cycle_nodes):
            print(f"\rProcessing cycle node ({i}/{len(cycle_nodes)}): {node} ", end='')
            if node not in G.nodes:
                continue
            predecessors = list(G.predecessors(node))
            predecessors_paths = {pred: nx.shortest_path_length(G, source=root_node, target=pred)
                                  for pred in predecessors}
            sorted_predecessors = sorted(predecessors_paths.keys(), key=lambda x: predecessors_paths[x])
            preds_edges_to_remove = sorted_predecessors[1:]
            for pred in preds_edges_to_remove:
                G.remove_edge(pred, node)

        if not nx.is_weakly_connected(G):
            raise ValueError("Graph has more than one connected component after removing duplicate branches.")

        # check if the root has any predecessors, if yes, remove the edge to the root
        if len(list(G.predecessors(root_node))) > 0:
            for pred in list(G.predecessors(root_node)):
                G.remove_edge(pred, root_node)

        if not nx.is_weakly_connected(G):
            raise ValueError("Graph has more than one connected component after removing duplicate branches.")

        assert len(G.nodes) == len(G.edges) + 1, "Graph is not a tree after removing duplicate branches."

        G = clean_tree(G)

    return G

