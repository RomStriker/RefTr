"""
Train and eval functions used in main.py
"""
import os
import math
import sys
import numpy as np
from typing import Iterable
from collections import defaultdict
import time
import networkx as nx
from tqdm import tqdm
import torch
from src.reftr.util import misc as utils
from monai.transforms import (
    Compose,
    LoadImage,
    ThresholdIntensity)
from src.reftr.datasets.transforms_tra import (
    CropAndPad,
    NormalizeIntensity,
    ComputeImageMin,
    LoadAnnotPickle)


class Engine:
    def __init__(self, args, logger, device):
        self.args = args
        self.logger = logger
        self.device = device

    @staticmethod
    def target_dict_to_device(dictionary, device):
        """
        Send targets to appropriate desired device.
        """
        for target in dictionary:
            for key, value in target.items():
                if key in ['target_branches', 'target_radii']:
                    for b_i, branch in enumerate(value):
                        value[b_i] = branch.to(device)
                elif key in ['divergence_matrix', 'end_position']:
                    target[key] = value.to(device)
                else:
                    pass
        return dictionary

    def train_one_epoch(self, model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                        scaler: torch.cuda.amp.GradScaler, epoch):
        """
        Training function for one epoch
        """
        model.train()
        criterion.train()
        skip_lr_step = False
        epoch_metrics = {'scaled_losses': 0, 'unscaled_losses': 0, 'total_loss': 0, }
        for i, batch in enumerate(data_loader):
            for j, sub_batch in enumerate(batch):
                node = 'selected_node' if j == 0 else 'pair_node'
                losses = []
                batch_metrics = {'scaled_losses': [], 'unscaled_losses': []}
                sample_imgs, sample_past_trs, targets = (sub_batch["image"], sub_batch["past_tr"], sub_batch["label"])

                if sample_imgs.device != self.device:
                    sample_imgs = sample_imgs.to(self.device)
                if sample_past_trs is not None:
                    sample_past_trs = sample_past_trs.to(self.device)
                targets = self.target_dict_to_device(targets, self.device)

                # prev_step_info
                if node == 'pair_node':
                    # update prev step info for new volume patch
                    prev_step_info['first_step'] = True
                    prev_step_info['memory'] = None
                    prev_step_info['pos'] = None
                    prev_step_info['out'] = None
                    prev_step_info['prev_out'] = None
                    prev_step_info['node_type'] = node
                    prev_step_info['prev_matched_targets'] = prev_step_info['matched_targets']
                    prev_step_info['matched_targets'] = None
                    prev_step_info['parent_target_index'] = [tgt['parent_target_index'] for tgt in targets]

                    # get index of the parent node of this volume patch from the previous volume patch tree
                    parent_pred_index = []
                    for pt_i, pt_idx in enumerate(prev_step_info['parent_target_index']):
                        prev_matched_indices = prev_step_info['prev_matched_targets']['indices'][pt_i]
                        matched_tgt_indices = torch.nonzero(prev_matched_indices[0] == pt_idx, as_tuple=True)[0]
                        matched_pred_indices = prev_matched_indices[1][matched_tgt_indices]
                        rand_idx = torch.randint(0, matched_pred_indices.size(0), (1,)).item()
                        parent_pred_index.append(matched_pred_indices[rand_idx].item())
                    prev_step_info['parent_pred_index'] = parent_pred_index
                    prev_step_info['level'] = j
                    prev_step_info['prev_hs_without_norm'] = prev_step_info['hs_without_norm'].detach()
                    prev_step_info['hs_without_norm'] = None

                else:
                    prev_step_info = {"first_step": True,
                                      "out": None,
                                      "memory": None,
                                      "pos": None,
                                      "hs_without_norm": None,
                                      "node_type": 'selected_node',
                                      "level": j,
                                      "parent_pred_index": [0] * self.args.batch_size}

                # predicting for the number of steps in our sequence
                for step in range(self.args.num_steps):
                    with torch.cuda.amp.autocast(enabled=self.args.amp):
                        norm_step = torch.tensor(step / (self.args.num_steps - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                        prev_step_info = model(sample_imgs, sample_past_trs, prev_step_info, norm_step, epoch)

                        # compute loss
                        loss_dict = criterion(prev_step_info, targets)
                        weight_dict = criterion.weight_dict
                        losses.append(sum(loss_dict[k] * weight_dict[k]
                                          for k in loss_dict.keys() if k in weight_dict))

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = utils.reduce_dict(loss_dict)
                    loss_dict_reduced_unscaled = {
                        f'{k}_unscaled': v for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
                    loss_dict_reduced_scaled = {
                        k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                    batch_metrics['scaled_losses'].append(losses_reduced_scaled.item())
                    batch_metrics['unscaled_losses'].append(losses_reduced_unscaled.item())
                    # Add all unscaled losses and metrics individually to the batch_metrics
                    for item in loss_dict_reduced:
                        if item in batch_metrics:
                            batch_metrics[item].append(loss_dict_reduced[item].item())
                        else:
                            batch_metrics.update({item: [loss_dict_reduced[item].item()]})
                    # Add all scaled losses and metrics individually to the batch_metrics
                    for item in loss_dict_reduced_scaled:
                        if item in batch_metrics:
                            batch_metrics[item].append(loss_dict_reduced_scaled[item].item())
                        else:
                            batch_metrics.update({item: [loss_dict_reduced_scaled[item].item()]})

                for item in batch_metrics:
                    if item in epoch_metrics:
                        epoch_metrics[item] += sum(batch_metrics[item])
                    else:
                        epoch_metrics.update({item: sum(batch_metrics[item])})

                # sum the losses for all steps
                total_losses = sum(losses)
                epoch_metrics['total_loss'] += total_losses.item()

                if not math.isfinite(total_losses):
                    self.logger("Loss is {}, stopping training".format(total_losses))
                    self.logger(loss_dict_reduced)
                    sys.exit(1)

                if self.args.amp:
                    optimizer.zero_grad()
                    scaler.scale(total_losses).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                    skip_lr_step = (scale > scaler.get_scale())
                else:
                    optimizer.zero_grad()
                    total_losses.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

            if skip_lr_step:
                self.logger("Skipping LR Scheduler step.")
            else:
                lr_scheduler.step()

        for item in epoch_metrics:
            epoch_metrics[item] = epoch_metrics[item] / len(data_loader)
        epoch_metrics['lr'] = lr_scheduler.get_last_lr()[0]

        return epoch_metrics

    def unnormalize_outputs_sv(self, outputs):
        unnorm_outputs = outputs * (self.args.focus_vol_size // 2)
        unnorm_outputs += (self.args.sub_vol_size // 2)

        return unnorm_outputs

    def unnormalize_outputs(self, outputs):
        unnorm_outputs = outputs * (self.args.focus_vol_size // 2)

        return unnorm_outputs

    def find(self, x, parent):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = self.find(parent[x], parent)  # Path compression
        return parent[x]

    def union(self, x, y, parent):
        parent[self.find(x, parent)] = self.find(y, parent)

    def find_duplicate_groups(self, pair_to_div):
        parent = {}

        for (i, j), divergence in pair_to_div.items():
            if divergence > float(self.args.seq_len - 2):
                self.union(i, j, parent)

        # Build groups from parent mappings
        groups = defaultdict(set)
        for node in parent:
            root = self.find(node, parent)
            groups[root].add(node)

        if len(groups):
            groups = {i: {"branches": list(group)} for i, group in enumerate(list(groups.values()))}
        else:
            # If no groups found, then every branch is its own group
            groups = {i: {"branches": [i]} for i in range(self.args.num_init_branches)}

        return groups

    def build_tree_from_trajectories_sv(self, graph, trajectories, radii,
                                        divergence_pos, end_pos, branch_indices, root_id):
        """
        Build a tree from multiple trajectories with divergence information.

        Args:
            graph: A NetworkX DiGraph object to build the tree
            trajectories: List of num_init_branches PyTorch tensors of shape (seq_len - 1, 3)
            radii: List of num_init_branches PyTorch tensors of shape (seq_len - 1, 1)
            divergence_pos: A tensor of size (num_init_branches^2 - num_init_branches) / 2
            integers representing divergence points for each pair
            end_pos: A tensor of size (num_init_branches, 1) representing the end position
            index of each trajectory
            branch_indices: A tensor of size (num_init_branches^2 - num_init_branches) / 2 , 2
            representing the indices of the branches that diverged
            root_id: The ID of the root node

        Returns:
            A NetworkX DiGraph representing the trajectory tree
        """
        # Create a mapping from trajectory pairs to divergence positions
        pair_to_div = {}
        for idx in range(len(divergence_pos)):
            i, j = branch_indices[idx].tolist()
            div_pos = divergence_pos[idx].item()
            if (i, j) in pair_to_div:
                pair_to_div[(i, j)] = (div_pos + pair_to_div[(i, j)]) / 2
                pair_to_div[(j, i)] = pair_to_div[(i, j)]
            else:
                pair_to_div[(i, j)] = div_pos
                pair_to_div[(j, i)] = div_pos
        pair_to_div = {k: float(round(v)) for k, v in pair_to_div.items()}

        current_level = [root_id]
        current_br_id = int(root_id.split('-')[0])

        # find duplicate branches
        representative_branches = self.find_duplicate_groups(pair_to_div)
        for i, group in representative_branches.items():
            group['positions'] = torch.mean(trajectories[group['branches']], dim=0)
            group['radii'] = torch.mean(radii[group['branches']], dim=0)
            group['end_pos'] = torch.round(torch.max(end_pos[group['branches']]))

        # compute divergence position between the representative branches
        group_pair_to_div = {}
        for i, group in representative_branches.items():
            for j, other_group in representative_branches.items():
                if i != j and (i, j) not in group_pair_to_div:
                    group_div_list = []
                    for branch in group['branches']:
                        for other_branch in other_group['branches']:
                            div_pos = pair_to_div.get((branch, other_branch), float('inf'))
                            group_div_list.append(div_pos)
                    div_pos = torch.round(torch.mean(torch.tensor(group_div_list))).item()
                    group_pair_to_div[(i, j)] = div_pos
                    group_pair_to_div[(j, i)] = div_pos

        # Process each timestep from 1 to seq_len - 1
        for t in range(1, self.args.seq_len):
            next_level = []
            for node in current_level:
                node_data = graph.nodes[node]
                traj_group = set()
                for traj in node_data['trajectories']:
                    for i, group in representative_branches.items():
                        if traj in group['branches']:
                            traj_group.add(i)

                if len(traj_group) == 1:
                    traj_idx = next(iter(traj_group))
                    new_pos = representative_branches[traj_idx]['positions'][t - 1]
                    new_pos = new_pos.cpu().numpy()
                    new_rad = representative_branches[traj_idx]['radii'][t - 1]
                    new_rad = new_rad.item()
                    node_br_id = node.split('-')[0]
                    node_pt_id = node.split('-')[1]
                    new_node = node_br_id + '-' + str(int(node_pt_id) + 1)
                    graph.add_node(new_node, position=new_pos, radius=new_rad, depth=t,
                                   trajectories=set(representative_branches[traj_idx]['branches']))
                    graph.add_edge(node, new_node)
                    next_level.append(new_node)
                    continue

                # Find subgroups that should stay together at this timestep
                G = nx.Graph()
                G.add_nodes_from(traj_group)
                for i in traj_group:
                    for j in traj_group:
                        if i < j:
                            div_point = group_pair_to_div.get((i, j), -float('inf'))
                            if t <= div_point:
                                G.add_edge(i, j)
                subgroups = [set(c) for c in nx.connected_components(G)]

                # Create nodes for each subgroup
                for subgroup in subgroups:
                    if len(subgroup) > 1:
                        positions = torch.stack([representative_branches[idx]['positions'][t - 1] for idx in subgroup])
                        new_pos = torch.mean(positions, dim=0)
                        radii = torch.stack([representative_branches[idx]['radii'][t - 1] for idx in subgroup])
                        new_rad = torch.mean(radii, dim=0)
                    else:
                        new_pos = representative_branches[next(iter(subgroup))]['positions'][t - 1].clone()
                        new_rad = representative_branches[next(iter(subgroup))]['radii'][t - 1].clone()

                    if len(subgroup) == len(traj_group):
                        node_br_id = node.split('-')[0]
                        node_pt_id = node.split('-')[1]
                        new_node = node_br_id + '-' + str(int(node_pt_id) + 1)
                    else:
                        current_br_id += 1
                        new_node = str(current_br_id) + '-0'

                    new_pos = new_pos.cpu().numpy()
                    new_rad = new_rad.item()
                    original_trajs = []
                    for traj in subgroup:
                        original_trajs.extend(representative_branches[traj]['branches'])
                    graph.add_node(new_node, position=new_pos, radius=new_rad, depth=t, trajectories=set(original_trajs))
                    graph.add_edge(node, new_node)
                    next_level.append(new_node)
            current_level = next_level

        # use the end positions to terminate branches
        end_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
        paths = [nx.shortest_path(graph, root_id, end) for end in end_nodes]
        # reduce the paths to only the nodes after the last bifurcation
        for p_i, path in enumerate(paths):
            bifur_nodes = [path[i] for i in range(len(path) - 1) if graph.out_degree(path[i]) > 1]
            if len(bifur_nodes) > 0:
                bifur_node = bifur_nodes[-1]
                bifur_idx = path.index(bifur_node)
                paths[p_i] = path[bifur_idx + 1:]

        for p_i, path in enumerate(paths):
            for node in path:
                node_depth = graph.nodes[node]['depth']
                traj_group = set()
                for traj in graph.nodes[node]['trajectories']:
                    for i, group in representative_branches.items():
                        if traj in group['branches']:
                            traj_group.add(i)
                assert len(traj_group) == 1, f"Node {node} has multiple trajectory groups: {traj_group}"
                traj_group = list(traj_group)[0]

                if node_depth > representative_branches[traj_group]['end_pos']:
                    descendants = list(nx.descendants(graph, node))
                    descendants.append(node)
                    if len(descendants) > 0:
                        predecessor = list(graph.predecessors(node))[0]
                        des_pos = [graph.nodes[predecessor]['position']] + [graph.nodes[p]['position'] for p in descendants]
                        avg_pos = np.mean(np.array(des_pos, dtype=np.float32), axis=0)
                        des_rad = [graph.nodes[predecessor]['radius']] + [graph.nodes[p]['radius'] for p in descendants]
                        avg_rad = np.mean(np.array(des_rad, dtype=np.float32), axis=0)
                        graph.nodes[predecessor]['position'] = avg_pos
                        graph.nodes[predecessor]['radius'] = avg_rad
                        graph.remove_nodes_from(descendants)
                        break

        return graph

    @staticmethod
    def find_leaf_paths(graph, root):
        leaf_paths = []
        for level in nx.bfs_layers(graph, root):
            for node in level:
                if graph.out_degree(node) == 0 and node != root:
                    leaf_paths.append(nx.shortest_path(graph, source=root, target=node))
        return leaf_paths

    def build_tree_from_trajectories(self, graph, trajectories, radii, divergence_pos, end_pos,
                                     branch_indices, root_id, global_branch_id):
        """
        Build a tree from multiple trajectories with divergence information.

        Args:
            graph: A NetworkX DiGraph object to build the tree
            trajectories: List of num_init_branches PyTorch tensors of shape (seq_len - 1, 3)
            radii: List of num_init_branches PyTorch tensors of shape (seq_len - 1, 1)
            divergence_pos: A tensor of size (num_init_branches^2 - num_init_branches) / 2
            integers representing divergence points for each pair
            end_pos: A tensor of size (num_init_branches, 1) representing the end position
            index of each trajectory
            branch_indices: A tensor of size (num_init_branches^2 - num_init_branches) / 2 , 2
            representing the indices of the branches that diverged
            root_id: The ID of the root node
            global_branch_id: The current max branch ID of the full tree

        Returns:
            A NetworkX DiGraph representing the trajectory tree
        """
        # Create a mapping from trajectory pairs to divergence positions
        pair_to_div = {}
        for idx in range(len(divergence_pos)):
            i, j = branch_indices[idx].tolist()
            div_pos = divergence_pos[idx].item()
            if (i, j) in pair_to_div:
                pair_to_div[(i, j)] = (div_pos + pair_to_div[(i, j)]) / 2
                pair_to_div[(j, i)] = pair_to_div[(i, j)]
            else:
                pair_to_div[(i, j)] = div_pos
                pair_to_div[(j, i)] = div_pos
        pair_to_div = {k: float(round(v)) for k, v in pair_to_div.items()}

        current_level = [root_id]
        root_pos = graph.nodes[root_id]['position']

        # find duplicate branches
        representative_branches = self.find_duplicate_groups(pair_to_div)
        for i, group in representative_branches.items():
            group['positions'] = torch.mean(trajectories[group['branches']], dim=0)
            group['radii'] = torch.mean(radii[group['branches']], dim=0)
            group['end_pos'] = torch.round(torch.max(end_pos[group['branches']]))

        # compute divergence position between the representative branches
        group_pair_to_div = {}
        for i, group in representative_branches.items():
            for j, other_group in representative_branches.items():
                if i != j and (i, j) not in group_pair_to_div:
                    group_div_list = []
                    for branch in group['branches']:
                        for other_branch in other_group['branches']:
                            div_pos = pair_to_div.get((branch, other_branch), float('inf'))
                            group_div_list.append(div_pos)
                    div_pos = torch.round(torch.mean(torch.tensor(group_div_list))).item()
                    group_pair_to_div[(i, j)] = div_pos
                    group_pair_to_div[(j, i)] = div_pos

        # Process each timestep from 1 to seq_len - 1
        for t in range(1, self.args.seq_len):
            next_level = []
            for node in current_level:
                node_data = graph.nodes[node]
                traj_group = set()
                for traj in node_data['trajectories']:
                    for i, group in representative_branches.items():
                        if traj in group['branches']:
                            traj_group.add(i)

                if len(traj_group) == 1:
                    traj_idx = next(iter(traj_group))
                    new_pos = representative_branches[traj_idx]['positions'][t - 1]
                    new_pos = new_pos.cpu().numpy()
                    new_rad = representative_branches[traj_idx]['radii'][t - 1]
                    new_rad = new_rad.item()
                    node_br_id = node.split('-')[0]
                    node_pt_id = node.split('-')[1]
                    new_node = node_br_id + '-' + str(int(node_pt_id) + 1)
                    global_pos = root_pos + new_pos
                    global_depth = node_data['global_depth'] + 1
                    graph.add_node(new_node, position=global_pos, radius=new_rad, global_depth=global_depth,
                                   depth=t, trajectories=set(representative_branches[traj_idx]['branches']))
                    graph.add_edge(node, new_node)
                    next_level.append(new_node)
                    continue

                # Find subgroups that should stay together at this timestep
                G = nx.Graph()
                G.add_nodes_from(traj_group)
                for i in traj_group:
                    for j in traj_group:
                        if i < j:
                            div_point = group_pair_to_div.get((i, j), -float('inf'))
                            if t <= div_point:
                                G.add_edge(i, j)
                subgroups = [set(c) for c in nx.connected_components(G)]

                # Create nodes for each subgroup
                for subgroup in subgroups:
                    if len(subgroup) > 1:
                        positions = torch.stack([representative_branches[idx]['positions'][t - 1] for idx in subgroup])
                        new_pos = torch.mean(positions, dim=0)
                        radii = torch.stack([representative_branches[idx]['radii'][t - 1] for idx in subgroup])
                        new_rad = torch.mean(radii, dim=0)
                    else:
                        new_pos = representative_branches[next(iter(subgroup))]['positions'][t - 1].clone()
                        new_rad = representative_branches[next(iter(subgroup))]['radii'][t - 1].clone()

                    if len(subgroup) == len(traj_group):
                        node_br_id = node.split('-')[0]
                        node_pt_id = node.split('-')[1]
                        new_node = node_br_id + '-' + str(int(node_pt_id) + 1)
                    else:
                        global_branch_id[0] += 1
                        new_node = str(global_branch_id[0]) + '-0'

                    new_pos = new_pos.cpu().numpy()
                    new_rad = new_rad.item()
                    original_trajs = []
                    for traj in subgroup:
                        original_trajs.extend(representative_branches[traj]['branches'])
                    global_pos = root_pos + new_pos
                    global_depth = node_data['global_depth'] + 1
                    graph.add_node(new_node, position=global_pos, radius=new_rad, global_depth=global_depth,
                                   depth=t, trajectories=set(original_trajs))
                    graph.add_edge(node, new_node)
                    next_level.append(new_node)

            current_level = next_level

        # use the end positions to terminate branches
        paths = self.find_leaf_paths(graph, root_id)
        for p_i, path in enumerate(paths):
            bifur_nodes = [path[i] for i in range(len(path) - 1) if graph.out_degree(path[i]) > 1]
            if len(bifur_nodes) > 0:
                bifur_node = bifur_nodes[-1]
                bifur_idx = path.index(bifur_node)
                paths[p_i] = path[bifur_idx + 1:]

        end_nodes = set()
        continuing_nodes = set()
        for p_i, path in enumerate(paths):
            finished = False
            for node in path:
                node_depth = graph.nodes[node]['depth']
                traj_group = set()
                for traj in graph.nodes[node]['trajectories']:
                    for i, group in representative_branches.items():
                        if traj in group['branches']:
                            traj_group.add(i)
                assert len(traj_group) == 1, f"Node {node} has multiple trajectory groups: {traj_group}"
                traj_group = list(traj_group)[0]

                if node_depth > representative_branches[traj_group]['end_pos']:
                    descendants = list(nx.descendants(graph, node))
                    descendants.append(node)
                    if len(descendants) > 0:
                        predecessor = list(graph.predecessors(node))[0]
                        des_pos = [graph.nodes[predecessor]['position']] + [graph.nodes[p]['position'] for p in descendants]
                        avg_pos = np.mean(np.array(des_pos, dtype=np.float32), axis=0)
                        des_rad = [graph.nodes[predecessor]['radius']] + [graph.nodes[p]['radius'] for p in descendants]
                        avg_rad = np.mean(np.array(des_rad, dtype=np.float32), axis=0)
                        graph.nodes[predecessor]['position'] = avg_pos
                        graph.nodes[predecessor]['radius'] = avg_rad
                        graph.remove_nodes_from(descendants)
                        end_nodes.add(predecessor)
                        finished = True
                        break
            if not finished:
                continuing_nodes.add(path[-1])

        return graph, list(end_nodes), list(continuing_nodes)

    @torch.no_grad()
    def evaluate_sv(self, model, data_loader):
        model.eval()
        all_masks = []
        all_samples = []
        all_samples_ids = []
        all_pro_preds = []
        all_pro_targets = []
        elapsed_time = []
        seq_len = self.args.seq_len - 1
        num_branches = self.args.num_init_branches

        for b_i, batch in tqdm(enumerate(data_loader), desc="Batch", leave=False, total=len(data_loader)):
            start = time.time()
            sample_imgs, sample_past_trs, targets = (batch["image"], batch["past_tr"], batch["label"])
            masks = batch["mask"]

            if sample_imgs.device != self.args.device:
                sample_imgs = sample_imgs.to(self.args.device)
            if sample_past_trs is not None:
                sample_past_trs = sample_past_trs.to(self.args.device)

            # prev_step_info
            prev_step_info = {"first_step": True,
                              "out": None,
                              "memory": None,
                              "pos": None,
                              "hs_without_norm": None,
                              "node_type": 'selected_node',
                              "level": 0,
                              "hidden_state_list": [],
                              "parent_pred_index": [0] * self.args.batch_size}

            # iterating over steps
            all_outputs = [torch.zeros(num_branches, self.args.num_steps, seq_len, 3).to(self.args.device)
                           for _ in range(self.args.batch_size)]
            all_radii = [torch.zeros(num_branches, self.args.num_steps, seq_len, 1).to(self.args.device)
                         for _ in range(self.args.batch_size)]
            all_processed_outputs = []
            for step in range(self.args.num_steps):
                norm_step = torch.tensor(step / (self.args.num_steps - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                prev_step_info = model(sample_imgs, sample_past_trs, prev_step_info, norm_step)
                outputs = prev_step_info['out']

                for o_i, (output_pos, output_rad) in enumerate(zip(outputs['direc_logits'], outputs['rad_logits'])):
                    reshaped_output = output_pos.reshape(num_branches, seq_len, 3)
                    reshaped_rad = output_rad.reshape(num_branches, seq_len, 1)
                    all_outputs[o_i][:, step, :, :] = reshaped_output
                    all_radii[o_i][:, step, :, :] = reshaped_rad

                if step == self.args.num_steps - 1:
                    for o_i in range(self.args.batch_size):
                        output_pos = all_outputs[o_i]
                        output_rad = all_radii[o_i]
                        output_div = outputs['div_logits'][o_i]
                        output_end = outputs['end_logits'][o_i]

                        # unnormalize
                        output_pos = self.unnormalize_outputs_sv(output_pos)
                        all_outputs[o_i] = output_pos
                        output_pos_last = output_pos[:, -1, :, :]
                        output_rad = output_rad * (self.args.sub_vol_size // 2)
                        all_radii[o_i] = output_rad
                        output_rad_last = output_rad[:, -1, :, :]

                        # get branch pairs
                        idxs = torch.arange(num_branches)
                        branch_pairs = torch.cartesian_prod(idxs, idxs)
                        branch_pairs = branch_pairs[branch_pairs[:, 0] != branch_pairs[:, 1]]
                        non_dup_branch_pairs_idxs = (branch_pairs[:, 0] != branch_pairs[:, 1]).nonzero().squeeze(1)
                        non_dup_branch_pairs = branch_pairs[non_dup_branch_pairs_idxs].to(output_pos.device)

                        # unnormalize
                        output_div = output_div[non_dup_branch_pairs_idxs].clone()
                        output_div *= (self.args.seq_len - 1)
                        output_end = output_end.clone()
                        output_end *= (self.args.seq_len - 1)
                        output_end = torch.round(output_end)

                        # build tree
                        # Initialize the tree with root node
                        pro_output = nx.DiGraph()
                        root_id = "0-0"
                        root_pos = targets[o_i]['root_position']
                        root_pos = root_pos + ((self.args.sub_vol_size // 2) - np.round(root_pos))
                        root_rad = targets[o_i]['root_radius']
                        pro_output.add_node(root_id, position=root_pos, radius=root_rad, depth=0,
                                            trajectories=set(range(num_branches)))
                        pro_output = self.build_tree_from_trajectories_sv(pro_output, output_pos_last, output_rad_last,
                                                                          output_div, output_end, non_dup_branch_pairs, root_id)
                        all_processed_outputs.append(pro_output)

            end = time.time()
            elp_time = (end - start) / len(targets)
            elapsed_time += [elp_time for _ in range(len(targets))]
            all_pro_preds += all_processed_outputs
            all_pro_targets += [tgt['selected_node'] for tgt in targets]
            for sam_i in range(len(targets)):
                all_samples.append(sample_imgs[sam_i].cpu().detach())
                all_samples_ids.append(targets[sam_i]['index'])
                all_masks.append(masks[sam_i].cpu().detach())
        return all_pro_preds, all_pro_targets, all_samples_ids, all_samples, all_masks, elapsed_time

    def check_finished(self, targets, tree_id, curr_level, level):
        break_loop = False
        if level >= self.args.max_inference_levels and self.args.eval_limit_levels:
            self.logger.info(f"Max inference levels reached! Sample: {str(targets[0]['index'])}, Tree ID: {tree_id:d}")
            break_loop = True

        if len(curr_level) > self.args.max_nodes_per_level and self.args.eval_limit_nodes_per_level:
            self.logger.info(f"Max nodes in a level reached! Skipping further evaluation.  "
                             f"Sample: {targets[0]['index']}, Tree ID: {tree_id}")
            break_loop = True
        return break_loop

    def get_global_pred_root_node_nx(self, pred_tree, target_tree):
        """
        Create a Node for the root point
        """
        in_degrees = dict(target_tree.in_degree())
        root_node_id = [node for node, in_degree in in_degrees.items() if in_degree == 0][0]
        root_node_info = self.get_node(root_node_id, target_tree)
        root_pos = root_node_info['position']
        radius = root_node_info['radius']
        pred_tree.add_node(root_node_id, position=root_pos,
                           radius=radius, global_depth=0, depth=0,
                           bifur_parent='None')

        return root_node_info

    def get_sub_vol_and_past_tr(self, node_batch, samples, pred_tree, samples_min, crop_pad, masks=None):
        """
        Get the sub-volume and past trajectory for the nodes in the node_batch
        """
        sub_vol_batch = []
        sub_mask_batch = []
        past_traj_pos_batch = []
        to_remove = []
        for node in node_batch:
            image_size = torch.tensor(samples.squeeze().size())
            node_pos = torch.tensor(self.get_node(node, pred_tree)['position'])
            if torch.logical_and(torch.all(node_pos < (image_size - 1)), torch.all(node_pos >= 0)):
                sub_vol = crop_pad(samples.squeeze(dim=0), node_pos.tolist(), samples_min.item())
                sub_vol = sub_vol.unsqueeze(0)
                if masks is not None:
                    sub_mask = crop_pad(masks.squeeze(dim=0), node_pos.tolist(), False)
            else:
                to_remove.append(node)
                continue

            if len(sub_vol.shape) == 6:
                sub_vol = sub_vol.squeeze(0)
            if masks is not None and len(sub_mask.shape) == 6:
                sub_mask = sub_mask.squeeze(0)

            past_traj_pos = self.generate_past_trajectory_pp_nx(pred_tree, node)
            past_traj_pos = past_traj_pos.to(self.device).unsqueeze(0)
            sub_vol_batch.append(sub_vol)
            past_traj_pos_batch.append(past_traj_pos)

        for node in to_remove:
            node_batch.remove(node)

        return sub_vol_batch, past_traj_pos_batch, sub_mask_batch

    @staticmethod
    def get_node(node_id, tree):
        root_node_info = tree.nodes[node_id]
        root_node_info['id'] = node_id
        return root_node_info

    def generate_past_trajectory_pp_nx(self, tree, node_id):
        past_traj_label = []
        current_node = self.get_node(node_id, tree)
        parent = current_node
        root_position = np.asarray(current_node['position'])

        for _ in range(self.args.num_prev_pos):
            node_info_list = []
            node_position = (np.asarray(parent['position']) - root_position).astype(float)
            node_position /= (self.args.focus_vol_size // 2)
            node_info_list.append(node_position)
            node_radius = np.asarray(parent['radius']).reshape(1)
            node_radius /= (self.args.sub_vol_size // 2)
            node_info_list.append(node_radius)
            past_traj_label.append(np.concatenate(node_info_list))

            parent_id = list(tree.predecessors(parent['id']))
            if not len(parent_id):
                break
            parent = self.get_node(parent_id[0], tree)

        past_traj_label_np = np.asarray(past_traj_label)

        num_missing_points = self.args.num_prev_pos - past_traj_label_np.shape[0]
        if num_missing_points:
            last_point_info = past_traj_label_np[-1:]
            padding_info = np.tile(last_point_info, (num_missing_points, 1))
            past_traj_label_np = np.vstack((past_traj_label_np, padding_info))
        past_traj_label_t = torch.FloatTensor(past_traj_label_np).flatten()

        return past_traj_label_t

    @torch.no_grad()
    def evaluate(self, model, data_loader):
        model.eval()
        all_masks = []
        all_samples = []
        all_samples_ids = []
        all_preds = []
        all_targets = []
        elapsed_time = []
        crop_pad = CropAndPad(self.args.sub_vol_size, 'area')
        seq_len = self.args.seq_len - 1

        num_branches = self.args.num_init_branches

        # samples consist of a batch of full volumes (currently only batch_size 1 supported)
        for i, batch in enumerate(data_loader):
            samples, samples_min, targets, masks = (batch["image"], batch["image_min"], batch["label"], batch["mask"])
            if self.args.eval_only:
                if self.args.distributed:
                    self.logger(f"Sample: {targets[0]['index']}", force=True)
                else:
                    self.logger(f"Sample: {targets[0]['index']}")

            for tree_id in range(len(targets[0]['networkx'])):
                start = time.time()
                if self.args.eval_only:
                    self.logger(f"Tree ID: {tree_id}")
                finished = False
                global_branch_id = [0]
                level = 0

                target_tree = targets[0]['networkx'][tree_id]
                pred_tree = nx.DiGraph()
                root_node = self.get_global_pred_root_node_nx(pred_tree, target_tree)
                curr_level = [root_node['id']]
                curr_level_info = {'prev_hs_without_norm': [None],
                                   'prev_matched_targets': {'indices': [None]},
                                   'parent_target_index': [None],
                                   'past_prev_hs': [None]}

                while not finished:
                    if self.check_finished(targets, tree_id, curr_level, level):
                        break
                    if self.args.eval_only:
                        self.logger(f"Level: {level} \t | \t Nodes: {len(curr_level)} \t")

                    # initializing next_level which is populated as we process root points in curr_level
                    next_level = []
                    next_level_info = {'prev_hs_without_norm': [],
                                       'prev_matched_targets': {'indices': []},
                                       'parent_target_index': [],
                                       'past_prev_hs': []}

                    # processing each node (root point of a sub-volume) in curr_level
                    for level_node_i in range(0, len(curr_level), self.args.batch_size_per_sample):
                        node_batch = curr_level[level_node_i: level_node_i + self.args.batch_size_per_sample]
                        sub_vol_batch, past_traj_pos_batch, _ = self.get_sub_vol_and_past_tr(node_batch, samples,
                                                                                             pred_tree, samples_min, crop_pad)
                        if not len(sub_vol_batch):
                            continue

                        curr_level_info_batch = {'prev_hs_without_norm': [],
                                                 'prev_matched_targets':  {'indices': []},
                                                 'parent_target_index': [],
                                                 'past_prev_hs': []}
                        for node in node_batch:
                            node_idx = curr_level.index(node)
                            curr_level_info_batch['prev_hs_without_norm'].append(curr_level_info['prev_hs_without_norm'][node_idx])
                            curr_level_info_batch['prev_matched_targets']['indices'].append(curr_level_info['prev_matched_targets']['indices'][node_idx])
                            curr_level_info_batch['parent_target_index'].append(curr_level_info['parent_target_index'][node_idx])
                            curr_level_info_batch['past_prev_hs'].append(curr_level_info['past_prev_hs'][node_idx])

                        sample_imgs = torch.cat(sub_vol_batch, dim=0)
                        sample_past_trs = torch.cat(past_traj_pos_batch, dim=0)
                        if sample_imgs.device != self.args.device:
                            sample_imgs = sample_imgs.to(self.args.device)
                        if sample_past_trs is not None:
                            sample_past_trs = sample_past_trs.to(self.args.device)
                        if level == 0:
                            prev_step_info = {"first_step": True,
                                              "out": None,
                                              "memory": None,
                                              "pos": None,
                                              "hs_without_norm": None,
                                              "node_type": "selected_node",
                                              "level": level,
                                              "parent_pred_index": [0]}
                        else:
                            prev_step_info = {"first_step": True,
                                              "out": None,
                                              "memory": None,
                                              "pos": None,
                                              "node_type": "pair_node",
                                              "prev_hs_without_norm": torch.stack(curr_level_info_batch['prev_hs_without_norm']),
                                              "prev_matched_targets": curr_level_info_batch['prev_matched_targets'],
                                              "parent_target_index": curr_level_info_batch['parent_target_index'],
                                              'past_prev_hs': torch.stack(curr_level_info_batch['past_prev_hs']),
                                              "level": level}

                            parent_pred_index = []
                            for pt_i, pt_idx in enumerate(prev_step_info['parent_target_index']):
                                prev_matched_indices = prev_step_info['prev_matched_targets']['indices'][pt_i]
                                matched_tgt_indices = torch.nonzero(prev_matched_indices[0] == pt_idx, as_tuple=True)[0]
                                matched_pred_indices = prev_matched_indices[1][matched_tgt_indices]
                                rand_idx = torch.randint(0, matched_pred_indices.size(0), (1,)).item()
                                parent_pred_index.append(matched_pred_indices[rand_idx].item())
                            prev_step_info['parent_pred_index'] = parent_pred_index
                            prev_step_info['prev_hs_without_norm'] = torch.stack(curr_level_info_batch['prev_hs_without_norm'])
                            prev_step_info['hs_without_norm'] = None

                        # iterating over steps
                        batch_size = sample_imgs.shape[0]
                        all_outputs = [torch.zeros(num_branches, self.args.num_steps, seq_len, 3).to(self.args.device)
                                       for _ in range(batch_size)]
                        all_radii = [torch.zeros(num_branches, self.args.num_steps, seq_len, 1).to(self.args.device)
                                     for _ in range(batch_size)]

                        for step in range(self.args.num_steps):
                            norm_step = torch.tensor(step / (self.args.num_steps - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                            prev_step_info = model(sample_imgs, sample_past_trs, prev_step_info, norm_step)
                            outputs = prev_step_info['out']

                            for o_i, (output_pos, output_rad) in enumerate(zip(outputs['direc_logits'], outputs['rad_logits'])):
                                reshaped_output = output_pos.reshape(num_branches, seq_len, 3)
                                reshaped_rad = output_rad.reshape(num_branches, seq_len, 1)
                                all_outputs[o_i][:, step, :, :] = reshaped_output
                                all_radii[o_i][:, step, :, :] = reshaped_rad

                            if step == self.args.num_steps - 1:
                                for o_i in range(batch_size):
                                    output_pos = all_outputs[o_i]
                                    output_rad = all_radii[o_i]
                                    output_div = outputs['div_logits'][o_i]
                                    output_end = outputs['end_logits'][o_i]

                                    # unnormalize
                                    output_pos = self.unnormalize_outputs(output_pos)
                                    all_outputs[o_i] = output_pos
                                    output_pos_last = output_pos[:, -1, :, :]
                                    output_rad = output_rad * (self.args.sub_vol_size // 2)
                                    all_radii[o_i] = output_rad
                                    output_rad_last = output_rad[:, -1, :, :]

                                    # get branch pairs
                                    idxs = torch.arange(num_branches)
                                    branch_pairs = torch.cartesian_prod(idxs, idxs)
                                    branch_pairs = branch_pairs[branch_pairs[:, 0] != branch_pairs[:, 1]]
                                    non_dup_branch_pairs_idxs = (branch_pairs[:, 0] != branch_pairs[:, 1]).nonzero().squeeze(1)
                                    non_dup_branch_pairs = branch_pairs[non_dup_branch_pairs_idxs].to(output_pos.device)

                                    # unnormalize
                                    output_div = output_div.clone()
                                    output_div *= (self.args.seq_len - 1)
                                    output_end = output_end.clone()
                                    output_end *= (self.args.seq_len - 1)
                                    output_end = torch.round(output_end)

                                    # build tree
                                    root_id = node_batch[o_i]
                                    pred_tree.nodes[root_id]['trajectories'] = set(range(num_branches))
                                    pred_tree.nodes[root_id]['depth'] = 0
                                    pred_tree, end_nodes, continuing_nodes = self.build_tree_from_trajectories(pred_tree,
                                                                                                               output_pos_last, output_rad_last, output_div,
                                                                                                               output_end, non_dup_branch_pairs, root_id, global_branch_id)
                                    next_level += continuing_nodes

                                    for cont_node in continuing_nodes:
                                        next_level_info['prev_hs_without_norm'].append(prev_step_info['hs_without_norm'][o_i])
                                        if self.args.traj_train_len > 1:
                                            next_level_info['past_prev_hs'].append(prev_step_info['past_prev_hs'][o_i])
                                        node_traj = list(pred_tree.nodes[cont_node]['trajectories'])
                                        matched_pred_idxs = torch.arange(num_branches, dtype=torch.long)
                                        matched_tgt_idxs = torch.ones(num_branches, dtype=torch.long)
                                        matched_tgt_idxs[node_traj] = 0
                                        matched_targets = [matched_tgt_idxs, matched_pred_idxs]
                                        next_level_info['prev_matched_targets']['indices'].append(matched_targets)
                                        next_level_info['parent_target_index'].append(0)

                    curr_level = next_level
                    curr_level_info = next_level_info
                    if len(curr_level) == 0:
                        finished = True
                    level += 1

                end = time.time()
                elapsed_time.append(end - start)
                all_preds.append(pred_tree)
                all_targets.append(targets[0]['networkx'][tree_id])
                all_samples_ids.append(targets[0]['index'])
                if self.args.mask:
                    all_masks.append(masks)
                all_samples.append(samples)
        return all_preds, all_targets, all_samples_ids, all_samples, all_masks, elapsed_time

    @torch.no_grad()
    def evaluate_sinsam(self, model, sample_id):
        model.eval()
        crop_pad = CropAndPad(self.args.sub_vol_size, 'area')
        seq_len = self.args.seq_len - 1

        annot_dir = os.path.join(self.args.data_dir, 'annots_test')
        img_dir = os.path.join(self.args.data_dir, 'images_test')
        img_path = os.path.join(img_dir, sample_id + '.nii.gz')
        annot_path = os.path.join(annot_dir, sample_id + '.pickle')

        # load, window and normalize the image
        image_reader = LoadImage(image_only=True)
        samples = image_reader(img_path)
        if self.args.window_input:
            window_transform = Compose([ThresholdIntensity(threshold=self.args.window_max,
                                                           above=False, cval=self.args.window_max),
                                        ThresholdIntensity(threshold=self.args.window_min,
                                                           above=True, cval=self.args.window_min)])
            samples = window_transform(samples)
        norm = NormalizeIntensity()
        samples = norm(samples)
        samples = samples.unsqueeze(0).unsqueeze(0).to(self.args.device)
        compute_min = ComputeImageMin()
        samples_min = compute_min(samples).to(self.args.device)

        # load the target and mask
        annot_reader = LoadAnnotPickle()
        targets = [annot_reader(annot_path)]
        targets[0]['index'] = sample_id

        num_branches = self.args.num_init_branches

        if self.args.distributed:
            self.logger(f"Sample: {targets[0]['index']}", force=True)
        else:
            self.logger(f"Sample: {targets[0]['index']}")

        # predict the all the vessel trees for each sample one at a time
        for tree_id in range(len(targets[0]['networkx'])):
            start = time.time()
            self.logger(f"Tree ID: {tree_id}")
            finished = False
            global_branch_id = [0]

            # To perform inference we traverse the vessel tree in level order fashion, starting from the
            # root point (level = 0), we find 'seq_len' centerline points for all possible branches, and
            # move on to the next level (level = 1), the starting points for centerline detections now
            # consist of the edge points (points where there is still a vessel when we reach step = seq_len).
            level = 0
            target_tree = targets[0]['networkx'][tree_id]
            pred_tree = nx.DiGraph()
            root_node = self.get_global_pred_root_node_nx(pred_tree, target_tree)
            curr_level = [root_node['id']]
            curr_level_info = {'prev_hs_without_norm': [None],
                               'prev_matched_targets': {'indices': [None]},
                               'parent_target_index': [None]}
            if self.args.traj_train_len > 1:
                curr_level_info['past_prev_hs'] = [None]
            curr_level_info['hidden_state_list'] = [[]]

            while not finished:
                if self.check_finished(pred_tree, targets, tree_id, curr_level, level):
                    break
                self.logger(f"Level: {level} \t | \t Nodes: {len(curr_level)} \t")
                next_level = []
                next_level_info = {'prev_hs_without_norm': [],
                                   'prev_matched_targets': {'indices': []},
                                   'parent_target_index': []}
                if self.args.traj_train_len > 1:
                    next_level_info['past_prev_hs'] = []

                # processing each node (root point of a sub-volume) in curr_level
                for level_node_i in range(0, len(curr_level), self.args.batch_size_per_sample):
                    node_batch = curr_level[level_node_i: level_node_i + self.args.batch_size_per_sample]
                    sub_vol_batch, past_traj_pos_batch, _ = self.get_sub_vol_and_past_tr(node_batch, samples,
                                                                                         pred_tree, samples_min, crop_pad)
                    if not len(sub_vol_batch):
                        continue

                    curr_level_info_batch = {'prev_hs_without_norm': [],
                                             'prev_matched_targets': {'indices': []},
                                             'parent_target_index': [],
                                             'past_prev_hs': []}

                    for node in node_batch:
                        node_idx = curr_level.index(node)
                        curr_level_info_batch['prev_hs_without_norm'].append(curr_level_info['prev_hs_without_norm'][node_idx])
                        curr_level_info_batch['prev_matched_targets']['indices'].append(curr_level_info['prev_matched_targets']['indices'][node_idx])
                        curr_level_info_batch['parent_target_index'].append(curr_level_info['parent_target_index'][node_idx])
                        curr_level_info_batch['past_prev_hs'].append(curr_level_info['past_prev_hs'][node_idx])

                    sample_imgs = torch.cat(sub_vol_batch, dim=0)
                    sample_past_trs = torch.cat(past_traj_pos_batch, dim=0)
                    if sample_imgs.device != self.args.device:
                        sample_imgs = sample_imgs.to(self.args.device)
                    if sample_past_trs is not None:
                        sample_past_trs = sample_past_trs.to(self.args.device)

                    if level == 0 or self.args.traj_train_len == 1:
                        prev_step_info = {"first_step": True,
                                          "out": None,
                                          "memory": None,
                                          "pos": None,
                                          "hs_without_norm": None,
                                          "node_type": "selected_node",
                                          "level": level,
                                          "hidden_state_list": [],
                                          "parent_pred_index": [0]}
                    else:
                        prev_step_info = {"first_step": True,
                                          "out": None,
                                          "memory": None,
                                          "pos": None,
                                          "node_type": "pair_node",
                                          "prev_hs_without_norm": torch.stack(curr_level_info_batch['prev_hs_without_norm']),
                                          "prev_matched_targets": curr_level_info_batch['prev_matched_targets'],
                                          "parent_target_index": curr_level_info_batch['parent_target_index'],
                                          'past_prev_hs': torch.stack(curr_level_info_batch['past_prev_hs']),
                                          "level": level,
                                          "hidden_state_list": []}

                        parent_pred_index = []
                        for pt_i, pt_idx in enumerate(prev_step_info['parent_target_index']):
                            prev_matched_indices = prev_step_info['prev_matched_targets']['indices'][pt_i]
                            matched_tgt_indices = torch.nonzero(prev_matched_indices[0] == pt_idx, as_tuple=True)[0]
                            matched_pred_indices = prev_matched_indices[1][matched_tgt_indices]
                            rand_idx = torch.randint(0, matched_pred_indices.size(0), (1,)).item()
                            parent_pred_index.append(matched_pred_indices[rand_idx].item())
                        prev_step_info['parent_pred_index'] = parent_pred_index
                        prev_step_info['prev_hs_without_norm'] = torch.stack(curr_level_info_batch['prev_hs_without_norm'])
                        prev_step_info['hs_without_norm'] = None

                    # iterating over steps
                    batch_size = sample_imgs.shape[0]
                    all_outputs = [torch.zeros(num_branches, self.args.num_steps, seq_len, 3).to(self.args.device)
                                   for _ in range(batch_size)]
                    all_radii = [torch.zeros(num_branches, self.args.num_steps, seq_len, 1).to(self.args.device)
                                 for _ in range(batch_size)]

                    for step in range(self.args.num_steps):
                        if self.args.num_steps > 1:
                            norm_step = torch.tensor(step / (self.args.num_steps - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                        else:
                            norm_step = torch.tensor(1.0).unsqueeze(0).unsqueeze(0).to(self.args.device)
                        prev_step_info = model(sample_imgs, sample_past_trs, prev_step_info, norm_step)
                        outputs = prev_step_info['out']

                        for o_i, (output_pos, output_rad) in enumerate(zip(outputs['direc_logits'], outputs['rad_logits'])):
                            reshaped_output = output_pos.reshape(num_branches, seq_len, 3)
                            reshaped_rad = output_rad.reshape(num_branches, seq_len, 1)
                            all_outputs[o_i][:, step, :, :] = reshaped_output
                            all_radii[o_i][:, step, :, :] = reshaped_rad

                        if step == self.args.num_steps - 1:
                            for o_i in range(batch_size):
                                output_pos = all_outputs[o_i]
                                output_rad = all_radii[o_i]
                                output_div = outputs['div_logits'][o_i]
                                output_end = outputs['end_logits'][o_i]

                                # unnormalize
                                output_pos = self.unnormalize_outputs(output_pos)
                                all_outputs[o_i] = output_pos
                                output_pos_last = output_pos[:, -1, :, :]
                                output_rad = output_rad * (self.args.sub_vol_size // 2)
                                all_radii[o_i] = output_rad
                                output_rad_last = output_rad[:, -1, :, :]

                                # get branch pairs
                                idxs = torch.arange(num_branches)
                                branch_pairs = torch.cartesian_prod(idxs, idxs)
                                branch_pairs = branch_pairs[branch_pairs[:, 0] != branch_pairs[:, 1]]
                                non_dup_branch_pairs_idxs = (branch_pairs[:, 0] != branch_pairs[:, 1]).nonzero().squeeze(1)
                                non_dup_branch_pairs = branch_pairs[non_dup_branch_pairs_idxs].to(output_pos.device)

                                # unnormalize
                                output_div = output_div.clone()
                                output_div *= (self.args.seq_len - 1)
                                output_end = output_end.clone()
                                output_end *= (self.args.seq_len - 1)
                                output_end = torch.round(output_end)

                                # build tree
                                root_id = node_batch[o_i]
                                pred_tree.nodes[root_id]['trajectories'] = set(range(num_branches))
                                pred_tree.nodes[root_id]['depth'] = 0
                                pred_tree, end_nodes, continuing_nodes = self.build_tree_from_trajectories(pred_tree,
                                                                                                           output_pos_last, output_rad_last, output_div,
                                                                                                           output_end, non_dup_branch_pairs, root_id, global_branch_id)
                                next_level += continuing_nodes

                                for cont_node in continuing_nodes:
                                    next_level_info['prev_hs_without_norm'].append(prev_step_info['hs_without_norm'][o_i])
                                    if self.args.traj_train_len > 1:
                                        next_level_info['past_prev_hs'].append(prev_step_info['past_prev_hs'][o_i])
                                    node_traj = list(pred_tree.nodes[cont_node]['trajectories'])
                                    matched_pred_idxs = torch.arange(num_branches, dtype=torch.long)
                                    matched_tgt_idxs = torch.ones(num_branches, dtype=torch.long)
                                    matched_tgt_idxs[node_traj] = 0
                                    matched_targets = [matched_tgt_idxs, matched_pred_idxs]
                                    next_level_info['prev_matched_targets']['indices'].append(matched_targets)
                                    next_level_info['parent_target_index'].append(0)

                curr_level = next_level
                curr_level_info = next_level_info
                if len(curr_level) == 0:
                    finished = True
                level += 1

            all_preds = [pred_tree]
            all_targets = [targets[0]['networkx'][tree_id]]
            all_samples_ids = [targets[0]['index']]
            elapsed_time = [time.time()-start]
        return all_preds, all_targets, all_samples_ids, elapsed_time
