# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RefTr model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from src.reftr.util.misc import (get_world_size,
                                 is_dist_avail_and_initialized,
                                 linear_sum_assignment_with_inf)


class Reftr(nn.Module):
    """
    This is the RefTr module that performs centerline detection.
    """

    def __init__(self, transformer, num_prev_pos=1, num_prod_dec_layers=1, num_dir_dec_layers=1,
                 seq_len=10, num_init_branches=20, traj_train_len=1, div_mlp_depths=(1024, 512, 256, 128),
                 end_mlp_depths=(512, 256, 128, 64), mlp_activation='swish', retain_past_lvls_num=0):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture with SwinUNETR encoder. See transformer.py
            num_prev_pos: number of previous positions to be used in the past trajectory embedding.
            num_prod_dec_layers: number of producer decoder layers in the transformer.
            num_dir_dec_layers: number of director decoder layers in the transformer.
            seq_len: length of each branch including the starting point.
            num_init_branches: number of branches (token sets) for tracking.
            traj_train_len: number of sub-trajectories in a super trajectory.
            div_mlp_depths: the depth of each divergence MLP layer in the transformer.
            end_mlp_depths: the depth of each end MLP layer in the transformer.
            mlp_activation: the activation function for the MLP layers.
            retain_past_lvls_num: the number of past trajectory embeddings to retain in past memory.
        """
        super().__init__()

        self.transformer = transformer
        self.num_init_branches = num_init_branches
        self.traj_train_len = traj_train_len
        self.num_prod_dec_layers = num_prod_dec_layers
        self.num_dir_dec_layers = num_dir_dec_layers
        self.div_mlp_depths = div_mlp_depths
        self.end_mlp_depths = end_mlp_depths
        self.mlp_activation = mlp_activation
        self.retain_past_lvls_num = retain_past_lvls_num
        self.seq_len = seq_len - 1

        # number of points, divergence, end, class, and extra tokens
        num_pts_tokens = self.num_init_branches * self.seq_len
        num_div_tokens = self.num_init_branches
        num_end_tokens = self.num_init_branches
        num_ext_tokens = 4

        assert num_dir_dec_layers == 1 and num_prod_dec_layers == 1, "Number of layers for both decoders should be 1."
        num_dir_heads = num_dir_dec_layers
        num_prod_heads = num_prod_dec_layers
        total_num_heads = num_prod_heads + num_dir_heads

        # Direction head
        direc_output_dim = 3
        _direction_embed = MLP(self.hidden_dim, self.hidden_dim, direc_output_dim, 3)
        direction_embed_layerlist = [_direction_embed for _ in range(total_num_heads)]
        self.direction_embed = nn.ModuleList(direction_embed_layerlist)

        # Radius head
        rad_output_dim = 1
        _radius_embed = MLP(self.hidden_dim, self.hidden_dim, rad_output_dim, 3)
        radius_embed_layerlist = [_radius_embed for _ in range(total_num_heads)]
        self.radius_embed = nn.ModuleList(radius_embed_layerlist)

        # Divergence head
        div_input_dim = (self.hidden_dim * 2)
        _div_embed = MLPVar(div_input_dim, self.div_mlp_depths, 1, self.mlp_activation)
        self.div_embed = nn.ModuleList([_div_embed])

        # End position head
        end_input_dim = self.hidden_dim
        _end_embed = MLPVar(end_input_dim, self.end_mlp_depths, 1, self.mlp_activation)
        self.end_embed = nn.ModuleList([_end_embed])

        num_prev_pos_dim = 4
        self.num_past_tr_tokens = 3
        self.prev_pos_embed = MLP(num_prev_pos * num_prev_pos_dim,
                                  self.num_past_tr_tokens * self.hidden_dim,
                                  self.num_past_tr_tokens * self.hidden_dim, 3)
        self.step_embed = nn.Linear(1, self.hidden_dim)

        # number of all tokens for curr step
        num_curr_tokens = num_div_tokens + num_end_tokens + num_pts_tokens + num_ext_tokens
        self.pos_embed = nn.Embedding(num_curr_tokens, self.hidden_dim)

        # positional encoding for past tokens
        num_past_tokens = 0
        num_past_tokens_sv = 0
        if self.traj_train_len > 1:
            num_past_tokens = self.seq_len + 1
            num_past_tokens += 1
            if self.retain_past_lvls_num:
                num_past_tokens_sv = num_past_tokens
                num_past_tokens = num_past_tokens * self.retain_past_lvls_num
            self.prev_step_embed = nn.Embedding(num_past_tokens, self.hidden_dim)

        # create a dict for starting and end points of all token groups
        curr_max_idx = 0
        self.token_idx = {}
        self.token_idx.update({f'end_st': curr_max_idx, f'end_nm': num_end_tokens, f'end_en': curr_max_idx + num_end_tokens})
        self.token_idx[f'end_idxs'] = list(range(curr_max_idx, curr_max_idx + num_end_tokens))
        curr_max_idx = self.token_idx[f'end_en']
        self.token_idx.update({f'div_st': curr_max_idx, f'div_nm': num_div_tokens, f'div_en': curr_max_idx + num_div_tokens})
        self.token_idx[f'div_idxs'] = list(range(curr_max_idx, curr_max_idx + num_div_tokens))
        curr_max_idx = self.token_idx[f'div_en']
        self.token_idx.update({f'pts_st': curr_max_idx, f'pts_nm': num_pts_tokens, f'pts_en': curr_max_idx + num_pts_tokens})
        self.token_idx[f'pts_idxs'] = list(range(curr_max_idx, curr_max_idx + num_pts_tokens))
        curr_max_idx = self.token_idx[f'pts_en']
        self.token_idx.update({'ext_st': curr_max_idx, 'ext_nm': num_ext_tokens, 'ext_en': curr_max_idx + num_ext_tokens})
        self.token_idx.update({'pst_tr_st': curr_max_idx, 'pst_tr_nm': self.num_past_tr_tokens, 'pst_tr_en': curr_max_idx + self.num_past_tr_tokens})
        self.token_idx['curr_nm'] = num_curr_tokens
        self.token_idx['past_nm'] = num_past_tokens
        self.token_idx['past_sv_nm'] = num_past_tokens_sv
        self.token_idx['step_idx'] = num_curr_tokens - 1

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    def detach_dict(self, d):
        """Recursively detaches all tensors in a nested dictionary."""
        if isinstance(d, dict):
            return {k: self.detach_dict(v) for k, v in d.items()}
        elif isinstance(d, torch.Tensor):
            return d.detach()
        else:
            return d  # Return non-tensor values unchanged

    def forward(self, samples, past_trs, prev_step_info, step):
        device = samples.device
        batch_size = samples.shape[0]

        # get prev_step_info
        memory = prev_step_info['memory']
        prev_hs = prev_step_info['hs_without_norm']
        pos = prev_step_info['pos']
        prev_step_info['prev_out'] = self.detach_dict(prev_step_info['out']) if prev_step_info['out'] else None
        if 'past_prev_hs' in prev_step_info and prev_step_info['past_prev_hs'] is not None:
            prev_step_info['past_prev_hs'] = prev_step_info['past_prev_hs'].transpose(0, 1)

        query_embed = self.pos_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)

        # get the previous hidden states
        if prev_hs is None:
            prev_hs = torch.zeros_like(query_embed)
        else:
            num_prev_hs = self.token_idx[f'pts_en']
            zeros_hs = torch.zeros_like(query_embed)
            zeros_hs[:num_prev_hs, :, :] = prev_hs.transpose(0, 1)
            prev_hs = zeros_hs

        past_trs_embed = self.prev_pos_embed(past_trs).unsqueeze(0)
        past_trs_embed = torch.cat(past_trs_embed.chunk(self.token_idx['pst_tr_nm'], dim=2), dim=0)
        prev_hs[self.token_idx['pst_tr_st']: self.token_idx['pst_tr_en']] = past_trs_embed

        step_embedding = self.step_embed(step)
        step_embedding = step_embedding.expand(batch_size, -1)
        prev_hs[self.token_idx['step_idx']] = step_embedding

        # collect the past tokens if any
        if self.traj_train_len > 1:
            if prev_step_info['node_type'] == 'selected_node' and prev_step_info['first_step']:
                past_prev_hs = torch.zeros([self.token_idx['past_nm'], batch_size, self.hidden_dim]).to(device)
                past_query_embed = self.prev_step_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
                prev_step_info['past_query_embed'] = past_query_embed
                prev_step_info['past_prev_hs'] = past_prev_hs

            elif prev_step_info['node_type'] == 'pair_node' and prev_step_info['first_step']:
                past_prev_hs = []
                for pt_i, pt_idx in enumerate(prev_step_info['parent_pred_index']):
                    end_idxs = [self.token_idx[f'end_st'] + pt_idx] if self.token_idx[f'end_nm'] else []
                    div_idxs = [self.token_idx[f'div_st'] + pt_idx] if self.token_idx[f'div_nm'] else []
                    start = self.token_idx[f'pts_st'] + (pt_idx * self.seq_len)
                    end = start + self.seq_len
                    pts_idxs = list(range(start, end))
                    all_idxs = end_idxs + div_idxs + pts_idxs
                    past_prev_hs.append(prev_step_info['prev_hs_without_norm'][pt_i][all_idxs])
                past_prev_hs = torch.stack(past_prev_hs)
                past_prev_hs = past_prev_hs.transpose(0, 1)
                past_query_embed = self.prev_step_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)

                # store for later use
                prev_step_info['past_query_embed'] = past_query_embed
                # add the lastest prev_hs to the past_prev_hs
                if self.retain_past_lvls_num:
                    prev_step_info['past_prev_hs'][self.token_idx['past_sv_nm']:] = prev_step_info['past_prev_hs'][:-self.token_idx['past_sv_nm']].clone()
                    prev_step_info['past_prev_hs'][:self.token_idx['past_sv_nm']] = past_prev_hs
                else:
                    prev_step_info['past_prev_hs'] = past_prev_hs

            else:
                pass

            past_query_embed = prev_step_info['past_query_embed']
            past_prev_hs = prev_step_info['past_prev_hs']

        # get transformer output
        dec_type = 'producer' if prev_step_info['first_step'] else 'director'
        hs, hs_without_norm, memory, pos = self.transformer(prev_hs, query_embed, samples,
                                                            pos=pos, memory=memory, past_tgt=past_prev_hs,
                                                            past_query_embed=past_query_embed, dec_type=dec_type)
        pred_hs = hs[:, :, :self.token_idx[f'pts_en'], :]
        hs_without_norm = hs_without_norm[:, :, :self.token_idx[f'pts_en'], :]

        if self.traj_train_len > 1:
            prev_step_info['past_prev_hs'] = prev_step_info['past_prev_hs'].transpose(0, 1)

        # get branch positions and radii output
        point_hs = pred_hs[:, :, self.token_idx['pts_idxs'], :]
        if dec_type == 'producer':
            outputs_direc = torch.stack([layer_direc_embed(layer_hs)
                                         for layer_direc_embed, layer_hs in zip(self.direction_embed[:self.num_prod_dec_layers], point_hs)])
            outputs_rad = torch.stack([layer_rad_embed(layer_hs)
                                       for layer_rad_embed, layer_hs in zip(self.radius_embed[:self.num_prod_dec_layers], point_hs)])
        else:
            outputs_direc = torch.stack([layer_direc_embed(layer_hs)
                                         for layer_direc_embed, layer_hs in zip(self.direction_embed[self.num_prod_dec_layers:], point_hs)])
            outputs_rad = torch.stack([layer_rad_embed(layer_hs)
                                       for layer_rad_embed, layer_hs in zip(self.radius_embed[self.num_prod_dec_layers:], point_hs)])

        outputs_direc = outputs_direc.tanh()
        outputs_rad = outputs_rad.sigmoid()
        out = {'direc_logits': outputs_direc[-1], 'rad_logits': outputs_rad[-1]}

        # for last step, compute divergence and end output
        if step == 1:
            idxs = torch.arange(self.num_init_branches, device=device)
            pair_indices = torch.cartesian_prod(idxs, idxs)
            pair_indices = pair_indices[pair_indices[:, 0] != pair_indices[:, 1]]
            div_embeds = self.div_embed
            branch_hs = pred_hs[-1:, :, self.token_idx['div_idxs'], :]
            branch1 = branch_hs[:, :, pair_indices[:, 0], :]
            branch2 = branch_hs[:, :, pair_indices[:, 1], :]
            branch_pair = torch.cat([branch1, branch2], dim=-1)
            outputs_div = torch.stack([layer_div_embed(layer_hs)
                                       for layer_div_embed, layer_hs in zip(div_embeds, branch_pair)])
            outputs_div = outputs_div.sigmoid()
            out['div_logits'] = outputs_div[-1]

            end_embeds = self.end_embed
            branch_end_hs = pred_hs[-1:, :, self.token_idx['end_idxs'], :]
            outputs_end = torch.stack([layer_end_embed(layer_hs)
                                       for layer_end_embed, layer_hs in zip(end_embeds, branch_end_hs)])
            outputs_end = outputs_end.sigmoid()
            out['end_logits'] = outputs_end[-1]

        prev_step_info['hs_without_norm'] = hs_without_norm[-1].detach()

        if prev_step_info['memory'] is None and memory is not None:
            prev_step_info['memory'] = memory
            prev_step_info['pos'] = pos

        prev_step_info['out'] = out
        prev_step_info['first_step'] = False

        return prev_step_info

class SetCriterion(nn.Module):
    """ This class computes the loss for RefTr.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise position, radius,
        divergence and end position)
    """

    def __init__(self, losses, num_init_branches=20, cost_direction=1, cost_radius=1,
                 train_div_non_bif_prob=0.4, train_div_bif_prob=1.0, weight_dict=None):
        """ Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_init_branches: number of branches (token sets) used for tracking.
            cost_direction: cost weight of branch positions.
            cost_radius: cost weight of branch radius.
            train_div_non_bif_prob: probability of training a divergence sample for
            targets with not bifurcation (single branch).
            train_div_bif_prob: probability of training a divergence sample for
            targets with bifurcation (multiple branches).
            weight_dict: dictionary containing the weights for each loss.
        """
        super().__init__()
        self.losses = losses
        self.num_init_branches = num_init_branches
        self.cost_direction = cost_direction
        self.cost_radius = cost_radius
        self.train_div_non_bif_prob = train_div_non_bif_prob
        self.train_div_bif_prob = train_div_bif_prob
        self.weight_dict = weight_dict

        # get the indices corresponding to div outputs
        num_branches = self.num_init_branches
        idxs = torch.arange(num_branches)
        all_pred_pair_idxs = torch.cartesian_prod(idxs, idxs)
        all_pred_pair_idxs = all_pred_pair_idxs[all_pred_pair_idxs[:, 0] != all_pred_pair_idxs[:, 1]]
        self.all_pred_pair_idxs = all_pred_pair_idxs.to('cuda')

    def loss_direction(self, outputs, targets, num_points):
        """
        Compute the losses related to the point positions for each branch.
        """
        assert 'pred_pos' in outputs
        pred = outputs['pred_pos']
        tgt = targets['tgt_pos']
        pred = torch.cat([pred[i] for i in range(len(pred))], dim=0)
        tgt = torch.cat([tgt[i] for i in range(len(tgt))], dim=0)

        # compute L1 loss
        loss_direc = F.l1_loss(pred, tgt, reduction='none')
        losses = {'loss_direction': loss_direc.sum() / num_points}

        return losses

    def loss_radius(self, outputs, targets, num_points):
        """
        Compute the L1 regression loss related to the radius of the vessel
        """
        assert 'pred_rad' in outputs
        pred = outputs['pred_rad']
        tgt = targets['tgt_rad']
        pred = torch.cat([pred[i] for i in range(len(pred))], dim=0)
        tgt = torch.cat([tgt[i] for i in range(len(tgt))], dim=0)

        loss_radii = F.l1_loss(pred, tgt, reduction='none')
        losses = {'loss_radius': loss_radii.sum() / num_points}

        return losses

    def loss_divergence(self, outputs, targets, num_combs):
        """
        Compute the L1 regression loss related to the divergence position of any pair of branches
        """
        assert 'pred_div' in outputs
        pred = outputs['pred_div']
        tgt = targets['tgt_div']
        pred = torch.cat([pred[i] for i in range(len(pred))], dim=0)
        tgt = torch.cat([tgt[i] for i in range(len(tgt))], dim=0)

        loss_div = F.l1_loss(pred, tgt, reduction='none')
        losses = {'loss_divergence': loss_div.sum() / num_combs}

        return losses

    def loss_end(self, outputs, targets, num_branches):
        """
        Compute the L1 regression loss related to the end position of any pair of branches
        """
        assert 'pred_end' in outputs
        pred = outputs['pred_end']
        tgt = targets['tgt_end']
        pred = torch.cat([pred[i] for i in range(len(pred))], dim=0)
        tgt = torch.cat([tgt[i] for i in range(len(tgt))], dim=0)

        loss_end = F.l1_loss(pred, tgt, reduction='none')
        losses = {'loss_end': loss_end.sum() / num_branches}

        return losses

    def get_loss(self, loss, outputs, targets, num_points, num_combs, num_branches):
        loss_map = {
            'direction': self.loss_direction,
            'divergence': self.loss_divergence,
            'end': self.loss_end,
            'radius': self.loss_radius
        }
        if loss == 'divergence':
            num_points = num_combs
        elif loss == 'end':
            num_points = num_branches
        elif loss == 'class':
            num_points = num_branches
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_points)

    @staticmethod
    def compute_cost(target, prediction):
        target = target.unsqueeze(1)
        prediction = prediction.unsqueeze(0)
        cost = torch.sum(torch.abs(target - prediction), dim=(-2, -1))

        return cost

    @staticmethod
    def hungarian_matching(C):
        """
        Performs many to one Hungarian matching to ensure all points in B are matched to A,
        given that n > m (more points in B than in A).

        Parameters:
        C (numpy array): Cost matrix of shape (m, n) where m < n.

        Returns:
        tuple: Two lists containing the matched indices in A and B.
        """
        m, n = C.shape
        C = C.copy()

        if m == n:
            matched_A, matched_B = linear_sum_assignment_with_inf(C)
        elif m == 1:
            matched_A = [0 for _ in range(n)]
            matched_B = list(range(n))
        elif m < n:
            num_repeats = np.ceil(n / m).astype(int)
            extended_C = np.tile(C, (num_repeats, 1))
            # Truncate the extended cost matrix to the size of B
            extended_C = extended_C[:n]
            # create a mapping dict from the extended indices to the original indices
            mapping_dict = {i: i % m for i in range(extended_C.shape[0])}
            matched_A, matched_B = linear_sum_assignment_with_inf(extended_C)
            matched_A = [mapping_dict[i] for i in matched_A]
        else:
            raise ValueError("Number of points in A should be less than or equal to the number of points in B!")
        matched_A, matched_B = np.array(matched_A), np.array(matched_B)

        return matched_A, matched_B

    @torch.no_grad()
    def compute_matches(self, outputs, targets):
        indices = []
        device = outputs['direc_logits'].device
        for s_i, target in enumerate(targets):
            # position cost matrix
            output_pos = outputs['direc_logits'][s_i]
            output_pos = output_pos.reshape(self.num_init_branches, -1, 3)
            target_pos = torch.stack(target['target_branches'])
            cost_pos = self.compute_cost(target_pos, output_pos)
            cost_matrix = self.cost_direction * cost_pos

            if self.cost_radius:
                output_rad = outputs['rad_logits'][s_i]
                output_rad = output_rad.reshape(self.num_init_branches, -1, 1)
                target_rad = torch.stack(target['target_radii'])
                cost_rad = self.compute_cost(target_rad, output_rad)
                cost_matrix += (self.cost_radius * cost_rad)

            matched_tgt, matched_pred = self.hungarian_matching(cost_matrix.cpu().numpy())
            indices.append([torch.tensor(matched_tgt, dtype=torch.int64).to(device),
                            torch.tensor(matched_pred, dtype=torch.int64).to(device)])

        # sort the indices according to the pred indices
        for sample in indices:
            if torch.any(sample[1] != torch.sort(sample[1])[0]):
                perm = torch.argsort(sample[1])
                for i in range(2):
                    sample[i] = sample[i][perm]

        return indices

    def compute_step_targets_preds(self, indices, targets, outputs):
        matched_targets_pos = []
        matched_targets_rad = []
        matched_targets_div = []
        matched_targets_div_idxs = []
        matched_targets_end = []
        matched_preds_pos = []
        matched_preds_rad = []
        matched_preds_div = []
        matched_preds_end = []

        num_branches = self.num_init_branches
        device = outputs['direc_logits'].device
        bs, num_pos_tokens, _ = outputs['direc_logits'].shape
        seq_len = num_pos_tokens // num_branches
        outputs['direc_logits'] = outputs['direc_logits'].reshape(bs, num_branches, seq_len, -1)
        outputs['rad_logits'] = outputs['rad_logits'].reshape(bs, num_branches, seq_len, -1)

        # get the target and pred pos, rad, div and end
        for s_i, index in enumerate(indices):
            all_tgt_pos = torch.stack(targets[s_i]['target_branches'])
            all_tgt_rad = torch.stack(targets[s_i]['target_radii'])
            matched_targets_pos.append(all_tgt_pos[index[0]].reshape(-1, 3))
            matched_targets_rad.append(all_tgt_rad[index[0]].reshape(-1, 1))
            matched_preds_pos.append(outputs['direc_logits'][s_i][index[1]].reshape(-1, 3))
            matched_preds_rad.append(outputs['rad_logits'][s_i][index[1]].reshape(-1, 1))

            lookup = torch.full((self.num_init_branches,), -1).to(device)
            lookup[index[1]] = index[0]
            all_tgt_pair_idxs = lookup[self.all_pred_pair_idxs]

            target_div = targets[s_i]['divergence_matrix']
            mat_tgt_div = target_div[all_tgt_pair_idxs[:, 0], all_tgt_pair_idxs[:, 1]].reshape(-1, 1)
            random_value = torch.rand(mat_tgt_div.shape[0], device=device)
            if target_div.size(0) == 1:
                div_idxs = random_value < self.train_div_non_bif_prob
            else:
                div_idxs = random_value < self.train_div_bif_prob
            mat_tgt_div = mat_tgt_div[div_idxs]
            matched_targets_div.append(mat_tgt_div)
            if 'div_logits' in outputs:
                mat_pred_div = outputs['div_logits'][s_i]
                mat_pred_div = mat_pred_div[div_idxs]
                matched_preds_div.append(mat_pred_div)
            matched_targets_div_idxs.append(div_idxs)
            target_end = targets[s_i]['end_position']
            matched_targets_end.append(target_end[index[0]].reshape(-1, 1))
            if 'end_logits' in outputs:
                matched_preds_end.append(outputs['end_logits'][s_i])

        matched_targets = {'tgt_pos': matched_targets_pos,
                           'tgt_rad': matched_targets_rad,
                           'tgt_div': matched_targets_div,
                           'tgt_div_idxs': matched_targets_div_idxs,
                           'tgt_end': matched_targets_end,
                           'indices': indices}
        matched_preds = {
            'pred_pos': matched_preds_pos,
            'pred_rad': matched_preds_rad}
        if 'div_logits' in outputs:
            matched_preds['pred_div'] = matched_preds_div
        if 'end_logits' in outputs:
            matched_preds['pred_end'] = matched_preds_end

        return matched_targets, matched_preds

    def compute_step_preds(self, indices, div_idxs, outputs):
        matched_preds_pos = []
        matched_preds_rad = []
        matched_preds_div = []
        matched_preds_end = []

        num_branches = self.num_init_branches
        bs, num_pos_tokens, _ = outputs['direc_logits'].shape
        seq_len = num_pos_tokens // num_branches
        outputs['direc_logits'] = outputs['direc_logits'].reshape(bs, num_branches, seq_len, -1)
        outputs['rad_logits'] = outputs['rad_logits'].reshape(bs, num_branches, seq_len, -1)

        for s_i, index in enumerate(indices):
            # get the pred pos, rad, div and end
            matched_preds_pos.append(outputs['direc_logits'][s_i][index[1]].reshape(-1, 3))
            matched_preds_rad.append(outputs['rad_logits'][s_i][index[1]].reshape(-1, 1))
            if 'div_logits' in outputs:
                mat_pred_div = outputs['div_logits'][s_i]
                mat_pred_div = mat_pred_div[div_idxs[s_i]]
                matched_preds_div.append(mat_pred_div)
            if 'end_logits' in outputs:
                matched_preds_end.append(outputs['end_logits'][s_i])

        matched_preds = {
            'pred_pos': matched_preds_pos,
            'pred_rad': matched_preds_rad}
        if 'div_logits' in outputs:
            matched_preds['pred_div'] = matched_preds_div
        if 'end_logits' in outputs:
            matched_preds['pred_end'] = matched_preds_end

        return matched_preds

    def compute_curr_step_tgt_pred(self, outputs, targets, matched=False):
        if matched:
            matched_targets = targets
            indices = matched_targets['indices']
            div_idxs = matched_targets['tgt_div_idxs']
            matched_preds = self.compute_step_preds(indices, div_idxs, outputs)
        else:
            indices = self.compute_matches(outputs, targets)
            matched_targets, matched_preds = self.compute_step_targets_preds(indices, targets, outputs)

        return matched_targets, matched_preds

    def compute_losses(self, targets, preds):
        # Compute the average number of targets for normalization purposes
        num_points = sum([len(v) for v in targets["tgt_pos"]])
        num_combs = sum([len(v) for v in targets["tgt_div"]])
        num_branches = sum([len(v) for v in targets["tgt_end"]])
        device = next(iter(preds.values()))[0].device
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=device)
        num_combs = torch.as_tensor([num_combs], dtype=torch.float, device=device)
        num_branches = torch.as_tensor([num_branches], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
            torch.distributed.all_reduce(num_combs)
            torch.distributed.all_reduce(num_branches)
            num_points = torch.clamp(num_points / get_world_size(), min=1).item()
            num_combs = torch.clamp(num_combs / get_world_size(), min=1).item()
            num_branches = torch.clamp(num_branches / get_world_size(), min=1).item()
        else:
            num_points = torch.clamp(num_points, min=1).item()
            num_combs = torch.clamp(num_combs, min=1).item()
            num_branches = torch.clamp(num_branches, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'divergence' and 'pred_div' not in preds:
                continue
            if loss == 'end' and 'pred_end' not in preds:
                continue
            losses.update(self.get_loss(loss, preds, targets, num_points, num_combs, num_branches))

        return losses

    def forward(self, prev_step_info, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        prev_outputs = prev_step_info['prev_out']
        outputs = prev_step_info['out']

        if prev_outputs is None:
            targets, preds = self.compute_curr_step_tgt_pred(outputs, targets)
            # store the matched targets for the next step
            prev_step_info['matched_targets'] = targets
        else:
            targets = prev_step_info['matched_targets']
            targets, preds = self.compute_curr_step_tgt_pred(outputs, targets, matched=True)
        losses.update(self.compute_losses(targets, preds))

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class MLPVar(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        self.num_layers = len(hidden_dims) + 1
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x
