import pickle
import monai.transforms
import numpy as np
import torch
import re
import copy
import networkx as nx
from frechetdist import frdist
from scipy.interpolate import Rbf
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from scipy.stats import truncnorm, laplace
from scipy.interpolate import interp1d
import os
os.environ["BIGTREE_CONF_ASSERTIONS"] = ""
from bigtree import find_name, levelordergroup_iter, preorder_iter, find_full_path, Node
from monai.utils import convert_to_tensor, pytorch_after
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
    Crop,
    Pad,
    LoadImage,
    ScaleIntensity,
    NormalizeIntensity,
    ThresholdIntensity)


class LoadAnnotPickle(Transform):
    def __init__(self):
        pass

    def __call__(self, input):
        with open(input, 'rb') as handle:
            data = pickle.load(handle)
        data['index'] = [int(s) for s in re.findall(r'\d+', input)][-1]
        return data


class LoadAnnotPickled(MapTransform):
    def __init__(self, keys,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadAnnotPickle()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d


class CropAndPad(Transform):
    def __init__(self, sub_vol_size):
        self.crop = Crop()
        self.pad = Pad()
        self.sub_vol_size = sub_vol_size

    @staticmethod
    def compute_slices(roi_center, roi_size, img_size):
        """
        Compute the crop slices based on specified `center & size` or `start & end` or `slices`.

        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI
            img_size: size of the whole image

        Outputs:
            slices: list of slices for each of the spatial dimensions.
            padding: list of padding for each of the spatial dimensions.
        """

        assert roi_center is not None and roi_size is not None and img_size is not None
        img_size_t = convert_to_tensor(data=img_size, dtype=torch.int16, wrap_sequence=True, device="cuda")
        roi_center_t = convert_to_tensor(data=roi_center, dtype=torch.int16, wrap_sequence=True, device="cuda")
        roi_size_t = convert_to_tensor(data=roi_size, dtype=torch.int16, wrap_sequence=True, device="cuda")
        _zeros = torch.zeros_like(roi_center_t)
        half = (
            torch.divide(roi_size_t, 2, rounding_mode="floor")
            if pytorch_after(1, 8)
            else torch.floor_divide(roi_size_t, 2)
        )
        roi_start_t = roi_center_t - half
        roi_end_t = roi_start_t + roi_size_t
        roi_start_clipped_t = torch.maximum(roi_start_t, _zeros)
        roi_end_clipped_t = torch.minimum(roi_end_t, img_size_t)
        roi_start_pad_t = torch.abs(torch.minimum(roi_start_t, _zeros))
        roi_end_pad_t = torch.maximum(roi_end_t - img_size_t, _zeros)

        slices = [slice(int(s), int(e)) for s, e in zip(roi_start_clipped_t.tolist(), roi_end_clipped_t.tolist())]
        padding = [(0, 0)] + [(int(s), int(e)) for s, e in zip(roi_start_pad_t.tolist(), roi_end_pad_t.tolist())]

        return slices, padding

    def extract_image_crop(self, image_data, node_position, image_min):
        crop_slices, crop_padding = self.compute_slices(node_position,
                                                        self.sub_vol_size,
                                                        image_data.shape[-3:])
        cropped_image = self.crop(image_data, crop_slices)
        cropped_image = self.pad(cropped_image, crop_padding, value=image_min)

        return cropped_image

    def __call__(self, data, position, image_min):
        data = self.extract_image_crop(data, position, image_min)

        return data


class CropAndPadd(MapTransform):
    def __init__(self, keys, sub_vol_size):
        super().__init__(keys)
        self.transform = CropAndPad(sub_vol_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            images = []
            for target in d['label']:
                if key == 'image':
                    images.append(self.transform(d[key], target['root_position'], d['image_min'].item()))
                elif key == 'mask':
                    images.append(self.transform(d[key], target['root_position'], 0.0))
                else:
                    raise NotImplementedError
            d[key] = images

        return d


class ComputeImageMin(Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        data = torch.min(data)

        return data


class ComputeImageMax(Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        data = torch.max(data)

        return data


class ComputeImageStatsd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.transform_min = ComputeImageMin()
        self.transform_max = ComputeImageMax()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d['image_min'] = torch.min(d[key])
            d['image_max'] = torch.max(d[key])
        return d


class ExtRandomSubTree(Randomizable, Transform):
    def __init__(self, root_prob, bifur_prob, end_prob, seq_len, num_prev_pos,
                 traj_train_len):
        self.root_prob = root_prob
        self.bifur_prob = bifur_prob
        self.end_prob = end_prob
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.traj_train_len = traj_train_len
        self.traj_train_num_pts = (traj_train_len * (seq_len - 1)) + 1

    def randomize(self, data):
        """
        Pick a random point using curated sampling
        """
        num_trees = len(data['branches'])
        tree_id = self.R.randint(low=0, high=num_trees, size=1)[0]
        full_tree = data['branches'][tree_id]

        num_traj = len(data['trajectories'][tree_id])
        traj_id = self.R.randint(low=0, high=num_traj, size=1)[0]
        selected_trajectory = data['trajectories'][tree_id][traj_id]
        path = selected_trajectory['path']
        bifur_ids = selected_trajectory['bifur_ids']
        endpt_id = selected_trajectory['endpt_id']
        endpt_index = path.index(endpt_id)
        prob = self.R.random()

        if prob < self.bifur_prob:
            bifur_indices = [path.index(bifur) for bifur in bifur_ids]
            candidates = set()
            for bifur_index in bifur_indices:
                start = max(0, bifur_index - self.traj_train_num_pts + 1)
                end = bifur_index + 1
                candidates.update(range(start, end))
            max_valid_index = endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1))
            valid_candidates = [idx for idx in candidates if idx <= max_valid_index]
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'bifurcation'

        elif prob < (self.bifur_prob + self.end_prob):
            max_valid_index = endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1))
            min_valid_index = max(0, endpt_index - self.traj_train_num_pts + 1)
            valid_candidates = list(range(min_valid_index, max_valid_index + 1))
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'end'

        elif prob < (self.bifur_prob + self.end_prob + self.root_prob):
            min_valid_index = 0
            max_valid_index = min(self.seq_len - 1, endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1)))
            valid_candidates = list(range(min_valid_index, max_valid_index + 1))
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'root'

        else:
            endpt_index = path.index(endpt_id)
            min_valid_index = 0
            max_valid_index = endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1))
            valid_candidates = list(range(min_valid_index, max_valid_index + 1))
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'random'

        assert point_id is not None, "Selected node not found!"
        selected_node = find_name(full_tree, point_id)

        output = {'tree_id': tree_id,
                  'traj_id': traj_id,
                  'selected_path': path,
                  'selected_node': selected_node,
                  'point_id': point_id,
                  'point_type': point_type}

        return output

    @staticmethod
    def select_nearby_point(full_tree, point_id, dist):
        selected_node = find_name(full_tree, point_id)
        for _ in range(dist):
            parent = selected_node.parent
            if parent is not None:
                selected_node = parent
            else:
                continue

        return selected_node

    @staticmethod
    def get_past_traj_start(selected_node, num_prev_pos):
        past_traj_head_node = selected_node
        for past_nodes_num in range(num_prev_pos):
            if past_traj_head_node.parent is not None:
                past_traj_head_node = past_traj_head_node.parent
            else:
                break

        return past_traj_head_node, past_nodes_num

    @staticmethod
    def create_subtree_with_past_tr_val(selected_node, past_traj_num, seq_len):
        root_node = Node(selected_node.name,
                         position=selected_node.position,
                         radius=selected_node.radius,
                         label=selected_node.label)

        for i, level in enumerate(levelordergroup_iter(selected_node)):
            if i == 0:
                continue
            elif i <= seq_len:
                for node in level:
                    parent_node = find_name(root_node, node.parent.name)
                    Node(node.name,
                         position=node.position,
                         radius=node.radius,
                         parent=parent_node,
                         label=node.label)
            else:
                break

        if selected_node.parent is not None:
            current_node = selected_node.parent
            past_tr_end_node = Node(current_node.name,
                                    position=current_node.position,
                                    radius=current_node.radius,
                                    label=current_node.label)

            past_tr_head_node = past_tr_end_node
            for _ in range(1, past_traj_num):
                current_node = current_node.parent
                past_tr_head_node.parent = Node(current_node.name,
                                                position=current_node.position,
                                                radius=current_node.radius,
                                                label=current_node.label)
                past_tr_head_node = past_tr_head_node.parent
            root_node.parent = past_tr_end_node
        else:
            past_tr_head_node = root_node

        return root_node, past_tr_head_node

    def create_subtree_with_past_tr(self, selected_node, selected_path, max_root_buffer_nodes):
        selected_node_index = selected_path.index(selected_node.name)
        past_traj_start_index = max(selected_node_index - (self.num_prev_pos - 1), 0)
        num_prev_pos = selected_node_index - past_traj_start_index + 1
        buffer_start_index = max(past_traj_start_index - max_root_buffer_nodes, 0)
        num_root_buffer_nodes = past_traj_start_index - buffer_start_index
        buffer_start_node = find_name(selected_node.root, selected_path[buffer_start_index])

        new_root_node = Node(buffer_start_node.name,
                         position=buffer_start_node.position,
                         radius=buffer_start_node.radius,
                         label=buffer_start_node.label)

        new_level = None
        prev_level = None
        path_end_level = None
        end_nodes = []
        new_selected_node = None
        new_past_tr_head_node = None
        max_levels = self.traj_train_num_pts + num_prev_pos + num_root_buffer_nodes
        for i, level in enumerate(levelordergroup_iter(buffer_start_node)):
            if i == 0:
                continue

            if i == num_root_buffer_nodes + 1:
                if new_level:
                    new_past_tr_head_node = [node for node in new_level if node.name in selected_path][0]
                else:
                    new_past_tr_head_node = new_root_node

            if i == num_root_buffer_nodes + num_prev_pos:
                if new_level:
                    new_selected_node = [node for node in new_level if node.name in selected_path][0]
                else:
                    new_selected_node = new_root_node

            if i == max_levels:
                break

            if i == max_levels - 1:
                path_end_level = prev_level

            new_level = []
            for node in level:
                parent_node = find_name(new_root_node, node.parent.name)
                new_level.append(Node(node.name,
                     position=node.position,
                     radius=node.radius,
                     parent=parent_node,
                     label=node.label))

                if len(node.children) == 0:
                    end_nodes.append(node)

            prev_level = list(level)

        if path_end_level is None:
            path_end_level = prev_level

        if new_selected_node is None:
            new_selected_node = [node for node in new_level if node.name in selected_path][0]
        selected_node_index = selected_path.index(new_selected_node.name)
        path_end_node_name = [node.name for node in path_end_level + end_nodes
                              if node.name in selected_path][0]
        path_end_node_index = selected_path.index(path_end_node_name)
        selected_path = selected_path[selected_node_index:path_end_node_index + 1]

        return new_selected_node, new_past_tr_head_node, selected_path, num_prev_pos

    def __call__(self, data):
        output = self.randomize(data)
        max_root_buffer_nodes = 10
        (selected_node, past_traj_head_node,
         selected_path, num_prev_pos) = self.create_subtree_with_past_tr(
            output['selected_node'], output['selected_path'], max_root_buffer_nodes)

        data = {'index': data['index'],
                'tree_id': output['tree_id'],
                'traj_id': output['traj_id'],
                'point_id': output['point_id'],
                'point_type': output['point_type'],
                'selected_path': selected_path,
                'selected_node': selected_node,
                'past_traj_head_node': past_traj_head_node,
                'num_prev_pos': num_prev_pos,
                }

        return data


class ExtRandomSubTreed(Randomizable, MapTransform):
    def __init__(self, keys, root_prob, bifur_prob, end_prob, seq_len, num_prev_pos,
                 traj_train_len):
        super(Randomizable, self).__init__(keys)
        self.transform = ExtRandomSubTree(root_prob, bifur_prob, end_prob,
                                          seq_len, num_prev_pos, traj_train_len)

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class DivideSubTree(Transform):
    def __init__(self, seq_len, num_prev_pos, traj_train_len):
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.traj_train_len = traj_train_len
        self.traj_train_num_pts = (traj_train_len * (seq_len - 1)) + 1

    # get starting points for the sub volumes
    def get_starting_points(self, data):
        # get index of selected node in path
        selected_node_list = [data['selected_node']]
        point_index = data['selected_path'].index(data['selected_node'].node_name)

        for i in range(1, self.traj_train_len):
            next_point_id = data['selected_path'][point_index + (self.seq_len - 1) * i]
            next_node = find_name(data['selected_node'], next_point_id)

            assert next_node is not None, "Selected node not found!"
            selected_node_list.append(next_node)

        return selected_node_list

    @staticmethod
    def get_past_traj_start(selected_node, num_prev_pos):
        past_traj_head_node = selected_node
        for past_nodes_num in range(num_prev_pos):
            if past_traj_head_node.parent is not None:
                past_traj_head_node = past_traj_head_node.parent
            else:
                break

        return past_traj_head_node, past_nodes_num

    @staticmethod
    def create_subtree_with_past_tr(selected_node, past_traj_num, seq_len):
        # Create the main subtree
        # initialize the root node for the subtree
        root_node = Node(selected_node.name,
                         position=selected_node.position,
                         radius=selected_node.radius,
                         label=selected_node.label)

        # use level order traversal to go through each level and add the nodes to the subtree
        for i, level in enumerate(levelordergroup_iter(selected_node)):
            if i == 0:
                continue
            # here we also include one more level than required to be able to get the correct num of children for end points of sequences
            elif i <= seq_len:
                for node in level:
                    parent_node = find_name(root_node, node.parent.name)
                    Node(node.name,
                         position=node.position,
                         radius=node.radius,
                         parent=parent_node,
                         label=node.label)
            else:
                break

        # Create the past trajectory tree
        # We build this tree backwards to only keep relevant nodes to the selected node
        # check that selected_node's parent is not none and then start with it
        if selected_node.parent is not None:
            current_node = selected_node.parent
            # create a copy of the last node of past trajectory
            past_tr_end_node = Node(current_node.name,
                                    position=current_node.position,
                                    radius=current_node.radius,
                                    label=current_node.label)

            # the last node becomes the head node to which we will recursively add parents from the main tree
            past_tr_head_node = past_tr_end_node
            for i in range(1, past_traj_num):
                # update current_node to its parent
                current_node = current_node.parent
                # Update past_tr_head_node's parent from None to current_node
                past_tr_head_node.parent = Node(current_node.name,
                                                position=current_node.position,
                                                radius=current_node.radius,
                                                label=current_node.label)
                # update past_tr_head_node to its parent
                past_tr_head_node = past_tr_head_node.parent

            # use past_tr_end_node to append this past trajectory tree to the subtree
            root_node.parent = past_tr_end_node
        else:
            past_tr_head_node = root_node

        return root_node, past_tr_head_node

    def __call__(self, data):
        # get starting points of the sub volumes
        selected_node_list = self.get_starting_points(data)

        selected_nodes, past_traj_head_nodes = [], []
        for selected_node_orig in selected_node_list:
            # get the starting node for past trajectory
            _, past_traj_num = self.get_past_traj_start(selected_node_orig, self.num_prev_pos)

            # create new subtree with past_traj_head_node as root node
            selected_node, past_traj_head_node = self.create_subtree_with_past_tr(selected_node_orig,
                                                                                  past_traj_num, self.seq_len)
            selected_nodes.append(selected_node)
            past_traj_head_nodes.append(past_traj_head_node)

        data['selected_node'] = selected_nodes
        data['past_traj_head_node'] = past_traj_head_nodes

        return data


class DivideSubTreed(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, traj_train_len):
        super().__init__(keys)
        self.transform = DivideSubTree(seq_len, num_prev_pos, traj_train_len)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class ConvertTreeToTargets(Randomizable, Transform):
    def __init__(self, seq_len, num_prev_pos, sub_vol_size, focus_vol_size):
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.sub_vol_size = sub_vol_size
        self.focus_vol_size = focus_vol_size

    def generate_past_trajectory_pp(self, root_node):
        past_traj_label = []

        # the root_node's position is also included
        parent = root_node

        for i in range(self.num_prev_pos):
            node_info_list = []
            node_position = np.asarray(parent.position).astype(float)
            node_position -= (self.sub_vol_size // 2)
            node_position /= (self.focus_vol_size // 2)
            node_info_list.append(node_position)
            node_radius = np.asarray(parent.radius).reshape(1)
            node_radius /= (self.sub_vol_size // 2)
            node_info_list.append(node_radius)
            past_traj_label.append(np.concatenate(node_info_list))
            parent = parent.parent
            if parent is None:
                break

        past_traj_label_np = np.asarray(past_traj_label)
        num_missing_points = self.num_prev_pos - past_traj_label_np.shape[0]
        if num_missing_points:
            last_point_info = past_traj_label_np[-1:]
            padding_info = np.tile(last_point_info, (num_missing_points, 1))
            past_traj_label_np = np.vstack((past_traj_label_np, padding_info))

        past_traj_label_t = torch.FloatTensor(past_traj_label_np).flatten()

        return past_traj_label_t

    def radius_extrapolation(self, radii):
        """
        Extrapolate radii to match position extrapolation
        """
        radii = np.array(radii)
        n = radii.shape[0]

        if n < self.seq_len + 1:
            # Generate new points with a step size of 1
            new_radii = np.array([0.0 for _ in range(1, self.seq_len - n + 2)]).reshape(-1, 1)
            extrapolated_radii = np.concatenate((radii, new_radii), axis=0)
        elif n > self.seq_len + 1:
            extrapolated_radii = radii[:self.seq_len + 1]
        else:
            extrapolated_radii = radii
        start_idx = 2

        return extrapolated_radii[start_idx:]

    def position_extrapolation(self, points):
        extrapolated_points = self.last_pos_extrapolation(points)

        return extrapolated_points

    def last_pos_extrapolation(self, points):
        """
                Performs linear extrapolation on a set of 3D points with a step size of 1.

                Parameters:
                points (numpy array): Array of 3D points, shape (n, 3)
                self.seq_len (int): Total number of output points (including original ones)

                Returns:
                numpy array: Extrapolated 3D points of shape (num_output_points, 3)
                """
        points = np.array(points)
        n = points.shape[0]

        if n < self.seq_len + 1:
            # Use the last position to pad/extrapolate
            new_points = [points[-1] for _ in range(1, self.seq_len - n + 2)]
            extrapolated_points = np.vstack((points, new_points))
        elif n > self.seq_len + 1:
            extrapolated_points = points[:self.seq_len + 1]
        else:
            extrapolated_points = points
        start_idx = 2

        return extrapolated_points[start_idx:]

    def expand_list(self, list_to_expand):
        n = len(list_to_expand)

        if n < self.seq_len + 1:
            # Generate new points with a step size of 1
            new_points = [list_to_expand[-1] for _ in range(1, self.seq_len - n + 2)]
            expanded_list = list_to_expand + new_points
        elif n > self.seq_len + 1:
            expanded_list = list_to_expand[:self.seq_len + 1]
        else:
            expanded_list = list_to_expand

        return expanded_list[1:]

    @staticmethod
    def hungarian_matching(C):
        """
        Performs iterative Hungarian matching to ensure all points in B are matched to A,
        given that n > m (more points in B than in A).

        Parameters:
        C (numpy array): Cost matrix of shape (m, n) where m < n.

        Returns:
        tuple: Two lists containing the matched indices in A and B.
        """
        m, n = C.shape
        C = C.copy()  # Avoid modifying the original cost matrix

        if m == n:
            matched_A, matched_B = linear_sum_assignment(C)
        elif m == 1:
            matched_A = [0 for _ in range(n)]
            matched_B = list(range(n))
        elif m < n:
            num_repeats = np.ceil(n / m).astype(int)
            extended_C = np.tile(C, (num_repeats, 1))
            extended_C = extended_C[:n]  # Truncate the extended cost matrix to the size of B
            # create a mapping dict from the extended indices to the original indices
            mapping_dict = {i: i % m for i in range(extended_C.shape[0])}
            matched_A, matched_B = linear_sum_assignment(extended_C)
            matched_A = [mapping_dict[i] for i in matched_A]
        else:
            raise ValueError("Number of points in A should be less than or equal to the number of points in B!")

        return matched_A, matched_B

    def matcher(self, C):
        return self.hungarian_matching(C)

    @staticmethod
    def compute_div_matrix(paths):
        paths = np.array(paths)  # Convert to NumPy array for efficient indexing
        n = len(paths)
        div_matrix = np.full((n, n), len(paths[0]) - 1, dtype=np.float32)  # Default to len(paths[0]) - 1

        for i in range(n):
            for j in range(i + 1, n):  # Compute only for i < j to avoid duplicates
                diff_indices = np.where(paths[i] != paths[j])[0]
                if diff_indices.size > 0:
                    div_matrix[i, j] = div_matrix[j, i] = diff_indices[0] - 1

        return div_matrix

    @staticmethod
    def find_end_node(path):
        last_value = path[-1]
        i = len(path) - 1
        while i >= 0 and path[i] == last_value:
            i -= 1
        return i + 1

    def compute_target(self, sample_tree, selected_path, shuffle=True):
        assert sample_tree.parent is not None, "Selected node should not be the root node!"
        start_pos = np.asarray(sample_tree.position)
        start_radius = sample_tree.radius
        center_pos = np.round(start_pos)

        bifur_points = []
        for i, level in enumerate(levelordergroup_iter(sample_tree.root)):
            for node in level:
                if len(node.children) > 1:
                    bifur_points.append(node.name)
                node.position = list(np.array(node.position) + ((self.sub_vol_size // 2) - center_pos))

        sample_tree_nx = nx.DiGraph()
        sample_tree_nx = convert_to_networkx(sample_tree_nx, sample_tree.parent)

        end_points = [node for node in sample_tree_nx.nodes if sample_tree_nx.out_degree(node) == 0]
        if shuffle:
            self.R.shuffle(end_points)
        start_parent_name = sample_tree.parent.name
        paths = [nx.shortest_path(sample_tree_nx, start_parent_name, end_point) for end_point in end_points]
        path_positions = [np.array([sample_tree_nx.nodes[node]['position'] for node in path]) for path in paths]
        path_radii = [np.array([sample_tree_nx.nodes[node]['radius'] for node in path]).reshape(-1, 1) for path in paths]
        path_positions = [torch.tensor(self.position_extrapolation(pos), dtype=torch.float32)
                          for pos in path_positions]
        path_radii = [torch.tensor(self.radius_extrapolation(path), dtype=torch.float32)
                      for path in path_radii]

        # get divergence points
        expanded_paths = [self.expand_list(path) for path in paths]
        divergence_matrix = torch.tensor(self.compute_div_matrix(expanded_paths), dtype=torch.float32)
        end_position = torch.tensor([self.find_end_node(path) for path in expanded_paths], dtype=torch.float32)

        # get the index of end point that is in the selected path
        if selected_path:
            main_target_index = [n_i for n_i, e_path in enumerate(expanded_paths) if e_path[-1] in selected_path][0]
        else:
            main_target_index = None

        # normalize
        for i, pos in enumerate(path_positions):
            pos -= (self.sub_vol_size // 2)
            pos /= (self.focus_vol_size // 2)
        for i, rad in enumerate(path_radii):
            rad /= (self.sub_vol_size // 2)
        divergence_matrix /= (self.seq_len - 1)
        end_position /= (self.seq_len - 1)

        target = {'root_position': start_pos, 'root_radius':  start_radius, 'target_branches': path_positions,
                  'target_radii': path_radii, 'divergence_matrix': divergence_matrix, 'end_position': end_position}
        if selected_path:
            target['main_target_index'] = main_target_index

        return target

    def __call__(self, data):
        # extract subtree in a level order sequence and pad it while keeping order of nodes
        data = [data.copy() for _ in range(len(data['selected_node']))]
        parent_target_index = None
        for idx, label in enumerate(data):
            sample_tree = label['selected_node'][idx]
            target = self.compute_target(sample_tree, label['selected_path'])
            label['past_traj_head_node'] = label['past_traj_head_node'][idx]
            label['selected_node'] = sample_tree
            label['past_tr'] = self.generate_past_trajectory_pp(sample_tree)
            label['parent_target_index'] = parent_target_index
            parent_target_index = target['main_target_index']
            label.update(target)
            label.pop('selected_node')
            label.pop('past_traj_head_node')

        return data


class ConvertTreeToTargetsd(Randomizable, MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, sub_vol_size, focus_vol_size):
        super(Randomizable, self).__init__(keys)
        self.transform = ConvertTreeToTargets(seq_len, num_prev_pos, sub_vol_size, focus_vol_size)

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class ExtRandomSubTreeArch(Randomizable, Transform):
    def __init__(self, root_prob, bifur_prob, end_prob, enable_dist, seq_len, num_prev_pos):
        self.root_prob = root_prob
        self.bifur_prob = bifur_prob
        self.end_prob = end_prob
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.enable_dist = enable_dist

    # pick a random point using curated sampling
    def randomize(self, data):
        prob = self.R.random()

        # sample around bifucation points
        if prob < self.bifur_prob:
            # Cumulative index
            cumulative_index = -1
            index_mapping = {}
            for tree_id, lst in enumerate(data['bifur_ids']):
                for index_within_list, value in enumerate(lst):
                    cumulative_index += 1
                    index_mapping[cumulative_index] = [tree_id, index_within_list]

            point_type = 'bifurcation'
            idx = self.R.randint(low=0, high=cumulative_index, size=1)[0]
            tree_id, point_id_index = index_mapping[idx]
            point_id = data['bifur_ids'][tree_id][point_id_index]

            if self.enable_dist:
                # uniformly sample a distance and pick a point that distance up from the point_id
                dist = self.R.randint(low=0, high=self.seq_len, size=1)[0]
            else:
                dist = 0
            selected_node = self.select_nearby_point(data['branches'][tree_id], point_id, dist)

        # sample around end points
        elif prob < (self.bifur_prob + self.end_prob):
            # Cumulative index
            cumulative_index = -1
            index_mapping = {}
            for tree_id, lst in enumerate(data['endpts_ids']):
                for index_within_list, value in enumerate(lst):
                    cumulative_index += 1
                    index_mapping[cumulative_index] = [tree_id, index_within_list]

            point_type = 'end'
            idx = self.R.randint(low=0, high=cumulative_index, size=1)[0]
            tree_id, point_id_index = index_mapping[idx]
            point_id = data['endpts_ids'][tree_id][point_id_index]

            if self.enable_dist:
                # uniformly sample a distance and pick a point that distance up from the point_id
                dist = self.R.randint(low=0, high=self.seq_len, size=1)[0]
            else:
                dist = 0
            selected_node = self.select_nearby_point(data['branches'][tree_id], point_id, dist)

        # sample around root point
        elif prob < (self.bifur_prob + self.end_prob + self.root_prob):
            tree_id = self.R.randint(low=0, high=len(data['branches']), size=1)[0]
            full_tree = data['branches'][tree_id]
            point_type = 'root'
            point_id = full_tree.node_name
            if self.enable_dist:
                dist = self.R.randint(low=0, high=self.seq_len, size=1)[0]
            else:
                dist = 0
            selected_node = full_tree
            for i, level in enumerate(levelordergroup_iter(full_tree)):
                if i == dist:
                    point_index = self.R.randint(low=0, high=len(level), size=1)[0]
                    selected_node = level[point_index]
                    break

        else:
            tree_id = self.R.randint(low=0, high=len(data['branches']), size=1)[0]
            full_tree = data['branches'][tree_id]
            dist = 0
            point_type = 'random'
            point_id_index = self.R.randint(low=0, high=len(data['all_ids'][tree_id]), size=1)[0]
            point_id = data['all_ids'][tree_id][point_id_index]
            selected_node = find_name(full_tree, point_id)

        assert selected_node is not None, "Selected node not found!"

        output = {'tree_id': tree_id,
                  'selected_node': selected_node,
                  'point_id': point_id,
                  'point_type': point_type,
                  'dist': dist}

        return output

    @staticmethod
    def select_nearby_point(full_tree, point_id, dist):
        selected_node = find_name(full_tree, point_id)
        for i in range(dist):
            parent = selected_node.parent
            if parent is not None:
                selected_node = parent
            else:
                continue

        return selected_node

    @staticmethod
    def get_past_traj_start(selected_node, num_prev_pos):
        past_traj_head_node = selected_node
        for past_nodes_num in range(num_prev_pos):
            if past_traj_head_node.parent is not None:
                past_traj_head_node = past_traj_head_node.parent
            else:
                break

        return past_traj_head_node, past_nodes_num

    @staticmethod
    def create_subtree_with_past_tr(selected_node, past_traj_num, seq_len):
        root_node = Node(selected_node.name,
                         position=selected_node.position,
                         radius=selected_node.radius,
                         label=selected_node.label)

        for i, level in enumerate(levelordergroup_iter(selected_node)):
            if i == 0:
                continue
            elif i <= seq_len:
                for node in level:
                    parent_node = find_name(root_node, node.parent.name)
                    Node(node.name,
                         position=node.position,
                         radius=node.radius,
                         parent=parent_node,
                         label=node.label)
            else:
                break

        if selected_node.parent is not None:
            current_node = selected_node.parent
            past_tr_end_node = Node(current_node.name,
                                    position=current_node.position,
                                    radius=current_node.radius,
                                    label=current_node.label)

            past_tr_head_node = past_tr_end_node
            for i in range(1, past_traj_num):
                current_node = current_node.parent
                past_tr_head_node.parent = Node(current_node.name,
                                                position=current_node.position,
                                                radius=current_node.radius,
                                                label=current_node.label)
                past_tr_head_node = past_tr_head_node.parent
            root_node.parent = past_tr_end_node
        else:
            past_tr_head_node = root_node

        return root_node, past_tr_head_node

    def __call__(self, data):
        output = self.randomize(data)
        _, past_traj_num = self.get_past_traj_start(output['selected_node'], self.num_prev_pos)
        selected_nodes, past_traj_head_nodes = self.create_subtree_with_past_tr(output['selected_node'],
                                                                                past_traj_num, self.seq_len)

        data = {'index': data['index'],
                'tree_id': output['tree_id'],
                'point_id': output['point_id'],
                'point_type': output['point_type'],
                'dist': output['dist'],
                'selected_node': selected_nodes,
                'past_traj_head_node': past_traj_head_nodes
                }

        return data


class ExtRandomSubTreeArchd(Randomizable, MapTransform):
    def __init__(self, keys, root_prob, bifur_prob, end_prob, enable_dist, seq_len, num_prev_pos):
        super(Randomizable, self).__init__(keys)
        self.transform = ExtRandomSubTreeArch(root_prob, bifur_prob, end_prob, enable_dist,
                                              seq_len, num_prev_pos)

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class LoadImageCropsAndTrees(Transform):
    def __init__(self, seq_len, num_prev_pos, sub_vol_size, images, annots, image_mins, masks,
                 focus_vol_size):
        self.images = images
        self.annots = annots
        self.image_mins = image_mins
        self.masks = masks
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.crop_and_pad = CropAndPad(sub_vol_size)
        self.convert_tree_to_targets = ConvertTreeToTargets(seq_len, num_prev_pos, sub_vol_size, focus_vol_size)

    def get_annot_dict(self, selected_node, input_dict):
        data = {}
        _, past_traj_num = ExtRandomSubTreeArch.get_past_traj_start(selected_node, self.num_prev_pos)
        selected_node, past_traj_head_node = ExtRandomSubTreeArch.create_subtree_with_past_tr(selected_node,
                                                                                              past_traj_num, self.seq_len)
        target = self.convert_tree_to_targets.compute_target(selected_node, None, shuffle=False)
        data['past_tr'] = self.convert_tree_to_targets.generate_past_trajectory_pp(selected_node)
        data.update(target)
        data.update(input_dict)

        return data

    def get_image_crops(self, input, selected_node):
        # get the image crop
        image = self.crop_and_pad(self.images[input['sample_id']], selected_node.position,
                                  self.image_mins[input['sample_id']])

        if self.masks is not None:
            mask = self.crop_and_pad(self.masks[input['sample_id']], selected_node.position,
                                     0.0)
        else:
            mask = None

        return image, mask

    def __call__(self, input):
        annot_tree = self.annots[input['sample_id']]['branches'][input['tree_id']]
        selected_node = find_name(annot_tree, input['node_id'])

        # if the selected node is the root node, select the first child
        if selected_node.parent is None:
            selected_node = selected_node.children[0]

        input_dict = {'index': input['index'],
                      'tree_id': input['tree_id'],
                      'point_id': input['node_id'],
                      'point_type': input['point_type'],
                      'dist': input['distance']}

        data = self.get_annot_dict(selected_node, input_dict)
        image, mask = self.get_image_crops(input, selected_node)

        return data, image, mask


class LoadImageCropsAndTreesd(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, sub_vol_size, mask, paths,
                 window_input, window_min, window_max, focus_vol_size, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        image_mins = {}
        images = {}
        annots = {}
        if mask:
            masks = {}
        else:
            masks = None

        image_reader = LoadImage(image_only=True)
        annot_reader = LoadAnnotPickle()
        self.norm = NormalizeIntensity()

        if window_input:
            window = monai.transforms.Compose(
                [ThresholdIntensity(threshold=window_max, above=False, cval=window_max),
                 ThresholdIntensity(threshold=window_min, above=True, cval=window_min)])

        for path in paths:
            index = [int(s) for s in re.findall(r'\d+', path[0])][-1]
            if window_input:
                image = window(image_reader(path[0]).unsqueeze(0))
            else:
                image = image_reader(path[0]).unsqueeze(0)
            image = self.norm(image)
            image_min = torch.min(image)
            images[index] = image
            image_mins[index] = image_min
            annots[index] = annot_reader(path[1])
            if mask:
                masks[index] = image_reader(path[2]).unsqueeze(0)

        self.mask = mask
        self.transform = LoadImageCropsAndTrees(seq_len, num_prev_pos, sub_vol_size, images, annots,
                                                image_mins, masks, focus_vol_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key], image, mask = self.transform(d[key])
            d['image'] = image
            d['mask'] = mask
        return d


def convert_to_networkx(G, tree):
    """
    Function to traverse the bigtree tree and create the NetworkX graph
    """
    G.add_node(tree.node_name, position=tree.position, radius=tree.radius)

    if tree.parent is not None:
        G.add_edge(tree.parent.node_name, tree.node_name)

    for child in tree.children:
        convert_to_networkx(G, child)

    return G