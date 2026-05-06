from dataclasses import dataclass
import torch

from CVRPTopKProblemDef import (
    select_topk_cells,
    make_demand_from_scores,
    get_random_scored_grids,
    augment_xy_data_by_8_fold,
)


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None          # (batch, 1, 2)
    node_xy: torch.Tensor = None           # (batch, top_k, 2)
    node_demand: torch.Tensor = None       # (batch, top_k)
    selected_scores: torch.Tensor = None   # (batch, top_k)
    selected_indices: torch.Tensor = None  # (batch, top_k)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None


class CVRPTopKEnv:
    """POMO-style CVRP environment for a top-k road-dust routing problem.

    Pipeline:
        22500 scored grid cells -> select top_k cells -> convert scores to demand -> CVRP.

    Node 0 is the depot. Nodes 1..top_k are selected cleaning targets.
    Vehicle capacity is normalized to 1. Returning to depot refills load to 1.
    """

    def __init__(self, **env_params):
        self.env_params = env_params
        self.total_grid_size = env_params.get('total_grid_size', 22500)
        self.top_k = env_params['top_k']
        self.problem_size = self.top_k
        self.pomo_size = env_params.get('pomo_size', self.top_k)
        self.depot_xy_value = env_params.get('depot_xy', None)
        self.demand_scaler = env_params.get('demand_scaler', None)

        if self.pomo_size != self.problem_size:
            raise ValueError('For POMO CVRP, set pomo_size equal to top_k/problem_size.')

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None

        self.all_coords = None
        self.all_scores = None
        self.depot_node_xy = None
        self.depot_node_demand = None
        self.selected_scores = None
        self.selected_indices = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.at_the_depot = None
        self.load = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def _make_depot(self, batch_size, device):
        if self.depot_xy_value is None:
            # Default depot: center of normalized grid. Change this to your garage location.
            depot_xy = torch.tensor([0.5, 0.5], device=device).reshape(1, 1, 2).expand(batch_size, 1, 2).clone()
        else:
            depot_xy = torch.tensor(self.depot_xy_value, dtype=torch.float, device=device).reshape(1, 1, 2)
            depot_xy = depot_xy.expand(batch_size, 1, 2).clone()
        return depot_xy

    def load_problems(self, batch_size, aug_factor=1, coords=None, scores=None, depot_xy=None, demands=None):
        """Load batch problems.

        Args:
            coords: optional (batch, total_grid_size, 2). If None, synthetic grids are generated.
            scores: optional (batch, total_grid_size). Used for top-k selection.
            depot_xy: optional (batch, 1, 2). If None, env default depot is used.
            demands: optional full-grid workload (batch, total_grid_size). If provided, top-k
                demands are gathered from this instead of being derived from selected_scores.
        """
        self.batch_size = batch_size

        device = coords.device if coords is not None else torch.empty(0).device

        if coords is None or scores is None:
            coords, scores = get_random_scored_grids(batch_size, self.total_grid_size, device=device)
        else:
            if coords.dim() != 3 or scores.dim() != 2:
                raise ValueError('coords must be (batch, total_grid_size, 2), scores must be (batch, total_grid_size).')
            if coords.size(0) != batch_size or scores.size(0) != batch_size:
                raise ValueError('coords/scores batch dimension must match batch_size.')
            if coords.size(1) < self.top_k:
                raise ValueError('Number of candidate cells must be >= top_k.')

        self.all_coords = coords
        self.all_scores = scores
        node_xy, selected_scores, selected_indices = select_topk_cells(coords, scores, self.top_k)

        if demands is None:
            node_demand = make_demand_from_scores(selected_scores, self.demand_scaler)
        else:
            selected_demand_idx = selected_indices
            node_demand = demands.gather(dim=1, index=selected_demand_idx)
            node_demand = node_demand.clamp(min=1e-6, max=1.0)

        if depot_xy is None:
            depot_xy = self._make_depot(batch_size, coords.device)
        else:
            if depot_xy.dim() != 3 or depot_xy.size(1) != 1 or depot_xy.size(2) != 2:
                raise ValueError('depot_xy must be (batch, 1, 2).')

        if aug_factor > 1:
            if aug_factor != 8:
                raise NotImplementedError('Only 8-fold augmentation is implemented.')
            self.batch_size *= 8
            depot_xy = augment_xy_data_by_8_fold(depot_xy)
            node_xy = augment_xy_data_by_8_fold(node_xy)
            node_demand = node_demand.repeat(8, 1)
            selected_scores = selected_scores.repeat(8, 1)
            selected_indices = selected_indices.repeat(8, 1)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(self.batch_size, 1), device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.selected_scores = selected_scores
        self.selected_indices = selected_indices

        self.BATCH_IDX = torch.arange(self.batch_size, device=node_xy.device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=node_xy.device)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.selected_scores = selected_scores
        self.reset_state.selected_indices = selected_indices

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.depot_node_xy.device)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.depot_node_xy.device)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size), device=self.depot_node_xy.device)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=self.depot_node_xy.device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=self.depot_node_xy.device)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.depot_node_xy.device)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        return self.step_state, None, False

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)

        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        selected_demand = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(dim=2)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1.0

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load[:, :, None] + 1e-5 < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(dim=3).sqrt()
        return segment_lengths.sum(dim=2)
