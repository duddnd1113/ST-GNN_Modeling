from dataclasses import dataclass
import torch

from TSProblemDef import select_topk_cells, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # selected top-k coordinates, shape: (batch, top_k, 2)
    selected_scores: torch.Tensor
    # selected top-k scores, shape: (batch, top_k)
    selected_indices: torch.Tensor
    # original grid/candidate indices, shape: (batch, top_k)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class TSPTopKEnv:
    """POMO-style TSP environment for top-k grid routing.

    Training mode:
        If coords/scores are not provided, generate random TSP instances with exactly top_k nodes.
        This avoids repeatedly generating/selecting from 18,000 fake cells during training.

    Real-grid inference mode:
        If coords/scores are provided, select top_k cells once from the full grid snapshot,
        then solve TSP only over those selected cells.
    """

    def __init__(self, **env_params):
        self.env_params = env_params
        self.total_grid_size = env_params.get('total_grid_size', 22500)
        self.top_k = env_params['top_k']
        self.problem_size = self.top_k
        self.pomo_size = env_params.get('pomo_size', self.top_k)

        if self.pomo_size != self.problem_size:
            raise ValueError('For TSP POMO, set pomo_size equal to top_k/problem_size.')

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None

        self.all_coords = None
        self.all_scores = None
        self.problems = None
        self.selected_scores = None
        self.selected_indices = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

    def load_problems(self, batch_size, aug_factor=1, coords=None, scores=None):
        """Load one batch of problems.

        Args:
            batch_size: number of problem instances.
            aug_factor: 1 or 8.
            coords: optional tensor (batch, total_grid_size, 2). If provided with scores,
                top-k selection is performed once before routing.
            scores: optional tensor (batch, total_grid_size).
        """
        self.batch_size = batch_size

        if coords is None or scores is None:
            # Training/debug mode: train directly on random TSP(top_k) instances.
            # No 18,000-grid top-k filtering is repeated here.
            self.problems = torch.rand(batch_size, self.problem_size, 2)
            self.selected_scores = torch.ones(batch_size, self.problem_size)
            self.selected_indices = torch.arange(self.problem_size)[None, :].expand(batch_size, self.problem_size)
            self.all_coords = self.problems
            self.all_scores = self.selected_scores
        else:
            # Real-grid mode: select top-k once from the provided grid snapshot.
            if coords.dim() != 3 or scores.dim() != 2:
                raise ValueError('coords must be (batch, total_grid_size, 2), scores must be (batch, total_grid_size).')
            if coords.size(0) != batch_size or scores.size(0) != batch_size:
                raise ValueError('coords/scores batch dimension must match batch_size.')
            if coords.size(1) < self.top_k:
                raise ValueError('Number of candidate cells must be >= top_k.')

            self.all_coords = coords
            self.all_scores = scores
            self.problems, self.selected_scores, self.selected_indices = select_topk_cells(
                coords, scores, self.top_k
            )

        if aug_factor > 1:
            if aug_factor != 8:
                raise NotImplementedError('Only 8-fold augmentation is implemented.')
            self.batch_size = self.batch_size * 8
            self.problems = augment_xy_data_by_8_fold(self.problems)
            self.selected_scores = self.selected_scores.repeat(8, 1)
            self.selected_indices = self.selected_indices.repeat(8, 1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))

        reward = None
        done = False
        return Reset_State(self.problems, self.selected_scores, self.selected_indices), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)

        self.step_state.current_node = selected
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        done = (self.selected_count == self.problem_size)
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(
            self.batch_size, self.pomo_size, self.problem_size, 2
        )
        seq_expanded = self.problems[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.problem_size, 2
        )
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(dim=3).sqrt()
        return segment_lengths.sum(dim=2)
