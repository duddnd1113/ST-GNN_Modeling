import torch
import pandas as pd
from logging import getLogger

from CVRPTopKEnvironment import CVRPTopKEnv as Env
from CVRPTopKModel import CVRPTopKModel as Model


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CVRPTopKTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.logger = getLogger(name='trainer')

        use_cuda = tester_params['use_cuda'] and torch.cuda.is_available()
        if use_cuda:
            cuda_device_num = tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.env = Env(**env_params)
        self.model = Model(**model_params)
        self.last_reward = None

        if tester_params.get('model_load', {}).get('enable', False):
            checkpoint = torch.load(tester_params['model_load']['path'], map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info('Loaded model: {}'.format(tester_params['model_load']['path']))

    def run(self):
        score_meter = AverageMeter()
        aug_score_meter = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:
            batch_size = min(self.tester_params['test_batch_size'], test_num_episode - episode)
            score, aug_score = self._test_one_batch(batch_size)
            score_meter.update(score, batch_size)
            aug_score_meter.update(aug_score, batch_size)
            episode += batch_size
            self.logger.info('episode {}/{}, score={:.4f}, aug_score={:.4f}'.format(
                episode, test_num_episode, score, aug_score
            ))

        self.logger.info('NO-AUG SCORE: {:.4f}'.format(score_meter.avg))
        self.logger.info('AUGMENTATION SCORE: {:.4f}'.format(aug_score_meter.avg))

    def _test_one_batch(self, batch_size, coords=None, scores=None, depot_xy=None, demands=None, augmentation_enable=None):
        if augmentation_enable is None:
            augmentation_enable = self.tester_params.get('augmentation_enable', False)
        aug_factor = self.tester_params.get('aug_factor', 8) if augmentation_enable else 1

        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor, coords=coords, scores=scores, depot_xy=depot_xy, demands=demands)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

        self.last_reward = reward
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float().mean()
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float().mean()
        return no_aug_score.item(), aug_score.item()

    def solve_real_csv(self, csv_path, x_col='x', y_col='y', score_col='score', id_col=None,
                       demand_col=None, depot_xy=None, output_csv_path=None, use_augmentation=True):
        """Solve one real grid snapshot.

        Expected CSV columns: grid_id,x,y,score. Optionally include demand/workload.
        Coordinates are normalized internally. The route output includes depot visits as
        rows with is_depot=True, so it can represent multiple cleaning sub-routes.
        """
        df = pd.read_csv(csv_path)
        if x_col not in df.columns or y_col not in df.columns or score_col not in df.columns:
            raise ValueError('CSV must contain columns: {}, {}, {}'.format(x_col, y_col, score_col))

        ids = df[id_col].tolist() if id_col and id_col in df.columns else list(range(len(df)))

        xy_raw_cpu = torch.tensor(df[[x_col, y_col]].values, dtype=torch.float)
        score_cpu = torch.tensor(df[score_col].values, dtype=torch.float)
        xy_min = xy_raw_cpu.min(dim=0).values
        xy_max = xy_raw_cpu.max(dim=0).values
        xy_cpu = (xy_raw_cpu - xy_min) / (xy_max - xy_min + 1e-12)

        coords = xy_cpu[None, :, :].to(self.device)
        scores = score_cpu[None, :].to(self.device)

        demands = None
        if demand_col is not None and demand_col in df.columns:
            demands = torch.tensor(df[demand_col].values, dtype=torch.float)[None, :].to(self.device)

        depot_tensor = None
        depot_raw = None
        if depot_xy is not None:
            depot_raw = torch.tensor(depot_xy, dtype=torch.float)
            depot_norm = (depot_raw - xy_min) / (xy_max - xy_min + 1e-12)
            depot_tensor = depot_norm.reshape(1, 1, 2).to(self.device)

        no_aug_score, aug_score = self._test_one_batch(
            batch_size=1,
            coords=coords,
            scores=scores,
            depot_xy=depot_tensor,
            demands=demands,
            augmentation_enable=use_augmentation,
        )

        reward = self.last_reward
        flat_best = reward.reshape(-1).argmax().item()
        best_row = flat_best // self.env.pomo_size
        best_pomo = flat_best % self.env.pomo_size

        topk_original_indices = self.env.selected_indices[best_row].detach().cpu().tolist()
        local_route = self.env.selected_node_list[best_row, best_pomo].detach().cpu().tolist()

        rows = []
        for order, local_node in enumerate(local_route):
            if local_node == 0:
                rows.append({
                    'route_order': order,
                    'is_depot': True,
                    'grid_id': 'DEPOT',
                    x_col: depot_xy[0] if depot_xy is not None else None,
                    y_col: depot_xy[1] if depot_xy is not None else None,
                    score_col: None,
                    'local_node': 0,
                })
            else:
                original_idx = topk_original_indices[local_node - 1]
                row = df.iloc[original_idx].to_dict()
                row.update({
                    'route_order': order,
                    'is_depot': False,
                    'local_node': local_node,
                    'original_index': original_idx,
                    'route_original_id': ids[original_idx],
                })
                rows.append(row)

        route_df = pd.DataFrame(rows)
        if output_csv_path is not None:
            route_df.to_csv(output_csv_path, index=False)
            self.logger.info('Saved route CSV: {}'.format(output_csv_path))

        return {
            'no_aug_score': no_aug_score,
            'aug_score': aug_score,
            'top_k_original_indices': topk_original_indices,
            'local_route': local_route,
            'route_df': route_df,
        }
