import torch
from logging import getLogger

from TSPTopKEnv import TSPTopKEnv as Env
from TSPTopKModel import TSPTopKModel as Model
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TSPTopKTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.logger = getLogger(name='trainer')

        use_cuda = trainer_params['use_cuda'] and torch.cuda.is_available()
        if use_cuda:
            cuda_device_num = trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model = Model(**model_params)
        self.env = Env(**env_params)
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])

    def run(self):
        for epoch in range(1, self.trainer_params['epochs'] + 1):
            self.scheduler.step()
            train_score, train_loss = self._train_one_epoch(epoch)
            self.logger.info(f'Epoch {epoch:3d}: score={train_score:.4f}, loss={train_loss:.4f}')

            if epoch % self.trainer_params['model_save_interval'] == 0 or epoch == self.trainer_params['epochs']:
                save_path = f"checkpoint-{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, save_path)
                self.logger.info(f'Saved {save_path}')

    def _train_one_epoch(self, epoch):
        score_meter = AverageMeter()
        loss_meter = AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0

        while episode < train_num_episode:
            batch_size = min(self.trainer_params['train_batch_size'], train_num_episode - episode)
            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_meter.update(avg_score, batch_size)
            loss_meter.update(avg_loss, batch_size)
            episode += batch_size

        return score_meter.avg, loss_meter.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # POMO shared baseline: average reward across pomo trajectories for each instance.
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return score_mean.item(), loss_mean.item()
