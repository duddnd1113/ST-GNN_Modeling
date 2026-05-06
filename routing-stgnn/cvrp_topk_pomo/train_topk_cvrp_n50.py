import logging
from CVRPTopKTrainer import CVRPTopKTrainer as Trainer

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

env_params = {
    'total_grid_size': 22500,
    'top_k': 50,
    'pomo_size': 50,
    'depot_xy': [0.5, 0.5],
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** 0.5,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {'lr': 1e-4, 'weight_decay': 1e-6},
    'scheduler': {'milestones': [501], 'gamma': 0.1},
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 510,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'model_save_interval': 10,
}


def main():
    if DEBUG_MODE:
        trainer_params['epochs'] = 2
        trainer_params['train_episodes'] = 128
        trainer_params['train_batch_size'] = 4
        env_params['top_k'] = 20
        env_params['pomo_size'] = 20

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    trainer = Trainer(env_params, model_params, optimizer_params, trainer_params)
    trainer.run()


if __name__ == '__main__':
    main()
