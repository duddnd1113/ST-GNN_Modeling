import logging
from TSPTopKTester import TSPTopKTester as Tester

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

env_params = {
    'total_grid_size': 18000,
    'top_k': 50,
    'pomo_size': 50,
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

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'enable': True,
        'path': './checkpoint-510.pt',
    },
    'test_episodes': 100 * 1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,
}


def main():
    if DEBUG_MODE:
        tester_params['test_episodes'] = 100
        env_params['total_grid_size'] = 200
        env_params['top_k'] = 20
        env_params['pomo_size'] = 20

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    tester = Tester(env_params, model_params, tester_params)
    tester.run()


if __name__ == '__main__':
    main()
