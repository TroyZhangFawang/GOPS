import argparse
import os
import numpy as np

from gops.create_pkg.create_alg import create_alg, register as register_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.create_pkg.create_apprfunc import register as register_func
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # 1. 环境设置
    parser.add_argument("--env_id", type=str, default="pyth_veh3dofconti_bimodaldiffusion_planning")
    parser.add_argument("--control_mode", type=str, default="planning")
    # 【关键】算法名改为自动注册生成的 CamelCase 名字
    parser.add_argument("--algorithm", type=str, default="DSACTDiffusion2")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=12345, help="Seed")

    # 2. 环境参数
    parser.add_argument("--action_type", type=str, default="continu")
    parser.add_argument("--is_render", type=bool, default=False)
    parser.add_argument("--pred_horizon", type=int, default=20)

    ################################################
    # 3. Parameters for approximate function (Networks)

    # Value Network (Critic) - 使用普通 MLP
    parser.add_argument("--value_func_name", type=str, default="DSACTCriticEncodingNet")
    parser.add_argument("--value_func_type", type=str, default="MLP")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--value_hidden_activation", type=str, default="relu")
    parser.add_argument("--value_output_activation", type=str, default="linear")

    parser.add_argument("--policy_func_name", type=str, default="DiffusionEncondingNet")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="relu")
    parser.add_argument("--policy_output_activation", type=str, default="linear")
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )

    # 扩散步数
    parser.add_argument("--diffusion_steps", type=int, default=20)  # 训练用小一点，推理更慢

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=100000)
    parser.add_argument("--ini_network_dir", type=str, default=None)
    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=10000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=2*500000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=64)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)


    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Parameters for algorithm
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--value_gamma", type=float, default=0.99)
    parser.add_argument("--delay_update", type=int, default=2, help="delay update steps")

    # 学习率
    parser.add_argument("--q_learning_rate", type=float, default=3e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--reward_scale", type=float, default=1.0)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)
    parser.add_argument("--log_save_interval", type=int, default=1000)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())

    # 创建环境
    env = create_env(**args)

    # 初始化参数 (GOPS 自动计算 obs_dim, act_dim 并注入 args)
    args = init_args(env, **args)

    # Start TensorBoard
    # start_tensorboard(args["save_folder"])

    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)

    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)

    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)

    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)

    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()

    print("Training Finished!")