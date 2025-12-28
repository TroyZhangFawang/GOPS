# 文件名: train_diffusion_offserial.py

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
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
from gops.algorithm import dsact_diffusion
from gops.apprfunc import apprfunc_diffusion

# 注册算法
register_alg("DSACT_Diffusion", dsact_diffusion.DSACT_Diffusion, dsact_diffusion.ApproxContainer)

# 注册网络: 名字 "mlp_diffusion" 对应 DiffusionMLP 类
register_func("mlp", "diffusion", apprfunc_diffusion.DiffusionMLP)

print("Custom Algorithm and Network Registered Successfully!")

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users

    # 1. 环境设置 (使用我们改好的 Bimodal Diffusion 环境)
    # 确保 pyth_veh3dofconti_bimodaldiffusion_planning.py 在同级目录或 PYTHONPATH 中
    parser.add_argument("--env_id", type=str, default="pyth_veh3dofconti_bimodaldiffusion_planning")
    parser.add_argument("--algorithm", type=str, default="DSACT_Diffusion")  # 使用新注册的算法名
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=3328005365, help="Seed")

    ################################################
    # 2. 1 Parameters for environment
    parser.add_argument("--action_type", type=str, default="continu")
    parser.add_argument("--is_render", type=bool, default=False)
    # 关键：传入 pred_horizon 给环境
    parser.add_argument("--pred_horizon", type=int, default=20)

    ################################################
    # 3. Parameters for approximate function (Networks)

    # Value Network (Critic) - 使用普通 MLP
    parser.add_argument("--q_func_name", type=str, default="mlp")  # mlp_mlp
    parser.add_argument("--q_func_type", type=str, default="mlp")
    parser.add_argument("--q_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--q_hidden_activation", type=str, default="relu")
    parser.add_argument("--q_output_activation", type=str, default="linear")

    # Policy Network (Actor) - 使用新注册的 mlp_diffusion
    parser.add_argument("--policy_func_name", type=str, default="mlp")
    parser.add_argument("--policy_func_type", type=str, default="diffusion")  # 对应注册时的 name
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="relu")
    parser.add_argument("--policy_output_activation", type=str, default="linear")

    # 扩散步数
    parser.add_argument("--diffusion_steps", type=int, default=20)  # 训练用小一点，推理更慢

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=100000)
    parser.add_argument("--ini_network_dir", type=str, default=None)
    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    parser.add_argument("--buffer_human_name", type=str, default="prioritized_replay_buffer")
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--replay_batch_size", type=int, default=64)
    parser.add_argument("--sampler_sync_interval", type=int, default=1)
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="eval_sampler")
    parser.add_argument("--num_eval_episode", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=500)
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

    # 打印一下维度确认
    print(f"Env Obs Dim: {args['obs_dim']}, Act Dim: {args['act_dim']}")

    # Start TensorBoard
    start_tensorboard(args["save_folder"])

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