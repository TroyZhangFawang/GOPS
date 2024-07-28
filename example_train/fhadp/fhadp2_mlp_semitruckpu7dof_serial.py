#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for fhadp2 + veh3dofconti + mlp + off_serial using full horizon action
#  Update Date: 2023-07-30, Jiaxin Gao: create example

import os
import argparse
import numpy as np

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args_abpo import init_args
from gops.utils.cost_update_abpo import cost_update
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
from gops.utils.common_utils import seed_everything


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_semitruckpu7dof")
    parser.add_argument("--cost_paras", type=list, default=[1, 0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.4, 2.0])#[1.0827611 , 1.00093552, 0.80229881, 0.57172105, 0.50084567, 0.50005888, 0.5023708 , 0.35336234, 1.95336234]
    parser.add_argument("--algorithm", type=str, default="FHADP2")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False)
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--is_render", type=bool, default=False)
    parser.add_argument("--is_adversary", type=bool, default=False)

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument("--value_func_type", type=str, default="MLP")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="FiniteHorizonFullPolicy"
    )
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="elu")
    
    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--policy_learning_rate", type=float, default=3e-5)
    parser.add_argument("--cost_learning_rate", type=float, default=1e-5)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=50000)
    parser.add_argument("--max_iteration_upper", type=int, default=20)  # iteration of outer loop (Theta update)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    # 4.1. Parameters for off_serial_trainer
    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=100000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=100)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=256)
    # Add noise to action for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={"mean": np.array([0], dtype=np.float32), "std": np.array([0.2], dtype=np.float32),},
    )

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--save_folder_upper", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=2000)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    # start_tensorboard(args["save_folder_upper"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    # buffer = create_buffer(**args)
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # initial cost_update class
    cost_update = cost_update(trainer.alg.envmodel, **args)

    # Start training ... ...
    cost_paras = np.array(args["cost_paras"])
    data = trainer.buffer.sample_batch(args["replay_batch_size"])
    for iter_up in range(args["max_iteration_upper"]):
        traj = trainer.alg.envmodel.Rollout_traj_NN(data, trainer.networks, args["pre_horizon"], cost_paras)

        args["save_folder"] = args["save_folder_upper"]+'/' + str(iter_up) + 'th-lower'
        os.makedirs(args["save_folder"], exist_ok=True)
        os.makedirs(args["save_folder"] + "/apprfunc", exist_ok=True)
        os.makedirs(args["save_folder"] + "/evaluator", exist_ok=True)
        trainer.save_folder = args["save_folder"]
        # trainer.evaluator.save_folder=args["save_folder"]

        trainer.train()
        trainer.iteration = 0
        cost_paras = cost_update.step(iter_up, traj, cost_paras)
        trainer.alg.envmodel.update_cost_paras(cost_paras)
        print("Training is finished!")
        ################################################
        # Plot and save training figures
        plot_all(args["save_folder"])
        save_tb_to_csv(args["save_folder"])
        print(iter_up, "-st/th update are finished!")
