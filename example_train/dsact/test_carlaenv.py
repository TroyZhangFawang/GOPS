# -*- coding: utf-8 -*-

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dsac + humanoidconti + mlp + offserial
#  Update Date: 2021-03-05, Wenxuan Wang: create example
import os
import argparse
import json

import numpy as np
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv



if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_offroadcarla", help="id of environment")
    # parser.add_argument("--task", type=str, default="control", help="planning/control")
    parser.add_argument("--vdes", type=float, default=10, help="target speed")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    ################################################

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**{**args, "vector_env_num": None})
    obs, _ = env.reset()
    for i in range(10000):
        action = np.random.rand(2,)*0.5
        print("iteration:", i)
        obs,_, _, _ = env.step(action)
        # print(obs[0], action)
