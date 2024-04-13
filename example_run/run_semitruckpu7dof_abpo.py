#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.sys_simulator.sys_run import PolicyRunner_Multiopt
import numpy as np
init_path = "FHADP2_240408-182414-upper_50-inner_50000/0th-lower"
abpo_path = "FHADP2_240408-182414-upper_50-inner_50000/0th-lower"
result_path = "../results/pyth_semitruckpu7dof/"
runner = PolicyRunner_Multiopt(
    log_policy_dir_list=[result_path+init_path, result_path+abpo_path],
    trained_policy_iteration_list=["50000", "50000"],
    is_init_info=True,
    init_info={"init_state": [0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, -1, -1, 10, 6]},
    save_render=False,
    legend_list=["FHADP2", "ABPO"],

    use_opt=True, # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 100,
        "gamma": 1,
        "mode": "shooting",
        "minimize_options": {
            "max_iter": 10,
            "tol": 1e-5,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
        },
        "use_terminal_cost": False,
    },
    multi_opt=True,
    multi_opt_args={"opt_run_times":2,
        "cost_paras_list":[[1, 0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.4, 2.0],
                           [1.29976688, 0.53561693, 0.79591853, 0.48577294, 0.4991067,
                            0.49982506, 0.49884216, 0.38836425, 1.98836425]],
                    },
    constrained_env=False,
    is_tracking=True,
    dt=0.01,
)

runner.run()
