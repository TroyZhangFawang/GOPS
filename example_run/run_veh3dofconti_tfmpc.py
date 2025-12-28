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


from gops.sys_simulator.sys_run import PolicyRunner
import numpy as np
result_path = "../results/pyth_veh3dofconti/"
runner = PolicyRunner(
    log_policy_dir_list=[result_path+"TFMPC128-42025-08-31-21-53-08",
                         result_path+"TFMPC256-22025-08-31-23-46-03",
                         result_path+"TFMPC256-42025-08-31-23-46-14",
                         result_path+"TFMPC256-82025-08-31-23-46-24",
                         result_path+"TFMPC512-4-3e-52025-08-31-15-12-42",],
    trained_policy_iteration_list=["46000_opt", "46000_opt", "46000_opt", "46000_opt", "46000_opt"],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0.0, 0.0, 0, 0], "ref_time":0.0, "ref_num": 0}, #
    save_render=False,
    legend_list=["128-4", "256-2", "256-4", "256-8", "512-4"],
    use_opt=False, # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 30,
        "gamma": 0.99,
        "mode": "shooting",
        "minimize_options": {
            "max_iter": 50,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
        },
        "use_terminal_cost": False,
    },
    constrained_env=False,
    is_tracking=True,
    dt=0.1,
)

runner.run()
