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

runner = PolicyRunner(
    log_policy_dir_list=["/home/bit/Troy.Z/1_code/GOPS/results/pyth_veh3dofconti/FHADP_240315-142233"],
    trained_policy_iteration_list=["87000_opt"],
    is_init_info=False,
    init_info={"init_state": [0.0, 0.0, 0.0, 0.0, 0, 0]}, # ref_num = [0, 1, 2,..., 7]
    save_render=False,
    legend_list=["FHADP2"],
    use_opt=True, # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 100,
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
