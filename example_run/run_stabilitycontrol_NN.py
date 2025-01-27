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
result_path = "../results/pyth_stabilitycontrol/TRANSStolenMpc_241230-205021/"
runner = PolicyRunner(
    log_policy_dir_list=[result_path],
    trained_policy_iteration_list=["100000"],
    is_init_info=True,
    init_info={"init_state": [0, 0.2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "ref_time":0.0, "ref_num": 2, 'u_num':0, 'slope_num':1}, #
    save_render=False,
    legend_list=["FHADP2"],
    use_opt=False,  # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 30,
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
    constrained_env=False,
    is_tracking=True,
    dt=0.01,
)

runner.run()
