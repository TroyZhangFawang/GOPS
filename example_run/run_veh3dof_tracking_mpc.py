#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from gops.sys_simulator.sys_run import OptRunner


runner = OptRunner(
    log_policy_dir_list=[
        "../results/pyth_veh3dofconti_surrcstr_penalty_planning/"
    ],
    is_init_info=True,
    init_info={
        # parameters of env.reset()
        "init_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ref_time": 0.0,
        "ref_num": 0,
    },
    save_render=True,
    legend_list=[],
    use_opt=True,  # Use optimal solution for comparison
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
    constrained_env=True,
    is_tracking=True,
    dt=0.1,
)

runner.run()
