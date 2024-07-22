#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system


from gops.sys_simulator.sys_run import PolicyRunner


runner = PolicyRunner(
    log_policy_dir_list=[
        "../results/pyth_veh3dofcontiplanning/FHADP2_240530-212447"
        # "PATH_TO_YOUR_RESULT_DIR",
    ],
    trained_policy_iteration_list=[
        "99900"
        # "ITERATION_NUM",
    ],
    is_init_info=True,
    init_info={
        # parameters of env.reset()
        "init_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ref_time": 0.0,
        "ref_num": 0,
    },
    save_render=True,
    legend_list=["FHADP2"],
    use_opt=False,  # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 50,
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
    dt=0.1,
)

runner.run()
