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
        "../results/pyth_semitruck7dof/FHADP2_240305-150727"
        # "PATH_TO_YOUR_RESULT_DIR",
    ],
    trained_policy_iteration_list=[
        "199600_opt"
        # "ITERATION_NUM",
    ],
    is_init_info=True,
    init_info={
        # parameters of env.reset()
        "init_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 100, 96],
    },
    save_render=False,
    legend_list=[
        "FHADP2"
        # "ALGORITHM_NAME",a
    ],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 100,
        "gamma": 1.0,
        "mode": "collocation",
        "minimize_options": {
            "max_iter": 4000,
            "tol": 1e-4,
        },
        "use_terminal_cost": False,
        "use_MPC_for_general_env": True,
    },
    constrained_env=False,
    is_tracking=True,
    dt=0.01,
)

runner.run()
