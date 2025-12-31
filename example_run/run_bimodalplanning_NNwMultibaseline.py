from gops.sys_simulator.sys_run import PlanningRunner
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
result_path = "../results/pyth_veh3dofconti_bimodal_planning/FHADP2Lagrangian_250510-200146"
runner = PlanningRunner(
    log_policy_dir_list=[result_path],
    trained_policy_iteration_list=["620000"],
    is_init_info=True,
    init_info={"init_state": [5, 0.0, 0.0, 0.0, 0.0, 0.0], "ref_num":0, "u_num":0, "ref_time":0},
    base_planner1="MPCPlanner",  # 配合MPCPlanner或者MPCController使用
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
    base_planner2="LatticePlanner", # LatticePlanner/BezierPlanner/MPCPlanner
    controller="SimpleController",  # IDMController/SimpleController/MPCController/IdealController
    legend_list=["MLP"],
    constrained_env=True,
    is_tracking=True,
    save_render=True,
    render_args={
        "evaluation": False,
        "collision_termination": False,
        "map_frequency": 0,  # 几帧更新一次map，0表示仅更新一次
        "snapshot": True,
        "save_video": True,
        "video_root": './',
        "video_name": None
    },
    dt=0.1,
)

runner.run()

