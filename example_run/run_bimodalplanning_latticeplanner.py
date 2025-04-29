from gops.sys_simulator.sys_run import PlanningRunner
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
result_path = "../results/pyth_veh3dofconti_bimodal_planning/"
runner = PlanningRunner(
    log_policy_dir_list=[result_path],
    env_id="pyth_veh3dofconti_bimodal_planning",
    planner="LatticePlanner", # LatticePlanner/BezierPlanner/MPCPlanner
    is_init_info=True,
    init_info={"init_state": [5, 0.0, 0.0, 0.0, 0.0, 0.0], "ref_num":0, "u_num":0, "ref_time":0}, #
    save_render=True,
    legend_list=[],
    controller="SimpleController",  # IDMController/SimpleController/MPCController/IdealController
    use_opt=True,
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

