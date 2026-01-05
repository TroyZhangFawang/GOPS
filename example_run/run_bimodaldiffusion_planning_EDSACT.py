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
#  Update: 2024-xx-xx, Added video/gif saving functionality

import os
# 尝试导入必要的模块
try:
    from gops.sys_simulator.sys_run import EnhancedPolicyRunner
    from gops.utils.common_utils import get_reference_from_info, get_robot_state_from_info
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

    # 定义简化版本
    def get_reference_from_info(info):
        return []

    def get_robot_state_from_info(info):
        return []




# 主执行代码
if __name__ == "__main__":
    result_path = "../results/pyth_veh3dofconti_bimodaldiffusion_planning/DSACT_260104-105706"

    # 确保路径存在
    if not os.path.exists(result_path):
        print(f"Error: Result path does not exist: {result_path}")
        # 尝试相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        result_path = os.path.join(current_dir, result_path)
        print(f"Trying: {result_path}")

    print(f"Using result path: {result_path}")

    # 使用增强的runner
    runner = EnhancedPolicyRunner(
        log_policy_dir_list=[result_path],
        trained_policy_iteration_list=["450000"],
        is_init_info=True,
        init_info={"init_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "ref_num": 0, "u_num": 0, "ref_time": 0},
        save_render=True,
        legend_list=[],
        use_opt=False,
        opt_args={
            "opt_controller_type": "MPC",
            "num_pred_step": 20,
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

        # 增强功能
        save_screenshots=True,  # 保存截图
        screenshot_interval=5,  # 每5步保存一次
        convert_to_gif=True,  # 转换为GIF
        gif_fps=10,  # GIF的帧率
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()