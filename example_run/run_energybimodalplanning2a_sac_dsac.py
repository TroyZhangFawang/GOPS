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

runner = PolicyRunner(
    log_policy_dir_list=["../results/pyth_energybimodalplanning2a0529/DSACT_250602-193643",
                         # "../results/pyth_energybimodalplanning2a0529/DSACT_250602-193643",
                            "../results/pyth_energybimodalplanning2a0529/SAC_250602-193321"],
    trained_policy_iteration_list=["1950000",
                                   "3150000"],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "ref_num": 14, "u_num": 0, "ref_time": 0},  #
    save_render=True,
    legend_list=["EDSACT", "SAC"],
    use_opt=False,  # Use optimal solution for comparison
    dt=0.05, # time interval between steps
)

runner.run()
