import gops.utils.Auxiliary_System as AuxiSys
from casadi import *
import torch
from torch.utils.tensorboard import SummaryWriter


class cost_update_pdp:
    def __init__(self, env, **kwargs):
        self.maxiters_u = kwargs.get("max_iteration_upper")
        self.batch_size = kwargs.get("replay_batch_size")
        self.save_folder_u = kwargs["save_folder_upper"]
        self.horizon = kwargs["pre_horizon"]
        self.pdp_lr = kwargs["cost_learning_rate"]
        self.env = env
        # initialize the cost_auxvar and path cost
        self.env.vehicle_dynamics.auxvar_setting()
        # define auxvar system
        self.cost_auxvar_oc = AuxiSys.OCSys()
        self.dyn = env.vehicle_dynamics.dyn
        self.cost_auxvar_oc.setAuxvarVariable(self.env.vehicle_dynamics.cost_auxvar)
        self.cost_auxvar_oc.setControlVariable(self.env.vehicle_dynamics.U)
        self.cost_auxvar_oc.setStateVariable(self.env.vehicle_dynamics.X)
        self.cost_auxvar_oc.setDyn(self.env.vehicle_dynamics.dyn)
        self.cost_auxvar_oc.setPathCost(self.env.vehicle_dynamics.path_cost)
        self.cost_auxvar_oc.setFinalCost(self.env.vehicle_dynamics.final_cost)
        self.aux_solver = AuxiSys.LQR()  # use it to calc d\xi / d\theta
        self.writer_up = SummaryWriter(log_dir=self.save_folder_u, flush_secs=60)
        self.tb_info = dict()

    def step(self, iter_up, traj, cost_paras):
        self.writer_up.add_scalar(tb_tags["Q_y1_tt"], cost_paras[0], iter_up)
        self.writer_up.add_scalar(tb_tags["Q_u1_tt"], cost_paras[1], iter_up)
        self.writer_up.add_scalar(tb_tags["Q_psi1_tt"], cost_paras[2], iter_up)
        self.writer_up.add_scalar(tb_tags["Q_psi1dot_tt"], cost_paras[3], iter_up)
        self.writer_up.add_scalar(tb_tags["Q_beta1_tt"], cost_paras[4], iter_up)
        self.writer_up.add_scalar(tb_tags["Q_phi1_tt"], cost_paras[5], iter_up)
        self.writer_up.add_scalar(tb_tags["Q_phi1dot_tt"], cost_paras[6], iter_up)
        self.writer_up.add_scalar(tb_tags["R_u_tt"], cost_paras[7], iter_up)
        self.writer_up.add_scalar(tb_tags["R_udot_tt"], cost_paras[8], iter_up)
        ref_state_traj = np.zeros((self.horizon+1, 3))
        dp = torch.zeros(cost_paras.shape)


        aux_sys = self.cost_auxvar_oc.getAuxSys(state_traj_opt=traj['state_traj_opt'][:, :self.env.state_dim],
                                            control_traj_opt=traj['control_traj_opt'][:, :],
                                            costate_traj_opt=traj['costate_traj_opt'][:, :],
                                            auxvar_value=cost_paras)
        self.aux_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
        self.aux_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                    Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
        self.aux_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])

        aux_sol = self.aux_solver.lqrSolver(numpy.zeros((self.cost_auxvar_oc.n_state, self.cost_auxvar_oc.n_auxvar)),
                                            self.horizon)

        # take solution of the auxiliary control system
        dxdp_traj = aux_sol['state_traj_opt']  # dx /dtheta
        dudp_traj = aux_sol['control_traj_opt']  # du/dtheta

        # evaluate the loss
        state_traj = torch.from_numpy(traj['state_traj_opt'])
        state_traj.requires_grad_(True)
        control_traj = torch.from_numpy(traj['control_traj_opt'])
        control_traj.requires_grad_(True)

        dldX_traj = np.zeros((self.horizon, 1, self.env.state_dim))
        x_1, x_2 = MX.sym('x_1', (1, self.env.state_dim)), MX.sym('x_2', (1, 3))
        # (y2-y2_ref)**2 +phi2**2+ phi2dot**2+ (psi2-psi2_ref)**2 + psi2dot**2
        dloss = jacobian(sum1(sum2((x_1[5] - x_2[1]) ** 2+(x_1[6]-x_2[2])**2+x_1[12]**2+x_1[13]**2+x_1[14]**2+x_1[15]**2)), x_1)
        dloss_fn = casadi.Function('dfx', [x_1, x_2], [dloss])
        for i_step in range(self.horizon):
            # ref_state_tensor = self.env.ref_traj.find_nearst_point(torch.tensor([traj['state_traj_opt'][i_step, 14], traj['state_traj_opt'][i_step, 12]]).reshape(1,-1))
            # ref_state = np.array(ref_state_tensor)
            # ref_state_traj[i_step, :] = np.array(ref_state)
            next_t2 = traj['ref_time2_rollout'][i_step]
            path_num = traj['path_num_rollout'][i_step]
            u_num = traj['u_num_rollout'][i_step]
            ref_state = np.array(
                [self.env.ref_traj.compute_x(
                    next_t2, path_num, u_num
                ),
                    self.env.ref_traj.compute_y(
                        next_t2, path_num, u_num
                    ),
                    self.env.ref_traj.compute_phi(
                        next_t2, path_num, u_num
                    )])
            ref_state_traj[i_step, :] = ref_state
            dldX = dloss_fn(traj['state_traj_opt'][i_step, 0, :], ref_state)
            dldX_traj[i_step, :, :] = np.array(dldX)
        # chain rule
        dldX_traj = torch.from_numpy(dldX_traj)
        for t in range(self.horizon):
            dp = dp + torch.mm(dldX_traj[t, :, :], torch.from_numpy(dxdp_traj[t]))
        dp = dp + torch.mm(dldX_traj[-1, :, :], torch.from_numpy(dxdp_traj[-1]))

        upper_loss = self.loss_upper_evaluator_2d(traj['state_traj_opt'][0, :, :],
                                                            ref_state_traj[:, :])
        print('iter:', iter_up, 'loss:', upper_loss, 'paras:', cost_paras, 'dp:', dp)

        cost_paras = cost_paras - self.pdp_lr * dp  # update theta
        # save
        save_data = {'iter': iter_up,
                     'loss_trace': upper_loss,
                     'parameter_trace': cost_paras,
                     'learning_rate': self.pdp_lr}
        self.tb_info[tb_tags["Loss_upper"]] = upper_loss
        self.writer_up.add_scalar(tb_tags["Loss_upper"], upper_loss, iter_up)
        self.writer_up.flush()
        # Save Q and R
        with open(self.save_folder_u + '/results.txt', 'a') as file_handle:
            file_handle.write("-------" + str(iter_up) + " th upper iters-------" + '\n')
            file_handle.write(str({'results': save_data}) + '\n')
        return cost_paras

    def loss_upper_evaluator_3d(self, state, state_up):
        L = (((state[:, :,  5] - state_up[:, :, 1]) ** 2) +
             (state[:, :, 6] - state_up[:, :, 2]) ** 2 +
             state[:, :, 11] ** 2 +
             state[:, :, 12] ** 2 +
             state[:, :, 13] ** 2 +
             state[:, :, 14] ** 2).mean()
        return L

    def loss_upper_evaluator_2d(self, state, state_up):
        L = (((state[:,  5] - state_up[:, 1]) ** 2) +
             (state[:, 6] - state_up[:, 2]) ** 2 +
             state[:, 11] ** 2 +
             state[:, 12] ** 2 +
             state[:, 13] ** 2 +
             state[:, 14] ** 2).mean()
        return L

tb_tags = {
    "TAR of RL iteration": "Evaluation/1. TAR-RL iter",
    "TAR of total time": "Evaluation/2. TAR-Total time [s]",
    "Buffer RAM of RL iteration": "RAM/RAM [MB]-RL iter",
    "alg_time": "Time/Algorithm time [ms]-RL iter",
    "critic_avg_value": "Train/Critic avg value-RL iter",
    "Loss_lower": "Loss_lower/Lower evaluation loss(evaluate steps simu)-lower iteration (Policy iteration)",
    "Loss_upper": "Loss_upper/Upper evaluation loss(evaluate steps simu)-upper iteration",
    # car_tracking env
    "Q_y_ct": "Weight/Weight on delta y",
    "Q_u_ct": "Weight/Weight on delta u",
    "Q_v_ct": "Weight/Weight on v",
    "Q_phi_ct": "Weight/Weight on delta phi",
    "Q_omega_ct": "Weight/Weight on omega",
    "R_str_ct": "Weight/Weight on str action",
    "R_a_ct": "Weight/Weight on acceleration action",
    # scraper env
    "Q_y_sp": "Weight/Weight on delta y",
    "Q_u_sp": "Weight/Weight on delta u",
    "Q_phi_sp": "Weight/Weight on delta phi",
    "Q_theta_sp": "Weight/Weight on theta-joint angle",
    "R_a_sp": "Weight/Weight on acceleration action",
    "R_omega_sp": "Weight/Weight on omega action",
    # tractor trailer env
    "Q_y1_tt": "Weight/Weight on delta y_tractor_tractor",
    "Q_u1_tt": "Weight/Weight on delta u",
    "Q_v1_tt": "Weight/Weight on lat vel",
    "Q_psi1_tt": "Weight/Weight on delta heading angle_tractor",
    "Q_psi1dot_tt": "Weight/Weight on yawrate_tractor",
    "Q_beta1_tt": "Weight/Weight on side slip angle_tractor",
    "Q_phi1_tt": "Weight/Weight on rollangle_tractor",
    "Q_phi1dot_tt": "Weight/Weight on roll rate_tractor",
    "R_accel_tt": "Weight/Weight on acceleration action",
    "R_u_tt": "Weight/Weight on steering angle action",
    "R_udot_tt": "Weight/Weight on steering angle action increment",
}