from typing import Optional, Sequence

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.semitruck7dof import angle_normalize, Semitruck7DoFParam


class Semitrucks7DoFModel(RobotModel):
    dt: Optional[float] = 0.01
    robot_state_dim: int = 15

    def __init__(
        self, 
        robot_state_lower_bound: Optional[Sequence] = None, 
        robot_state_upper_bound: Optional[Sequence] = None, 
    ):
        super().__init__(
            robot_state_lower_bound=robot_state_lower_bound,
            robot_state_upper_bound=robot_state_upper_bound,
        )
        self.param = Semitruck7DoFParam()
        
    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_size = len(state[:, 0])
        state_next = torch.zeros_like(state)
        M_matrix = torch.zeros((self.param.state_dim - 2, self.param.state_dim - 2))
        M_matrix[0, 0] = self.param.M11
        M_matrix[0, 1] = self.param.M12
        M_matrix[0, 2] = self.param.M13

        M_matrix[1, 0], M_matrix[1, 1], M_matrix[1, 3] = self.param.M21, self.param.M22, self.param.M24

        M_matrix[2, 0], M_matrix[2, 3], M_matrix[2, 4], M_matrix[2, 7] = \
            self.param.M31, self.param.M34, self.param.M35, self.param.M38

        M_matrix[3, 4], M_matrix[3, 5], M_matrix[3, 7], = self.param.M45, self.param.M46, self.param.M48

        M_matrix[4, 4], M_matrix[4, 5], M_matrix[4, 7] = self.param.M55, self.param.M56, self.param.M58

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 3], M_matrix[5, 4], M_matrix[5, 5], M_matrix[5, 7] = \
            self.param.M61, self.param.M62, self.param.M64, self.param.M65, self.param.M66, self.param.M68

        M_matrix[6, 2] = self.param.M73
        M_matrix[7, 6] = self.param.M87
        M_matrix[8, 8] = self.param.M99
        M_matrix[9, 9] = self.param.M1010
        M_matrix[10, 0], M_matrix[10, 10] = self.param.M111, self.param.M1111
        M_matrix[11, 11] = self.param.M1212
        M_matrix[12, 12] = self.param.M1313

        A_matrix = torch.zeros((self.param.state_dim - 2, self.param.state_dim - 2))
        A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 6] = \
            self.param.A11, self.param.A12, self.param.A21, self.param.A22, self.param.A23, self.param.A24, self.param.A27
        A_matrix[2, 0], A_matrix[2, 1], A_matrix[2, 4], A_matrix[2, 5] \
            = self.param.A31, self.param.A32, self.param.A35, self.param.A36

        A_matrix[3, 4], A_matrix[3, 5] = self.param.A45, self.param.A46
        A_matrix[4, 2], A_matrix[4, 4], A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7] = \
            self.param.A53, self.param.A55, self.param.A56, self.param.A57, self.param.A58
        A_matrix[5, 1], A_matrix[5, 5], A_matrix[6, 3], A_matrix[7, 7], \
        A_matrix[8, 1], A_matrix[9, 5], A_matrix[11, 0], A_matrix[11, 8], \
        A_matrix[12, 4], A_matrix[12, 9] = self.param.A62, self.param.A66, self.param.A74, self.param.A88, \
                                           self.param.A92, self.param.A106, self.param.A121, self.param.A129, \
                                           self.param.A135, self.param.A1310

        B_matrix = torch.zeros((self.param.state_dim - 2, 1))
        B_matrix[0, 0] = self.param.B11
        B_matrix[1, 0] = self.param.B21
        B_matrix[2, 0] = self.param.B31

        X_dot_batch = torch.zeros((batch_size, self.param.state_dim - 2))
        for batch in range(batch_size):
            X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix),
                                  state[batch, :self.param.state_dim - 2]) +
                     torch.matmul(torch.inverse(M_matrix), torch.matmul(B_matrix, action[batch, :]))).squeeze()
            X_dot_batch[batch, :] = X_dot
        # state_next[:, :self.param.state_dim - 2] = state_curr[:, :self.param.state_dim - 2] + self.dt * X_dot_batch
        state_next[:, :self.param.state_dim - 3] = state[:, :self.param.state_dim - 3] + self.dt * X_dot_batch[:,
                                                                                        :self.param.state_dim - 3]
        state_next[:, 12] = state_next[:, 11].clone() - self.param.b * torch.sin(
            state_next[:, 8].clone()) - self.param.e * torch.sin(state_next[:, 9].clone())  # posy_trailer
        state_next[:, 13] = state[:, 13] + self.dt * self.param.v_x  # x_tractor
        state_next[:, 14] = state_next[:, 13].clone() - self.param.b * torch.cos(
            state_next[:, 8].clone()) - self.param.e * torch.cos(state_next[:, 9].clone())  # x_trailer
        return state_next