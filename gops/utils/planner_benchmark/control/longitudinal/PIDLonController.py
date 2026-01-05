from collections import deque
import numpy as np


class PIDLonController(object):

    def __init__(self, max_accel=3.0, dt=0.1):
        """
        全 SI 单位制 (m/s)
        """
        # --- 1. 参数调整建议 ---
        # K_P: 建议 1.0 ~ 2.0。意思是每差 1m/s，多给 1m/s^2 的加速度
        self._K_P = 1.5

        # K_I: 建议 0.1 ~ 0.5。用于消除稳态误差(对抗风阻)
        self._K_I = 0.2

        # K_D: 速度控制通常不需要D，或者给很小(0.01)，否则信号由于噪声会抖动
        self._K_D = 0.0

        self._dt = dt
        self._max_accel = max_accel

        # 积分防饱和 (Anti-windup) 限制
        self._integ_limit = max_accel
        self._integ = 0.0
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, current_speed, target_accel=0.0):
        """
        :param target_speed: m/s
        :param current_speed: m/s
        :param target_accel: m/s^2 (前馈量，从规划模块获取)
        """
        return self._pid_control(target_speed, current_speed, target_accel)

    def _pid_control(self, target_speed, current_speed, target_accel):
        # 1. 误差计算 (m/s)
        error = target_speed - current_speed

        # 2. 积分项 (带防饱和与重置逻辑)
        # 如果误差方向改变，或者处于停车状态，重置积分
        if error * self._integ < 0 or target_speed < 0.1:
            self._integ = 0.0

        self._integ += error * self._dt
        # 积分限幅
        self._integ = np.clip(self._integ, -self._integ_limit, self._integ_limit)

        self._e_buffer.append(error)

        # 3. 微分项
        if len(self._e_buffer) >= 2:
            d_error = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
        else:
            d_error = 0.0

        # 4. PID 计算
        pid_output = (self._K_P * error) + (self._K_I * self._integ) + (self._K_D * d_error)

        # 5. 【关键】加入前馈 (Feedforward)
        # 最终输出 = PID反馈纠正 + 规划期望加速度
        final_acc = pid_output + target_accel

        # 6. 限幅
        return np.clip(final_acc, -self._max_accel, self._max_accel)