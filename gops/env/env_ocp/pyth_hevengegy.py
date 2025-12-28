import numpy as np
import pandas as pd
from scipy import interpolate

import torch

########## 注意序号从0开始 ##########
########## 注意调用时方法加括号 ##########


class Hev:
    def __init__(self):
        self.state_size = 3  # 状态为Pdem，SOC，SOC_uc
        self.a1_size = 9  # 动作1，Pe
        self.a2_size = 10  # 动作2，Pbat
        self.action_size = 9*10  # a为0-18*19-1
        # self.action_space = np.linspace(3, 90, self.action_size)

        # 循环工况
        # df1 = pd.read_excel('cyc_LA92.xlsx')
        # df1 = pd.read_excel('cyc_NEDC.xlsx')
        df1 = pd.read_excel('cyc_HWFET.xlsx')
        self.pdem_cyc = np.array(df1['Pdem'])  # 这里的数据类型是float64，可能在后面导致问题

    def reset(self):
        t_initial = 0
        t_terminal = len(self.pdem_cyc)-1  # 长度为1436，t为0-1435
        pdem_initial = min(max(self.pdem_cyc[t_initial], -20), 140)
        soc_initial = 0.6
        soc_uc_initial = 0.7
        s_initial = np.array([pdem_initial, soc_initial, soc_uc_initial])
        return s_initial, t_initial, t_terminal

    def step(self, s, a, t):
        pdem = s[0]
        soc = s[1]
        soc_uc = s[2]
        a1, a2 = np.unravel_index(a, (self.a1_size, self.a2_size), order='F')  # 线性索引转换为下标，同时按matlab习惯先排布列
        pe = (a1+1)*10  # a1为0-8，转换为pe的10-90 发动机功率
        pbat = (a2-4)*10  # a2为0-9，转换为pb的-40-50 电池功率

        dt = 1  # 时间间隔为1s

        ###### 更新需求功率 #####
        t += 1
        pdem_ = self.pdem_cyc[t]  # t从0到1435
        pdem_ = min(max(pdem_, -20), 140)  # 限制需求功率上下限

        ##### 更新SOC #####
        series = 182
        parallel = 4
        cbat = 5 * parallel * 3600

        # 内阻
        rint_data = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
             [0.00249761120679617, 0.00232695461297971,  0.00221607526377662, 0.00221607526377662, 0.00221607526377662,
              0.00214347159301951,  0.00160593178722349, 0.0015395425036134, 0.0015395425036134, 0.0015395425036134]]
        )
        f_rint_soc = interpolate.interp1d(rint_data[0, :], rint_data[1, :], kind='slinear')
        rint = f_rint_soc(soc)*series/parallel

        # 开路电压
        ocv_soc = np.array([8526, -2.184e+4, 2.164e+4, -1.052e+4, 2669, 2992])
        voc = (ocv_soc[0]*soc**5+ocv_soc[1]*soc**4+ocv_soc[2]*soc**3+ocv_soc[3]*soc**2+ocv_soc[4]*soc+ocv_soc[5])/1000*series

        # SOC
        i = (voc-(voc**2-4*rint*(pbat*1000))**0.5)/(2*rint)
        soc_ = -i*dt/cbat+soc
        soc_ = min(max(soc_, 0.4), 0.8)  # 限制soc上下限

        ##### 更新超级电容SOC #####
        series_uc = 12
        parallel_uc = 1
        u_uc_cell = 48  # 单体电压，V
        c_uc_cell = 300  # 单体电容，F
        r_uc_cell = 6.3e-3  # 单体内阻，欧
        c_uc = u_uc_cell * c_uc_cell * parallel_uc  # 容量
        u_uc = u_uc_cell * series_uc  # 上截止电压

        # 内阻
        r_uc = r_uc_cell * series_uc / parallel_uc
        # 开路电压
        voc_uc = u_uc * soc_uc

        # 超级电容功率
        df2 = pd.read_excel('optimal_curve.xlsx')
        pe_optimal = np.array(df2['engine_power'])
        pg_optimal = np.array(df2['generator_power'])
        fuel_optimal = np.array(df2['fuel'])
        f_pg_pe = interpolate.interp1d(pe_optimal, pg_optimal, kind='slinear')
        pg = f_pg_pe(pe)

        puc = pdem - pg - pbat

        # SOC_uc
        soc_uc_ = (-voc_uc+(voc_uc**2-4*r_uc*(puc*1000))**0.5) / (2 * c_uc * r_uc) * dt + soc_uc
        soc_uc_ = min(max(soc_uc_, 0.5), 1)  # 限制soc上下限

        ##### 下一时刻状态 #####
        s_ = np.array([pdem_, soc_, soc_uc_])

        ##### 计算奖励 #####
        # 燃油消耗
        f_fuel_pe = interpolate.interp1d(pe_optimal, fuel_optimal, kind='slinear')
        fuel = f_fuel_pe(pe)

        # SOC限制
        delta_soc = soc_-0.6

        # SOH惩罚
        i_1c = 20
        c_rate = abs(i/i_1c)

        B_c_data = np.array([[0.5, 2, 6, 10], [31630, 21681, 12934, 15512]])
        c_rate_1 = min(max(c_rate, 0.5), 10)
        f_B_c = interpolate.interp1d(B_c_data[0, :], B_c_data[1, :], kind='slinear')
        B = f_B_c(c_rate_1)
        Ea = 31700 - 370.3*c_rate
        R = 8.31
        T = 313
        z = 0.55

        Ah = (20/(B*np.exp(-Ea/R/T)))**(1/z)
        N = 3600*Ah/cbat
        delta_soh = -abs(i) * dt / (2 * N * cbat)

        # SOC_uc限制
        delta_soc_uc = soc_uc_-0.7

        ##### 奖励 #####
        r = -fuel-5e5*abs(delta_soc)**2-5e7*abs(delta_soh)-1.25e4*abs(delta_soc_uc)

        return s_, r, t


