import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math
from gops.utils import mechanical_simulation
import ctypes
import os
import argparse
import random



#  Update: 2024-03-24, Fawang Zhang: create file
#  Description: put this in External Libraries/site-packages/gym/envs/classic_control

#  you need to register this env in gym,
#  1st step: add 'from gym.envs.classic_control.[your env file name] import [your env class name] in classic_control/init.py
#  2nd step: add register(
#     id="the_id_you_want-v0",
#     entry_point="gym.envs.classic_control.[your env file name]:[your env class name]",) in envs/init.py


class Env_Class_Name(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.sim_file_filename = 'D:/2_Install Root/18_TruckSim2019/TruckSim 2019.0_Data/Extensions/Custom_Py/simfile.sim'
        self.vs = mechanical_simulation.VehicleSimulation()
        self.vs_dll = ctypes.cdll.LoadLibrary('D:/2_Install Root/18_TruckSim2019/TruckSim2019.0_Prog/Programs/solvers/trucksim_64.dll')
        if self.vs_dll is not None:
            if self.vs.get_api(self.vs_dll):
                self.configuration = self.vs.ReadConfiguration(self.sim_file_filename)
                self.t_current = self.configuration.get('t_start')

        state_low = np.array([
            -45.0, #slip angle 1
            -20.0,# yaw rate 1
            -45.0, #roll angle 1
            -20.0 ,#roll rate 1
            -45., #slip angle2
            -20.0,#yaw rate 2
            -20.0, # roll angle 2
            -45.0, # roll rate 2
            -45.0, #yaw 1
            -45.0,#yaw 2
            -50,  # vy 1
            -200, #y1
            -200, #y2
            -1000,  # x1
            -1000,  # x2
            -50,  # vy 2
            0,  # vx1
            0,#vx2
            -90, # hitch angle
            -20,  # hitch rate
            -900, # steering wheel angle feedback,
            0 # throttle feedback
        ])
        state_high = np.array([
            45.0,  # slip angle 1
            20.0,  # yaw rate 1
            45.0,  # roll angle 1
            20.0,  # roll rate 1
            45.0,  # slip angle2
            20.0,  # yaw rate 2
            20.0,  # roll angle 2
            45.0,  # roll rate 2
            45.0,  # yaw 1
            45.0,  # yaw 2
            50,  # vy 1
            200,  # y1
            200,  # y2
            100000,  # x1
            100000,  # x2
            50,  # vy 2
            120,  # vx1
            120,  # vx2
            90,  # hitch angle
            20,  # hitch rate
            900,  # steering wheel angle feedback,
            1  # throttle feedback
            ])
        action_high = np.array([1125.0, 1])
        action_low = np.array([-1125.0, 0])
        self.action_space = spaces.Box(action_low, action_high)
        self.observation_space = spaces.Box(state_low, state_high)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.array(action)
        self.import_array = [action]
        self.export_array = np.array(self.state)

        for i in range(50): #make trucksim simulation time step = python time step
            # Call the VS API integration function
            self.status, self.export_array = self.vs.IntegrateIO(self.t_current, self.import_array, self.export_array)
            self.t_current = self.t_current + self.configuration.get('t_step')
        self.state = np.float32(self.export_array)
        reward = 0.0

        # if self.t_current%5 <=0.02:
        #     print('Time = ', self.t_current)
        #     print('State[Vx,Te,Gear,Qfuel,EngSpd,TransSpd,ig,Mfuel,crankshaft input torque, trans output torque] = ',self.state)
        # # print('Time = ', self.t_current)
        # # print('Reward = ',reward)
        # done = bool(self.t_current >= 20000)
        # 
        # if done:
        #     self.vs.TerminateRun(self.t_current)

        return np.array(self.state), reward #, done, {}

    def reset(self):
        self.export_array = self.vs.CopyExportVars(self.configuration.get('n_export'))
        self.state = np.array(self.export_array)
        return np.array(self.state)

    def render(self, mode='human'):
        return None