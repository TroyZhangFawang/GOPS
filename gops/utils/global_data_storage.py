import pandas as pd

class GlobalDataStorage:
    def __init__(self):
        self.all_obs_data = pd.DataFrame()

    def append_data(self, obs_data):
        self.all_obs_data = pd.concat([self.all_obs_data, obs_data], ignore_index=True)

    def save_data(self, path):
        self.all_obs_data.to_csv(path, index=False)

# 创建全局数据存储实例
global_data_instance = GlobalDataStorage()
