# 封装数据加载
from src.train.data_loader import load_dataset

class DataManager:
    def load(self):
        return load_dataset()