import numpy as np

## 评估指标
def rmse(true,pre):
    return np.sqrt(np.mean(np.square(true-pre)))

def mae(true, pre):
    return np.mean(np.abs(true - pre))

def mse(true,pre):
    return np.mean(np.square(true-pre))

def mape(true, pre):
    return np.mean(np.abs((true - pre) / true)) * 100 #单位%

## zero-score归一化方法
class Standardscale():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = self.mean
        std = self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean =  self.mean
        std = self.std

        # 如果参数是数组，数据是不一样的数组时，采用第一列特征进行放缩
        if len(mean.shape) != 0: # 参数不是数值并且与数据长度不一致时转为数值
            if data.shape[-1] != mean.shape[-1]:
                mean = mean[0]
                std = std[0]

        return (data * std) + mean

## Max-min最大最小归一化
class Maxminscale():
    def __init__(self):
        self.min = 0.
        self.max = 1.
    
    def fit(self, data):
        self.max = data.max(0)
        self.min = data.min(0)

    def transform(self, data):
        max = self.max
        min = self.min
        return (data - min) / (max - min)

    def inverse_transform(self, data):
        max =  self.max
        min = self.min

        if len(max.shape) != 0:
            if data.shape[-1] != max.shape[-1]: #如果数据的维度和参数的维度不一致 采用第0个
                max = max[0]
                min = min[0]
        return (data * (max - min)) + min

## 无需归一化
class Nonescale():
    def __init__(self):
        pass
    
    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data