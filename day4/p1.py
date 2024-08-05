import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random
import matplotlib.pyplot as plt


def load_data():
    dataFile = './data/housing.data'
    data = np.fromfile(dataFile, sep=' ', dtype=np.float32)
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                     'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    print(data.shape[0])
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    print(data.shape)


load_data()
