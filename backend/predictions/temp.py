import pandas as pd
import matplotlib.pyplot as plt

# 加载 .pkl 文件
file1_path = 'LSTM/AAPL_predictions.pkl'
data1 = pd.read_pickle(file1_path)
print(data1)

file2_path = 'LGBM/AAPL_predictions.pkl'
data2 = pd.read_pickle(file2_path)
print(data2)

file3_path = 'XGBoost/AAPL_predictions.pkl'
data3 = pd.read_pickle(file3_path)
print(data3)
