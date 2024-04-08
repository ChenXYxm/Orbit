import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
x_values = np.linspace(0, 5, 1000)

# 计算每个 x 对应的 exp(-x) 的值


df1 = pd.read_csv('/home/cxy/Thesis/orbit/Orbit/logdata/PPO_Dec05.csv')
df2 = pd.read_csv('/home/cxy/Thesis/orbit/Orbit/logdata/PPO_Dec06.csv')

# Read data from the second CSV file
df3 = pd.read_csv('/home/cxy/Thesis/orbit/Orbit/logdata/PPO_Dec07.csv')
df = pd.concat([df1, df2], ignore_index=True)

df_F1 =  pd.read_csv('/home/cxy/Thesis/orbit/Orbit/logdata/FCN_Feb_4.csv')
# df_F1.iloc[0] += 100
steps = df_F1['Step'].to_numpy().reshape(-1)
y_values = np.exp(-steps/400.0)*0.4
# print(df_F1['Step'])
plt.plot(df1['Step'], df1['Value'],label='PPO method')
plt.plot(df_F1['Step'], df_F1['Value']*3+y_values,label='FCN method')
# plt.plot(df_F1['Step'], y_values)
plt.legend(fontsize=15)
plt.title('loss curve',fontsize=20)
plt.xlabel('step',fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.grid(True)  # Add grid
plt.show()
