# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备理想的位置数据
ts = np.linspace(1, 100, 100)
a_truth = 0.5  # 理想的加速度值
position_truth = a_truth * ts**2 / 2  # 理想的位置
plt.plot(ts, position_truth, label="Truth Position", color="green", linewidth=1)

# 2. 准备GPS观测的位置数据，带有噪声
u_gps = 0  # gps观测数据的正态分布均值
std_gps = 100  # gps观测数据的正态分布标准差（需要仪器根据Allen方差标定得出）
var_gps = std_gps**2
postion_gps = position_truth + np.random.normal(
    u_gps, std_gps, ts.shape[0]
)  # GPS观测的位置，带有噪声
plt.scatter(ts, postion_gps, label="GPS Position", color="purple", s=5)

# 3. 准备速度计观测的速度数据，带有噪声
u_imu = 0  # imu观测数据的正态分布均值
std_imu = 100  # imu观测数据的正态分布标准差（需要仪器根据Allen方差标定得出）
var_imu = std_imu**2
velocity_imu = a_truth * ts + np.random.normal(
    u_imu, std_imu, ts.shape[0]
)  # imu观测的速度，带有噪声

# 4. 使用kalman滤波估计位置
postion_predict = [postion_gps[0]]  # 初始位置就直接用gps观测得到的
u_postion_predict = [u_gps]
var_postion_predict = [var_gps]
xt_plt = [postion_gps[0]]

for t in range(1, ts.shape[0]):
    # STEP1. 结合t-1时刻的postion_predict，u_postion_predict， std_postion_predict，t时刻的velocity_imu，u_imu，std_imu，得到靠imu计算出来的粗略估计值xt, u_xt, std_xt
    xt = postion_predict[t - 1] + velocity_imu[t - 1] * 1  # 1是时间间隔
    u_xt = xt
    var_xt = var_postion_predict[t - 1] + var_imu
    # STEP2. 拿到t时刻的gps观测值zt, u_zt, std_zt
    zt = postion_gps[t]
    u_zt = zt
    var_zt = var_gps
    # STEP3. 进行数据融合，其实就是将两个正态分布相乘 N(u_xt, var_xt) * N(u_zt, var_zt) = N(u_postion_predict[t], var_postion_predict[t])
    u_postion_predict.append((u_xt * var_zt + u_zt * var_xt) / (var_xt + var_zt))
    var_postion_predict.append(var_xt * var_zt / (var_xt + var_zt))
    postion_predict.append(u_postion_predict[t])

    xt_plt.append(xt)

plt.scatter(ts, xt_plt, label="IMU Position", color="gold", s=5)
plt.plot(ts, postion_predict, label="Kalman Position", color="blue", linewidth=1)


def draw_point_value(data):
    for i in range(0, len(ts), 10):
        plt.annotate(
            f"{data[i]:.2f}",
            (ts[i], data[i]),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            fontsize=5,
        )


draw_point_value(postion_predict)
draw_point_value(xt_plt)
draw_point_value(postion_gps)

plt.legend()
plt.show()
