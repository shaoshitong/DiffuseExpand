import numpy as np
import matplotlib.pyplot as plt

# 生成数据
num_curves = 4
num_points = 10
data = np.random.rand(num_curves, num_points)

# 设置x轴
x = np.arange(num_points)

# 绘图
for i in range(num_curves):
    y = data[i,:]
    plt.bar(x + i * 0.2, y, width = 0.2, label = 'curve ' + str(i + 1))

plt.legend()
plt.show()
