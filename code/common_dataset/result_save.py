import numpy as np
import pickle

# 你的二维数组
data = [[26, 0, 1, 0, 0, 0, 0, 1],
        [0, 21, 0, 0, 0, 0, 0, 0],
        [0, 2, 25, 0, 1, 0, 0, 0],
        [0, 0, 0, 29, 0, 0, 0, 0],
        [1, 1, 0, 0, 27, 0, 0, 0],
        [0, 0, 0, 0, 0, 20, 1, 0],
        [0, 0, 1, 1, 0, 2, 16, 0],
        [3, 0, 0, 1, 1, 0, 0, 13]]

# 将二维数组转换为NumPy数组
array = np.array(data)

# 保存为二进制文件
with open("/home/ruixing/workspace/bc_interface/result/cm.pkl", "wb") as file:
    pickle.dump(array, file)

# 加载二进制文件
with open("/home/ruixing/workspace/bc_interface/result/cm.pkl", "rb") as file:
    loaded_array = pickle.load(file)

# 打印加载的数组
print(loaded_array)
