import numpy as np
import torch
import os

# 选择路径
result_path = './checkpoints/SLFnet/test/' + 'model_test.tar'

# 加载需要分析的数据
result_file = torch.load(result_path)
loss_list = result_file['loss_list']    # 每个样本的损失
time_list = result_file['time_list']    # 每个样本的运行时间

# 验证集长度
num = len(loss_list)

# 分类提取损失函数
MAE = []
RMSE = []
iMAE = []
iRMSE = []

for i in range(num):
    MAE.append(loss_list[i]['MAE'].item())
    RMSE.append(loss_list[i]['RMSE'].item())
    iMAE.append(loss_list[i]['iMAE'].item())
    iRMSE.append(loss_list[i]['iRMSE'].item())

# 预测精度分析（以RMSE作为参考）
max_loss = max(RMSE)
min_loss = min(RMSE)

# 平均精度分析
average_MAE = sum(MAE) / num
average_RMSE = sum(RMSE) / num
average_iMAE = sum(iMAE) / num
average_iRMSE = sum(iRMSE) / num

# 运行时间分析
# 注意，运行时间仅包括推理时间
max_time = max(time_list[1:])
min_time = min(time_list[1:])
average_time = sum(time_list[1:]) / num

# 考虑到第一个运行的样本中包含了模型加载的时间，
# 简便起见，忽略第一个样本的运行时间

# 输出结果
print('max loss(RMSE) : %.2f (mm)' %(max_loss))
print('min loss(RMSE) : %.2f (mm)' %(min_loss))

print('MAE loss : %.2f (mm)'   %(average_MAE))
print('RMSE loss : %.2f (mm)'%(average_RMSE))
print('iMAE loss : %.4f (1/km)'%(average_iMAE))
print('iRMSE loss : %.4f (1/km)'%(average_iRMSE))

print('max time : %.2f(ms)'%(max_time*1e3))
print('min time : %.2f(ms)'%(min_time*1e3))
print('avg time : %.2f(ms)'%(average_time*1e3))

RMSE.sort()
print('Min 1000 rmse loss: %.2f(mm)'%(sum(RMSE[:1000])/1000))

# 绘制数据分布情况，以均方根误差RMSE分布作为参考
# 绘图太麻烦了，直接统计数量分布
dist = {}
dist['0-1200'] = sum(i < 1200 for i in RMSE)
dist['2000+'] = sum(i > 2000 for i in RMSE)
dist['1200-2000'] = num - dist['0-1200'] - dist['2000+']
print(dist)

# 绘制 rmse 的统计直方图
import matplotlib.pyplot as plt
plt.figure()
plt.hist(np.array(RMSE), bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("rmse")
plt.ylabel("Number")
plt.show()