#coding:utf-8
# 本代码是一个最简单的线形回归问题，优化函数为 adadelta

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from util.data import data_init, shuffle_data, get_batch_data
from util.calculate import da, db, calc_loss
from util import draw

epsilon = 1e-2
gamma = 0.9
  
if __name__ == '__main__':
    #  模拟数据
    x,y = data_init()
    # 初始化a,b值
    a = 10.0
    b = -20.0

    [ha,hb,hallSSE] = draw.draw_hill(x,y)
    # 将所有的loss做一个转置。原因是矩阵是以左上角至右下角顺序排列元素，而绘图是以左下角为原点。
    hallSSE = hallSSE.T

    # 初始化图片
    plt.figure('adadelta', figsize=(11, 7))
    plt.suptitle('Method: adadelta', fontsize=15)

    # 绘制图1的曲面
    curved_surface = draw.draw_curved_surface(ha,hb,hallSSE)
    # 绘制图2的等高线图
    draw.draw_contour_line(ha,hb,hallSSE)

    plt.ion() # iteration on

    all_loss = []
    all_step = []
    last_a = a
    last_b = b
    theta = np.array([0,0]).astype(np.float32) # 每一次a,b迭代的更新值
    E_g = np.array([0,0]).astype(np.float32)
    E_theta = np.array([0,0]).astype(np.float32)

    for step in range(1,200):
        loss = 0
        all_da = 0
        all_db = 0
        [x_new,y_new] = get_batch_data(x,y,batch=8)
        for i in range(0,len(x_new)):
            y_p = a*x_new[i] + b
            loss = loss + (y_new[i] - y_p)*(y_new[i] - y_p)/2
            all_da = all_da + da(y_new[i],y_p,x_new[i])
            all_db = all_db + db(y_new[i],y_p)
        #loss_ = calc_loss(a = a,b=b,x=np.array(x),y=np.array(y))
        all_d = np.array([all_da/len(x_new), all_db/len(x_new)])
        #all_d = np.array([all_da, all_db])
        loss = loss/len(x_new)

        # 在图1的曲面上绘制 loss 点
        draw.draw_curved_surface_loss(a,b,loss,curved_surface)
        # 在图2的等高线上绘制 loss 点
        draw.draw_equal_altitude_loss(a,b,last_a,last_b)
        # 绘制图3中的回归直线
        draw.draw_regression_line(a,b,x,y)
        # 绘制图4的loss更新曲线
        draw.draw_loss(loss,step,all_loss,all_step)

        # 更新参数
        last_a = a
        last_b = b

        E_g = gamma*E_g + (1-gamma)*(all_db**2)
        rms_g = np.sqrt(E_g + epsilon)

        E_theta = gamma*E_theta + (1-gamma)*(theta**2)
        rms_theta = np.sqrt(E_theta + epsilon)

        theta = -(rms_theta/rms_g)*all_d
        [a,b] = [a,b] + theta

        if step%1 == 0:
            print("step: ", step, " loss: ", loss, 'rms_theta/rms_g:', rms_theta/rms_g)
            plt.show()
            plt.pause(0.01)
    plt.show()
    plt.pause(300)