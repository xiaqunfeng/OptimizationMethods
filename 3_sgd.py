#coding:utf-8
# 本代码是一个最简单的线形回归问题，优化函数为经典的 SGD

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from util.data import data_init, shuffle_data
from util.calculate import da, db, calc_loss
from util import draw

rate = 0.1 # learning rate

if __name__ == '__main__':    
    #  模拟数据
    x,y = data_init()
    # 初始化a,b值
    a = 10.0
    b = -20.0

    [ha,hb,hallSSE] = draw.draw_hill(x,y)
    # 重要，将所有的losses做一个转置。原因是矩阵是以左上角至右下角顺序排列元素，而绘图是以左下角为原点。
    hallSSE = hallSSE.T

    # 初始化图片
    fig = plt.figure('SGD', figsize=(11, 7))
    # 绘制图1的曲面
    curved_surface = draw.draw_curved_surface(ha,hb,hallSSE)
    # 绘制图2的等高线图
    draw.draw_contour_line(ha,hb,hallSSE)

    plt.ion() # iteration on

    all_loss = []
    all_step = []
    last_a = a
    last_b = b
    step = 1
    while step <= 100:
        loss = 0
        all_da = 0
        all_db = 0
        shuffle_data(x,y)
        # 随机梯度下降
        i = random.randrange(0,len(x))  # 随机选择一个index
        y_p = a*x[i] + b
        loss = (y[i] - y_p)*(y[i] - y_p)/2
        all_da = da(y[i],y_p,x[i])
        all_db = db(y[i],y_p)

        # 在图1的曲面上绘制 loss 点
        draw.draw_curved_surface_loss(a,b,loss,curved_surface)
        # 在图2的等高线上绘制 loss 点
        draw.draw_equal_altitude_loss(a,b,last_a,last_b)
        # 绘制图3中的回归直线
        draw.draw_regression_line(a,b,x,y)
        # 绘制图4的loss更新曲线
        draw.draw_loss(loss,step,all_loss,all_step)

        last_a = a
        last_b = b

        # 更新参数
        a = a - rate*all_da
        b = b - rate*all_db

        if step%1 == 0:
            print("step: ", step, " loss: ", loss)
            plt.show()
            plt.pause(0.01)
        step = step + 1
    plt.show()
    plt.pause(60)