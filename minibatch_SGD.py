#coding:utf-8
# 本代码是一个最简单的线形回归问题，优化函数为minibatch SGD

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from linear_regression import data_init
from grandent_descent import da, db, calc_loss, draw_hill
from SGD import shuffle_data

rate = 0.1 # learning rate

def get_batch_data(x,y,batch=3):
    shuffle_data(x,y)
    x_new = x[0:batch]
    y_new = y[0:batch]
    return [x_new,y_new]

if __name__ == '__main__':
    #  模拟数据
    x,y = data_init()

    [ha,hb,hallSSE] = draw_hill(x,y)
    hallSSE = hallSSE.T# 重要，将所有的losses做一个转置。原因是矩阵是以左上角至右下角顺序排列元素，而绘图是以左下角为原点。
    # 初始化a,b值
    a = 10.0
    b = -20.0
    fig = plt.figure(1, figsize=(12, 8))

    # 绘制图1的曲面
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_top_view()
    ax.plot_surface(ha, hb, hallSSE, rstride=2, cstride=2, cmap='rainbow')

    # 绘制图2的等高线图
    plt.subplot(2,2,2)
    ta = np.linspace(-20, 20, 100)
    tb = np.linspace(-20, 20, 100)
    plt.contourf(ha,hb,hallSSE,15,alpha=0.5,cmap=plt.cm.hot)
    C = plt.contour(ha,hb,hallSSE,15,colors='black')
    plt.clabel(C,inline=True)
    plt.xlabel('a')
    plt.ylabel('b')

    plt.ion() # iteration on

    all_loss = []
    all_step = []
    last_a = a
    last_b = b
    for step in range(1,100):
        loss = 0
        all_da = 0
        all_db = 0
        shuffle_data(x,y)
        [x_new,y_new] = get_batch_data(x,y,batch=4)
        for i in range(0,len(x_new)):
            y_p = a*x_new[i] + b
            loss = loss + (y_new[i] - y_p)*(y_new[i] - y_p)/2
            all_da = all_da + da(y_new[i],y_p,x_new[i])
            all_db = all_db + db(y_new[i],y_p)
        #loss_ = calc_loss(a = a,b=b,x=np.array(x),y=np.array(y))
        loss = loss/len(x_new)

        # 绘制图1中的loss点
        ax.scatter(a, b, loss, color='black')
        # 绘制图2中的loss点
        plt.subplot(2,2,2)
        plt.scatter(a,b,s=5,color='blue')
        plt.plot([last_a,a],[last_b,b],color='aqua')
        # 绘制图3中的回归直线
        plt.subplot(2, 2, 3)
        plt.plot(x, y)
        plt.plot(x, y, 'o')
        x_ = np.linspace(0, 1, 2)
        y_draw = a * x_ + b
        plt.plot(x_, y_draw)
        # 绘制图4的loss更新曲线
        all_loss.append(loss)
        all_step.append(step)
        plt.subplot(2,2,4)
        plt.plot(all_step,all_loss,color='orange')
        plt.xlabel("step")
        plt.ylabel("loss")


        # print('a = %.3f,b = %.3f' % (a,b))
        last_a = a
        last_b = b
        a = a - rate*all_da
        b = b - rate*all_db

        if step%1 == 0:
            print("step: ", step, " loss: ", loss)
            plt.show()
            plt.pause(0.01)
    plt.show()
    plt.pause(99999999999)