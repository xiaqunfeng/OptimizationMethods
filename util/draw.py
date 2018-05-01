#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from .calculate import calc_loss

def draw_hill(x,y):
    a = np.linspace(-20,20,100)
    b = np.linspace(-20,20,100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            SSE = calc_loss(a=a0,b=b0,x=x,y=y)
            allSSE[ai][bi] = SSE

    a,b = np.meshgrid(a, b)

    return [a,b,allSSE]

# 绘制左上角的 loss 曲面分布图
def draw_curved_surface(ha, hb, hallSSE):
    curved_surface = plt.subplot(2, 2, 1, projection='3d')
    curved_surface.set_top_view()
    curved_surface.plot_surface(ha, hb, hallSSE, rstride=2, cstride=2, cmap='rainbow')
    plt.title(u"loss分布")
    return curved_surface

# 绘制曲面上的loss
def draw_curved_surface_loss(a,b,loss,curved_surface):
	curved_surface.scatter(a, b, loss, color='black')

# 绘制右上角的等高线图
def draw_contour_line(ha, hb, hallSSE):
	plt.subplot(2,2,2)
	ta = np.linspace(-20, 20, 100)
	tb = np.linspace(-20, 20, 100)
	plt.contourf(ha,hb,hallSSE,15,alpha=0.5,cmap=plt.cm.hot)
	C = plt.contour(ha,hb,hallSSE,15,colors='black')
	plt.clabel(C,inline=True)
	plt.xlabel('a')
	plt.ylabel('b')
	plt.title(u"等高线")

# 绘制等高线上的 loss
def draw_equal_altitude_loss(a,b,last_a,last_b):
	plt.subplot(2,2,2)
	plt.scatter(a,b,s=5,color='blue')
	plt.plot([last_a,a],[last_b,b],color='aqua')

# 绘制左下角的回归直线
def draw_regression_line(a,b,x,y):
	plt.subplot(2, 2, 3)
	plt.plot(x, y)
	plt.plot(x, y, 'o')
	x_ = np.linspace(0, 1, 2)
	y_draw = a * x_ + b
	plt.plot(x_, y_draw)
	plt.xlabel('a')
	plt.ylabel('b')
	plt.title(u"回归直线")

# 绘制右下角的loss曲线
def draw_loss(loss,step,all_loss,all_step):
	all_loss.append(loss)
	all_step.append(step)
	plt.subplot(2,2,4)
	plt.plot(all_step,all_loss,color='orange')
	plt.xlabel("step")
	plt.ylabel("loss")
	plt.title(u"loss曲线")

