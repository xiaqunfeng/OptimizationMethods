#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def data_init():
	x = np.array([30,35,37,59,70,76,88,100]).astype(np.float32)
	y = np.array([1100,1423,1377,1800,2304,2588,3495,4839]).astype(np.float32)

	x_max = max(x)
	x_min = min(x)
	y_max = max(y)
	y_min = min(y)

	# 归一化到 0~1 之间
	for i in range(0,len(x)):
	    x[i] = (x[i] - x_min)/(x_max - x_min)
	    y[i] = (y[i] - y_min)/(y_max - y_min)

	return x,y

if __name__ == '__main__':
	x,y = data_init()
	
	# 设置 a,b 的值
	a,b = 1,0
	x_ = np.array([0,1])
	y_ = a*x_+b

	yp = a*x +b	    # 计算预测值

	# 计算loss: Mean Squared Error (MSE), 即均方差
	loss = sum(np.square(np.round(yp-y,4))) / (2*len(x))
	print("loss = %f" % loss)

	plt.xlabel(u"x轴")
	plt.ylabel(u"y轴")

	plt.scatter(x,y)				# 绘制散点图
	plt.plot(x_,y_,color='green')	# 绘制 y_ = a*x_+b
	plt.pause(5)
