#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from util.init import data_init


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

	plt.figure(u'线性回归')
	plt.xlabel(u"x轴")
	plt.ylabel(u"y轴")

	plt.scatter(x,y)				# 绘制散点图
	plt.plot(x_,y_,color='green')	# 绘制 y_ = a*x_+b
	plt.pause(5)
