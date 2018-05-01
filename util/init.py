#coding:utf-8
import numpy as np

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
