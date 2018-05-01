#coding:utf-8

import numpy as np

def da(y,y_p,x):
    return (y_p - y) * x

def db(y,y_p):
    return (y_p - y)

def calc_loss(a,b,x,y):
    tmp = y - (a * x + b)
    tmp = tmp ** 2  
    SSE = sum(tmp) / (2 * len(x))
    return SSE