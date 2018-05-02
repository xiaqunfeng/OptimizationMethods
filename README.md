# 机器学习优化函数

1、线性回归

线性模型：y = ax + b

Loss方法：Mean Squared Error (MSE), 即均方差

![mse](pic/loss-mse.jpg)

实验：

```
▶ python 1_linear_regression.py
loss = 0.013575
```

结果图：

![1](pic/1.jpg)

2、梯度下降

每一次迭代按照一定的学习率 αα 沿梯度的反方向更新参数，直至收敛。

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$

\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)

$${loss=\frac{1}{2m}\sum_{i=1}^m(y_{p,i}-y_i)^2 }$$

\\({loss=\frac{1}{2m}\sum_{i=1}^m(y_{p,i}-y_i)^2 }\\)

\\(loss=\frac{1}{2m}\sum_{i=1}^m(y_{p,i}-y_i)^2 \\)



grandent_descent.py：梯度下降

SGD.py：随机梯度下降

minibatch_SGD.py：minibatch SGD

momentum.py：动量

momentum_SGD.py：动量 SGD

Nesterov_momentum.py

adagrad.py: adagrad

adadelta.py: adadelta

adam.py: adam

### 算法



梯度下降的基本步骤是:

1. 对成本函数进行微分, 得到其在给定点的梯度. 梯度的正负指示了成本函数值的上升或下降:
2. 选择使成本函数值减小的方向, 即梯度负方向, 乘以以学习率 计算得参数的更新量, 并更新参数:
3. 重复以上步骤, 直到取得最小的成本

梯度下降，又称批量梯度下降

在每次更新时用所有样本，要留意，在梯度下降中，对于     的更新，所有的样本都有贡献，也就是参与    调整 .其计算得到的是一个标准梯度，**对于最优化问题，凸问题，**也肯定可以达到一个全局最优。因而理论上来说一次更新的幅度是比较大的。如果样本不多的情况下，当然是这样收敛的速度会更快啦。

随机梯度算法

在每次更新时用1个样本，可以看到多了随机两个字，随机也就是说我们用样本中的一个例子来近似我所有的样本，来调整*θ*，因而随机梯度下降是会带来一定的问题，因为计算得到的并不是准确的一个梯度，**对于最优化问题，凸问题，**虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。但是相比于批量梯度，这样的方法更快，更快收敛，虽然不是全局最优，但很多时候是我们可以接受的，所以这个方法用的也比上面的多。

mini-batch梯度下降

在每次更新时用b个样本,其实批量的梯度下降就是一种折中的方法，他用了一些小样本来近似全部的，其本质就是我1个指不定不太准，那我用个30个50个样本那比随机的要准不少了吧，而且批量的话还是非常可以反映样本的一个分布情况的。在深度学习中，这种方法用的是最多的，因为这个方法收敛也不会很慢，收敛的局部最优也是更多的可以接受！

$c^2$



#### 来源

代码来源于以下博客系列文章，稍作改动和重构

http://blog.csdn.net/column/details/19920.html

#### 参考

p1 参考 https://zhuanlan.zhihu.com/p/27297638

p2~pn 参考 http://ruder.io/optimizing-gradient-descent/index.html

###### 单独参考

p5 参考 http://cs231n.github.io/neural-networks-3/

p6 参考 https://zhuanlan.zhihu.com/p/22252270

p7 参考 https://arxiv.org/abs/1212.5701 (原始论文)

p8 参考 http://www.ijiandao.com/2b/baijia/63540.html
