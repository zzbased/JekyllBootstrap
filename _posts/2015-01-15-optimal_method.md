---
layout: post
title: "optimal_method"
description: ""
category: 
tags: []
---
{% include JB/setup %}


## 最优化方法

一般我们接触到的最优化分为两类：

- 无约束最优化
- 有约束最优化

### 无约束最优化

通常对于无约束最优化，首先要判断是否为凸函数。

[无约束最优化](http://www.52nlp.cn/unconstrained-optimization-one)

[机器学习中导数最优化方法](http://www.cnblogs.com/daniel-D/p/3377840.html)

[最大似然、逻辑回归和随机梯度训练](http://cseweb.ucsd.edu/~elkan/250B/logreg.pdf)

#### 梯度下降法

#### 牛顿法

#### 拟牛顿法

#### 共轭梯度法


### 有约束最优化

一般采用拉格朗日方程，kkt，对偶问题求解。[关于拉格朗日乘子法与KKT条件](http://www.moozhi.com/topic/show/54a8a261c555c08b3d59d996)

譬如svm里，最大化几何间隔 max y(wx+b)/||w||

[支持向量机](http://blog.csdn.net/v_july_v/article/details/7624837)

首先写出cost function：min [ 1/2*w^2 + max(0, 1 - y(wx+b) ) ]

可以看出，这是一个有约束的问题，那么就可以用到"拉普拉斯+KKT+对偶"来求解了。

### 最优化算法的并行化

- [Distributed LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/)
- [Large scale learning spotlights](http://nips.cc/Conferences/2014/Program/event.php?ID=4831)

## Loss Function

[loss.pdf](http://web.mit.edu/lrosasco/www/publications/loss.pdf)

[vowpal_wabbit Loss-functions ](https://github.com/JohnLangford/vowpal_wabbit/wiki/Loss-functions)

[Loss function wiki](http://en.wikipedia.org/wiki/Loss_function)

[shark loss function](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/concepts/library_design/losses.html)