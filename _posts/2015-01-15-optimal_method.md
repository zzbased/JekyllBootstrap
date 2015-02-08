---
layout: post
title: "最优化方法"
description: ""
category: 
tags: []
---
{% include JB/setup %}
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 最优化方法

一般我们接触到的最优化分为两类：

- 无约束最优化
- 有约束最优化

## 无约束最优化

通常对于无约束最优化，首先要判断是否为凸函数。

[无约束最优化](http://www.52nlp.cn/unconstrained-optimization-one)

[机器学习中导数最优化方法](http://www.cnblogs.com/daniel-D/p/3377840.html)

[最大似然、逻辑回归和随机梯度训练](http://cseweb.ucsd.edu/~elkan/250B/logreg.pdf)

### 梯度下降法

### 牛顿法

### 拟牛顿法

### 共轭梯度法

## 有约束最优化

一般采用拉格朗日方程，kkt，对偶问题求解。[关于拉格朗日乘子法与KKT条件](http://www.moozhi.com/topic/show/54a8a261c555c08b3d59d996)

譬如svm里，最大化几何间隔 max y(wx+b)/||w||

[支持向量机](http://blog.csdn.net/v_july_v/article/details/7624837)

首先写出cost function：min [ 1/2*w^2 + max(0, 1 - y(wx+b) ) ]

可以看出，这是一个有约束的问题，那么就可以用到"拉普拉斯+KKT+对偶"来求解了。

## 最优化算法的并行化
### Logistic Regression

这里主要以Logistic regression为例，讲一讲最优化算法的并行化实现。

先看一下Logistic regression的损失函数：

Logistic函数（或称为Sigmoid函数）：
$$sigmoid(z)=\frac{1}{1+e^{-z}}$$
	
对于线性回归来说，其分类函数为：
$$h(x)=w_0+\sum_{i=1}^p{w_ix_i}=w^Tx$$
其中p是输入向量x的维度，也就是特征向量维度；w是特征权重向量。

逻辑回归本质上是一个被logistic函数归一化后的线性回归，即在特征到结果的映射中加入了一层sigmoid函数映射。相比于线性回归，模型输出取值范围为[0，1]，

如果y的取值是0或1，定义事件发生的条件概率为：
$$P(y=1|x)=\pi(x)=\frac{1}{1+exp(-h(x))}$$

定义事件不发生的条件概率为：
$$P(y=0|x)=1-P(y=1|x)=\frac{1}{1+exp(h(x))}$$

假设有n个观测样本，分别为：(\\(x_1\\),\\(y_1\\))，(\\(x_2\\),\\(y_2\\)) ... (\\(x_n\\),\\(y_n\\))。得到一个观测值(\\(x_i\\),\\(y_i\\))的概率为：
$$P(y_i)=p_i^{y_i}(1-p_i)^{1-y_i}$$  其中\\(p_i=P(y_i=1|x_i)=\pi(x_i)\\)
	
由于各项观测独立，所以它们的联合分布可以表示为各边际分布的乘积：
$$l(w)=\prod_{i=1}^n{p_i^{y_i}(1-p_i)^{1-y_i}}$$

对上述函数取对数，根据最大似然估计，得到最优化目标为：
$$\max{L(w)}=\max{log[l(w)]}=\max{\sum_{i=1}^n{y_i*log[\pi(x_i)] + (1-y_i)*log[1-\pi(x_i)]}}$$
而如果y的取值是1或-1，则最优化目标为：

$$\max{L(w)}=\max{\sum_{i=1}^n{-log[1+exp(-y_iw^Tx_i)]}}$$
	
加上正则项后则是：

$$\min_w \frac{1}{2}||w||^2+C\sum_{i=1}^{n}log[1+exp(-y_iw^Tx_i)]$$
### LR的MapReduce并行![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lbfgs.png)
	
![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lbfgs_two_loops.png)

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/lbfgs_two_loops_vf.png)
- [Distributed LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/)
- [Large scale learning spotlights](http://nips.cc/Conferences/2014/Program/event.php?ID=4831)

## Loss Function

[loss.pdf](http://web.mit.edu/lrosasco/www/publications/loss.pdf)

[vowpal_wabbit Loss-functions ](https://github.com/JohnLangford/vowpal_wabbit/wiki/Loss-functions)

[Loss function wiki](http://en.wikipedia.org/wiki/Loss_function)

[shark loss function](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/concepts/library_design/losses.html)

adaboost，svm，lr三个算法的关系：

三种算法的分布对应exponential loss（指数损失函数），hinge loss，log loss（对数损失函数），无本质区别。应用凸上界取代0、1损失，即凸松弛技术。从组合优化到凸集优化问题。凸函数，比较容易计算极值点。
