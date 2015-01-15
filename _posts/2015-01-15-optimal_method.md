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

#### 梯度下降法

#### 牛顿法

#### 拟牛顿法

#### 共轭梯度法

### 有约束最优化

一般采用拉格朗日方程，kkt，对偶问题求解。

譬如svm里，最大化几何间隔 max y(wx+b)/||w||

[支持向量机](http://blog.csdn.net/v_july_v/article/details/7624837)

首先写出cost function：min [ 1/2*w^2 + max(0, 1 - y(wx+b) ) ]

可以看出，这是一个有约束的问题，那么

