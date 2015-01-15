---
layout: post
title: "gbdt_adaboost_bootstrap"
description: ""
category: 
tags: []
---
{% include JB/setup %}


## gbdt 与 adaboost

这两个算法在一定程度上其实是有些接近的，我们不妨分别来看看。

[文档学习链接](http://suanfazu.com/t/gbdt-die-dai-jue-ce-shu-ru-men-jiao-cheng/135)
[Boosting Decision Tree入门教程](http://www.schonlau.net/publication/05stata_boosting.pdf)
[LambdaMART用于搜索排序入门教程](http://research.microsoft.com/pubs/132652/MSR-TR-2010-82.pdf)

### gbdt

意为 gradient boost decision tree。又叫MART（Multiple Additive Regression Tree)

#### 分类树和回归树

- 分类树：预测分类标签；C4.5；选择划分成两个分支后熵最大的feature；
- 回归树：预测实数值；回归树的结果是可以累加的；最小化均方差；

#### boosting

- gbdt的核心在于：每一棵树学的是之前所有树的结论和残差。每一步的残差计算其实变相地增大了分错instance的权重，而已经分对的instance则都趋向于0。
- Adaboost：是另一种boost方法，它按分类对错，分配不同的weight，计算cost function时使用这些weight，从而让“错分的样本权重越来越大，使它们更被重视”。
- Bootstrap也有类似思想，它在每一步迭代时不改变模型本身，也不计算残差，而是从N个instance训练集中按一定概率重新抽取N个instance出来（单个instance可以被重复sample），对着这N个新的instance再训练一轮。由于数据集变了迭代模型训练结果也不一样，而一个instance被前面分错的越厉害，它的概率就被设的越高，这样就能同样达到逐步关注被分错的instance，逐步完善的效果。

#### Shrinkage

Shrinkage（缩减）的思想认为，每次走一小步逐渐逼近结果的效果，要比每次迈一大步很快逼近结果的方式更容易避免过拟合。即它不完全信任每一个棵残差树，它认为每棵树只学到了真理的一小部分，累加的时候只累加一小部分，通过多学几棵树弥补不足。


