---
layout: post
title: "有价值的参考文献和业界分享整理"
description: ""
category: 
tags: [machine learning, nlp]
---
{% include JB/setup %}

把自己的论文库浏览了一遍，将一些我自己认为有价值的论文整理一下，做一点读书笔记，权作备忘。

### types of machine learning ###
Types of Machine Learning Algorithms; Taiwo Oladipupo Ayodele


### 最优化 ###
[lbfgs](http://www.chokkan.org/software/liblbfgs/)
一种拟牛顿法，经常用来做凸优化。任何凸函数优化，将cost fuction和gradient function写好了，输入进去，就可以求得最优值。
譬如对于svm模型，求svm的squared hinge loss:
 	yp=x\*w; idx=find(yp.\*y "<" 1); e=yp(idx)-y(idx); f=e'\*e+c\*w\*w; df=2(x(idx,:)'\*e+c\*w);
计算完cost function: f和gradient function: df, 输入到lbfgs即可。
余凯：说hinge loss, 当年主要卖点是得到稀疏解，也即所谓支撑向量 (SV). 今天看来，这个性质已不重要：1 这是一个特殊凸优化算法的结果，而非统计学习的本质问题； 2 实践中，当年kernel系统得到稀疏解带来计算上的便捷，但现在我们已不用kernel; 3 甚至稀疏性这个结论已被推翻，已证明SV是线性增长。

### classification ###
Supervised Machine Learning: A Review of Classification Techniques; S. B. Kotsiantis 

### crf ###
An Introduction to Conditional Random Fields;Charles Sutton。 crf相关的文章有很多，我是看这篇入门的。其实想深入学习，还是推荐看代码。[crtsuite](http://www.chokkan.org/software/crfsuite/)是一份不错的代码。

### SVM ###
A Gentle Introduction to Support Vector Machines in Biomedicine;Alexander Statnikov。 这个介绍很全面。

july的这篇文章写得也不错。[支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/v_july_v/article/details/7624837)


### topic model ###
Parameter estimation for text analysis; Gregor Heinrich。最佳入门材料。

A Note on the Expectation-Maximization (EM) Algorithm; ChengXiang Zhai。em算法最佳入门材料。

[概率语言模型及其变形系列](http://blog.csdn.net/yangliuy/article/details/8330640)  一个博客文章，写的挺不错的。

TOPIC MODELS;  DAVID M. BLEI , JOHN D. LAFFERTY

GIBBS SAMPLING FOR THE UNINITIATED; Philip Resnik。gibbs sampling最佳入门材料。

YahooLDA:An Architecture for Parallel Topic Models;Alexander Smola。开源实现yahoo_lda的论文。看完论文，还是建议看代码，跑demo。[code](https://github.com/sudar/Yahoo_LDA)

PLDA: Parallel Latent Dirichlet Allocation for Large-scale Applications; Yiwang。yiwang基于mpi和mapreduce的一个lda实现。后来yiwang又实现了一个更牛逼的系统:peacock。

### boosting ###
A Short Introduction to Boosting; Yoav Freund

The Boosting Approach to Machine Learning An Overview; Robert E. Schapire

### sas ###
A Web-based Kernel Function for Measuring the Similarity
of Short Text Snippets; Mehran Sahami , Timothy D. Heilman。


###deep learning###
[UFLDL](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)

[Deep Learning（深度学习）学习笔记整理系列](http://blog.csdn.net/zouxy09/article/details/8775360)

Learning Deep Architectures for AI; Yoshua Bengio

Deep Learning for NLP(without Magic)	; Richard Socher, Yoshua bengio

Representation Learning: A Review and New Perspectives; Yoshua Bengio  
Hierarchical Convolutional Deep Learning in Computer Vision; Matthew D. Zeiler。zeiler是imagenet 2013 image classification的第一名，这是他的博士论文，挺有阅读价值的。

### recommender ###
数据挖掘技术在推荐系统的应用;陈运文 

[Tutorial: Recommender Systems;Dietmar Jannach](http://ijcai13.org/files/tutorial_slides/td3.pdf)

Google News Personalization: Scalable Online Collaborative Filtering。大规模cf的实现。

###Online learning###
Online Learning and Online Convex Optimization; By Shai Shalev-Shwartz


### DSP ###
DSP中的算法初探; 江申 @ 力美移动广告 

实时定向广告技术-DSP框架及算法;则成

### ctr ###
Ad Click Prediction: a View from the Trenches;H. Brendan McMahan