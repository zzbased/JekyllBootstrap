---
layout: post
title: "HMM and CRF"
description: ""
category:
tags: [machine learning]
---
{% include JB/setup %}

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

##HMM与CRF


看到语音识别的时候，觉得是该找个机会把HMM与CRF相关的知识点做一个总结了。
之前看过很多这方面的文章，但都是零零碎碎的，没有形成知识体系。


### 推荐文章

首先推荐几篇文章：

[classical probabilistic model and conditional random field](http://www.scai.fraunhofer.de/fileadmin/images/bio/data_mining/paper/crf_klinger_tomanek.pdf)

[An Introduction to Conditional Random Fields for Relational Learning](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)

[隐马尔可夫模型 最大熵马尔可夫模型 条件随机场 区别和联系](http://1.guzili.sinaapp.com/?p=133#comment-151)

[52nlp hmm](http://www.52nlp.cn/tag/hmm)

[浅谈中文分词](http://www.isnowfy.com/introduction-to-chinese-segmentation/)
 
### 模型之间的联系
从下面两张图看各个模型之间的联系：

![crf_hmm1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_hmm1.png)

![crf_hmm2](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf_hmm2.png)

### 生成模型与判别模型

#### 生成模型，Generative Model

- 假设o是观察值，q是模型。如果对P(o|q)建模，就是Generative模型。
- 其基本思想是首先建立样本的概率密度模型，再利用模型进行推理预测。一般建立在统计力学和bayes理论的基础之上。
- 估计的是联合概率分布（joint probability distribution），p(class, context)=p(class|context)*p(context)。
- 代表：Gaussians，Naive Bayes，HMMs，Bayesian networks，Markov random fields

#### 判别模型，Discriminative Model

- 假设o是观察值，q是模型。如果对条件概率(后验概率) P(q|o)建模，就是Discrminative模型。
- 基本思想是有限样本条件下建立判别函数，不考虑样本的产生模型，直接研究预测模型。代表性理论为统计学习理论。
- 估计的是条件概率分布(conditional distribution)， p(class|context)。利用正负例和分类标签，focus在判别模型的边缘分布。目标函数直接对应于分类准确率。
- 代表：logistic regression，SVMs，neural networks，Conditional random fields(CRF)

### 隐马尔科夫模型

隐马尔科夫模型是由初始状态概率向量，状态转移概率矩阵，观测概率矩阵决定。

隐马尔科夫模型做了两个基本假设：

- 齐次马尔科夫性假设：假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一个时刻，与其他时刻的状态和观测无关。
- 观测独立性假设：假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其他观测及状态无关。

三个基本问题：

- 概率计算问题。给定模型和观测系列，计算在模型下观测系列出现的概率。
  前向-后向算法。
- 学习问题。已知观测系列，估计模型参数，使得该模型下观测系列概率最大。
  EM算法，Baum-Welch算法。
- 预测问题，也称解码问题。已知模型和观测系列，求对给定观测系列条件概率P(I|O)最大的状态系列。
  Viterbi算法。

为什么是生成模型？
$$P(O|\lambda)=\sum_I P(O|I,\lambda)P(I|\lambda)$$
从上面公式可以看出，这是生成模型。
而观测系列的生成，与LDA的生成过程类似。

### 条件随机域，CRF
- [CRF++学习](http://blog.csdn.net/gududanxing/article/details/10827085)
- [三种CRF实现在中文分词任务上的表现比较](https://jianqiangma.wordpress.com/2011/11/14/%E4%B8%89%E7%A7%8Dcrf%E5%AE%9E%E7%8E%B0%E7%9A%84%E7%AE%80%E5%8D%95%E6%AF%94%E8%BE%83/)
- [CRF++ library](http://crfpp.googlecode.com/svn/trunk/doc/index.html?source=navbar)

### 对比
![hmm1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/hmm1.png)

上图是HMM的概率图，属生成模型。以P(Y,X)建模，即P(O，q) = P(q)P(O|q) 建模。

![crf1](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/crf1.png)

上图是CRF的概率图，属判别模型。以P(Y|X)建模。

### 参考文献
- [Markdown中插入数学公式的方法](http://blog.csdn.net/xiahouzuoxin/article/details/26478179)
- [LaTeX/数学公式](http://zh.wikibooks.org/zh-cn/LaTeX/%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F)
- [LaTeX数学公式输入初级](http://blog.sina.com.cn/s/blog_5e16f1770100fs38.html)