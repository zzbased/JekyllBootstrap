---
layout: post
title: "topic model"
description: ""
category: 
tags: [machine learning, topic model]
---
{% include JB/setup %}

## topic model 思考
topic model通常指的是lda, plsa等模型，但从广义上将，类目也是一种topic。本文主要把以前接触到的plsa, lda等知识回顾一下，顺便把类目型的topic model也做一个总结。

### 常见topic model

plsa

LDA:
采样算法：collapsed gibbs sampler，SparseLDA，[AliasLDA](http://www.sravi.org/pubs/fastlda-kdd2014.pdf)，Metropolis-Hastings sampler
[gibbs sampling for the uninitiated](www.umiacs.umd.edu/~resnik/pubs/LAMP-TR-153.pdf)

data-parallelism(splitting documents over machines) versus model-parallelism (splitting the word-topic distributions over machines).

data-parallelism:YahooLDA, Scaling distributed machine learning with the parameter server
	
model-parallelism:PLDA+, peacock

lightLDA:LightLDA adopts a different data-and-model-parallel strategy to maximize memory and CPU efficiency: we slice the word-topic distributions (the LDA model) in a structure-aware modelparallel manner [9, 24], and we fix blocks of documents to workers while transferring needed model parameters to them via a bounded-asynchronous data-parallel scheme [8]. 

Metropolis-Hastings sampler

**接下去的重点就是把最新的采样算法和怎么样做并行化好好研究一下，并写成文章**

### topic model on twitter 阅读笔记
- 这篇论文其实和前段时间做过的广告分类有些相似，区别在于难度要大一些，考虑的点要更丰富一些。文中提到的很多点我以前都或多或少实践过，接触过，不过没有这么系统的总结出来，所以把这篇论文细细研读一遍，对自己的知识回顾与整理也是有帮助的。

- 文中说到topic model，为啥没有直接用lda等模型。我自己总结，主要原因是lda模型可控性可解释性相对比较差：对于每个topic，不能用很明确的语义归纳出这个topic在讲什么；重新训练一遍lda模型，每个topic id所对应的语义可能发生了变化；有些topic的准确性比较好，有些比较差，而对于比较差的topic，没有特别好的针对性的方法去优化它；但lda等模型也有好处，无监督训练。

- 在我们以往的运用中，lda模型比较适合用于做某些机器学习任务的特征，譬如pctr，relevance等，而不适合作为一种独立的方法去解决某种特定的问题，例如触发，分类。Blei是这样评价lda的：it can easily be used as a module in more complicated models for more complicated goals.

- 有监督的topic model中，第一个任务就是确定topic taxonomy。通常情况里，taxonomy都是由人工创建的。人工创建的最大问题是，知识面有限，可能不能覆盖住所有情形。文中有提到一种方法：ODP和Freebase。我的理解是，可以先用某种无监督的聚类方法，将待分类的文本划分到某些clusters，然后人工review这些clusters，切分或者合并cluster，提炼topic name，再然后根据知识体系，建立层级的taxonomy。

#### 文本分类
- 过滤聊天类文本，也就是筛选出un-labeled data。文中有提到一种方法:

- 训练数据获取。这其实是一种常见的方法，记得以前在afs做分类器时，也有一种方法，先对每一个类人工筛选一些特征词，然后根据这些特征词对亿级文本网页分类，再然后对每一个明确属于该类的网页提取更多的特征词，加入原有的特征词词表，再去做分类；中间再辅以一定的人工校验，这种方法做下来，效果还是蛮不错的，更关键的是，如果发现那个类有badcase，可以人工根据badcase调整某个特征词的权重，简单粗暴又有效。还有一个任务，term weighting工作里，我们是利用logistic regression的方法来做的，其中面临的最主要任务就是训练数据的获取，这里我们采用了三种方法。
twitter利用了三种方法，user-level priors（发布tweet的用户属于的领域），entity-level priors（话题，类似于微博中的#***#），url-level priors（tweet中的url）。通过上述基于规则的方法获取到的训练数据是有较大噪声的，这时可以利用co-learning。
获取到正例样本后，还需要负例样本，按照常见的方法，从非正例样本里随机抽取作为负例的方法，效果并不是好，文中用到了pu-learning去获取高质量的负例样本。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/training_data_acquisition.png)

- 特征提取。为了考虑实时性，jubjub并没有用到很复杂的特征，主要用到了两种特征：binary hashed byte 4gram，hashed unigram frequency。

- model pretraing。
在正则化上，文中采用了ElasticNet regularization，也就是L1和L2的组合。模型上，选择了one-be-all LR，而不是Multinomial LR，主要考虑是：并行训练multiple topics model更复杂；不能重新训练 a subset of topics。

- relation regularization。因为是层级分类器，需要考虑label之间的relation。譬如Label expansion。

- model calibration。

- quality evaluation。

- model fine-tuning。

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/fine_tuning.png)

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/fine_tuning_formula.png)

