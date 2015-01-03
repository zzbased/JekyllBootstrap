---
layout: post
title: "semantic analysis new method"
description: ""
category: 
tags: [machine learning]
---
{% include JB/setup %}

## 语义分析新玩法

### 新方法

#### NBSVM
[Baselines and bigrams: simple, good sentiment and topic classification]()

我的理解是：把每个抽取得到的feature按照log-count ratio计算了一个weight，再送给标准的svm去做训练。

[Python collections模块实例讲解](http://www.jb51.net/article/48771.htm)

#### RNNLM
[rnnlm lib](http://www.fit.vutbr.cz/~imikolov/rnnlm/)
目前最先进的language model。
cache models (that describe long context information) or class-based models (that improve parameter estimation for short contexts by sharing parameters between similar words)

[Mikolov 的 RNNLM](http://licstar.net/archives/328#s24)，最好的参考资料：博士论文《Statistical Language Models based on Neural Networks》

循环神经网络的最大优势在于，可以真正充分地利用所有上文信息来预测下一个词，而不像前面的其它工作那样，只能开一个 n 个词的窗口，只用前 n 个词来预测下一个词。在训练rnnlm的同时，也可以得到派生物词向量。

word2vec也是mikolov的作品，与rnnlm相比目的不太一样，rnnlm主要是想做语言模型；word2vec的本身目的就是训练词向量，速度会更快一些。

rnnlm是怎么用到二分类上去的? 针对不同的label，训练两个不同的语言模型p+(x|y=+1)和p-(x|y=-1)。对于一个testcase x，求解r= p+(x|y=+1)/p-(x|y=-1)*p(y=+1)/p(y=-1)，如果r>1，则x属于label(+1)，否则x属于label(-1)。

其预测脚本为：
	./rnnlm -rnnlm model-pos -test test-id.txt -debug 0 -nbest > model-pos-score	./rnnlm -rnnlm model-neg -test test-id.txt -debug 0 -nbest > model-neg-score	paste model-pos-score model-neg-score | awk '{print $1 " " $2 " " $1/$2;}' > ../scores/RNNLM-TEST

![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/rnnlm_image.png)
![](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/rnnlm_formula.png)

#### sentence vector
句向量是通过词向量演化出来的。具体请参考论文[Distributed representations of sentences and documents]()

常见做法是：先利用word2vec训练出每个句子的embedding表示，然后把embedding作为特征送入liblinear。

#### emsemble
ensemble的方法有很多，线性ensemble，指数ensemble。

### 参考论文 
- [zero-shot leanring by convex combination of semantic embeddings]()
- [distributed representations of sentences and documents]()

