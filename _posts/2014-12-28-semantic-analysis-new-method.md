---
layout: post
title: "semantic analysis new method"
description: ""
category: 
tags: [machine learning]
---
{% include JB/setup %}

## 语义分析方法

语义分析，这里指运用各种机器学习方法，挖掘与学习文本、图片等的深层次概念。wikipedia上的解释：[In machine learning, semantic analysis of a corpus is the task of building structures that approximate concepts from a large set of documents.](http://en.wikipedia.org/wiki/Semantic_analysis_(machine_learning))。

工作这几年，陆陆续续做过一些项目，其实有些是：文本语义分析，图片语义理解，语义索引，短串语义关联，用户广告语义匹配... 感觉只要沾上点"语义"，就摇身一变顿时高大上了，有木有？可是真的有高大上吗？其实不然了。接下来我将聊一聊我所认识的语义分析，虽说接触也许并不深入，不过权当自己的总结好了。

下文主要由以下两部分组成：传统语义分析方法，基于深度学习的语义分析。

先把nlp的任务过一遍，顺便讲一下基本的方法。
这些任务包括下面列到的：切词，语言模型，termweighting

### 文本基本处理
#### 分词
拿到一段文本后，通常情况下，首先要做分词。分词的方法一般有如下几种：

- 基于字符串匹配的分词方法。此方法按照不同的扫描方式，逐个查找词库进行分词。根据扫描方式可细分为：正向最大匹配，反向最大匹配，最小切分，双向最大匹配。
- 全切分方法。它首先切分出与词表匹配的所有可能的词，再运用统计语言模型决定最优的切分结果。它的优点在于可以发现所有的切分歧义并且容易将新词提取出来。
- 基于知识理解的分词方法。该方法主要基于句法、语法分析，并结合语义分析，通过对上下文内容所提供信息的分析对词进行定界。

[1](http://blog.sina.com.cn/s/blog_7eb42b5a0100vf8l.html)
[2](http://blog.sina.com.cn/s/blog_6876a34b0100uq49.html)
[3](http://www.cnblogs.com/flish/archive/2011/08/08/2131031.html)

一般而言，方法一，二用得比较多，方法二相对更加准确，但在工业界方法一可能用得更多。

#### 语言模型
前面在讲全切分分词方法时，提到了语言模型。所以先把语言模型简单阐述一下。

fandy的统计语言模型。

rnn - 做语言模型。

#### Term Weighting
对文本分词后，接下来需要对分词后的每个term计算一个权重，重要的term应该给与更高的权重。举例来说，"什么产品对减肥帮助最大？"的term weighting结果可能是: "什么 0.1，产品 0.5，对 0.1，减肥 0.8，帮助 0.3，最大 0.2"。

- 最常见的方法是Tf-idf。tf-idf

- 利用机器学习方法来预测weight。

#### 命名实体识别

#### 短串成分分析

### 文本语义分析

#### 文本分类

#### 文本topic分析

#### 词向量，句向量


### 图片语义分析

#### 图片分类

#### 图片topic分析

#### image2text，image2sentence





### 语义分析的任务
我的理解里，文本语义分析主要包括以下任务：
- 切词/Term weighting
- 命名实体识别/新词发现
- 短串成分分析
- 文本翻译
- 文本分类
- 文本topic分析

图片语义分析主要包括以下任务：
- 图片分类
- 图片topic分析

- Sequence labelingPOS tagging & Name Entity Recognition [Turian et al., 2010, Collobert et al., 2011, Wang and Manning, 2013, Ma et al., 2014, Tsuboi, 2014, Guo et al., 2014, Qi et al., 2014]Word Segmentation [Zheng et al., 2013, Pei et al., 2014] 

- Syntax & MorphologyDependency Parsing [Stenetorp, 2013, Chen et al., 2014a, Levy and Goldberg, 2014, Bansal et al., 2014, Chen and Manning, 2014, Le and Zuidema, 2014]Constituency Parsing [Billingsley and Curran, 2012, Socher et al., 2013a, Andreas and Klein, 2014]CCG [Hermann and Blunsom, 2013], Selectional Preference [Van de Cruys, 2014], Morphology [Luong et al., 2013] 

- SemanticsWord Representations [Tsubaki et al., 2013, Srivastava et al., 2013, Rockt ̈aschel et al., 2014, Baroni et al., 2014, Hashimoto et al., 2014, Pennington et al., 2014, Neelakantan et al., 2014, Chen et al., 2014b, Milajevs et al., 2014]Semantic Role Labeling: [Hermann et al., 2014, Roth and Woodsend, 2014] Paraphrase [Socher et al., 2011]Grounding/Multi-modal [Fyshe et al., 2014, Kiela and Bottou, 2014]- Discourse[Ji and Eisenstein, 2014, Li et al., 2014a]- Question Answering, Knowledge Bases, & Relation Extraction[Hashimoto et al., 2013, Fu et al., 2014, Chang et al., 2014, Yih et al., 2014, Bordes et al., 2014, Iyyer et al., 2014, Yang et al., 2014, Gardner et al., 2014]- Sentiment Analysis[Glorot et al., 2011, Socher et al., 2013b, Irsoy and Cardie, 2014]Summarization[Liu et al., 2012]- Novel ApplicationsPoetry [Zhang and Lapata, 2014], Interestingness [Gao et al., 2014b], Hashtags [Weston et al., 2014]

What we’ll do: Summarize 2 ways Deep/Neural ideas can be used1 As non-linear classifier2 As distributed representation

- Exploiting Non-linear ClassifiersIt’s possible to directly apply Deep Learning to text problems with little modification, as evidenced by [Glorot et al., 2011]But sometimes NLP-specific modifications are needed, e.g. training objective mismatch in Machine Translation N-best experiment- Exploiting Distributed RepresentationDistributed Representation is a simple way to improve robustness of NLP, but it’s not the only way (POS tagging experiment)Promising direction: distributed representations beyond words, considering e.g. compositionality [Socher et al., 2013a]


ppt: deep learning for natural language processing and machine translation：

Language Models (LM) using Neural Nets。见72页。

Recurrent Neural Net Language Models [Mikolov et al., 2010]。见75页。

P99。参考文献。

### 传统文本语义分析
本小节主要内容：term weighting，短串匹配，topic model，分类。

【Query意图分析：记一次完整的机器学习过程（scikit learn library学习笔记）】http://t.cn/RvUNAsG 博客园zero_learner的博文。利用Python机器学习包Scikit Learn具体解决Query意图问题，是一个机器学习实践的很好示例。另外作者推荐阅读相关文章“如何选择机器学习分类器”：http://t.cn/RvA6amn

@刘知远THU:略短啊，相对来讲个人感觉北大赵鑫 @BatmanFly 对LDA的总结更全面系统一些：http://t.cn/aBmeRA
http://net.pku.edu.cn/~zhaoxin/Topic-model-xin-zhao-wayne.pdf

@Wenpeng_Yin 推荐一个NLP工具: http://t.cn/8s0E6w6 中文,英文,西班牙文, 法语,德语的lemmatizer, POS tagging, denpendency parsing以及Morphologic Tagging一条龙服务, 简单易用,效果不输stanford

转//@徐君_: SMIR 2014的邀请报告云集了IR和NLP的大牛，UMass的Bruce Croft教授、微软研究院的Jianfeng Gao研究员、CMU的Ed Hovy教授、马里兰大学的Doug Oard教授和阿姆斯特丹大学的Maarten de Rijek教授将与大家探讨信息检索和自然语言处理中的语义匹配，敬请期待！
@徐君_ 语义匹配(Semantic Matching)是信息检索与自然语言处理的核心问题之一，欢迎大家关注与投稿SIGIR 2014 Workshop on Semantic Matching in Information Retrieval (SMIR 2014) http://t.cn/8sVnfFi 。时间：7月11日，地点：澳大利亚黄金海岸，投稿截止日期：5月10日。 @李航博士

### 传统图片语义分析
本小节主要内容：图片topic model。

### 基于深度学习的文本语义分析
本小节主要内容：rnnlm，word embeddings。
word2vec，sentence-vector。glove。

word embeddings

ML = Representation + Objective + Optimization
Good Representation is Essential for Good Machine Learning

Semantic Hierarchy Extraction
Cross-lingual Joint Representation
Visual-Text Joint Representation


- Replicated Softmax: an Undirected Topic Model (NIPS 2010)- A Deep Architecture for Matching Short Texts (NIPS 2013)- Modeling Documents with a Deep Boltzmann Machine (UAI 2013)
- A Convolutional Neural Network for Modelling Sentences(ACL 2014)

Distributed representation can be used•  as pre-training of deep learning•  to build features of machine learning tasks•  as a unified model to integrate heterogeneous information (text, image, ...)



放弃词向量(WordEmbedding)工作，直接在词袋模型(BoW)上用卷积神经网络做文本分类任务(CNN for Text Categorization)，看这篇文章的工作（待评论。。），http://t.cn/RZbN2b6 提出seq-CNN与bow-CNN两种模型直接在BoW上使用卷积层，同state-of-the-art方法相比取得较好结果

继续来广告一把和@BB-Rev 的一个用deep learning做文本推荐的工作：http://t.cn/RhJKetG 我们在这个工作里把dl和矩阵分解结合，目标不再是之前dl的分类或回归。基本思想是利用dl去学习文本(item)上的feature，mf去学习user之间的偏好，两者通过gradient交互。结果比KDD11 Blei的best paper好了一个量级

开源Recurrent Neural Network库RNNSharp. 支持标准RNN和RNN-CRF，可用于分类，序列标注等任务。支持模板特征，上下文特征，word embedding以及运行时特征。地址：http://t.cn/RvFJHPj 求推广 @52nlp

lda结构是word-hidden topic。类lda结构假设在topic下产生每个word是条件独立而且参数相同。这种假设导致参数更匹配长文而非短文。这篇文章提出word-hidden topic-hidden word，其实是(word,hidden word)-hidden topic。增加的hidden word平衡了参数对短文的适配，在分类文章数量的度量上更好很自然。
@王威廉 经过不懈的努力，Geoff Hinton及其弟子终于用Deep Boltzmann Machine捣鼓出了类似LDA的隐变量文本模型，号称其抽取的特征在文本检索与文本分类上的结果比LDA好。UAI2013论文：http://t.cn/zQbzwVi


@张俊林say 最近几个月比较关注深度学习在自然语言处理方面的应用，虽然总体而言DL在NLP并未像图像和语音识别领域一样展现出突破性的进展，但是我个人比较看好这个方向的发展前景，把目前主流的相关工作梳理了一下，后续会不断维护版本更新以及我们在这方面的探索结果。http://t.cn/Rz4Jkia

@王威廉 对命名实体识别与消歧(Named Entity Recognition/Disambiguation)与词义消歧(Word Sense Disambiguation)有兴趣的同学注意了：ACL主席Gertjan van Noord等欧洲NLP科学家近日编撰了一篇非常棒的报告，汇总了大量的实体识别与消歧，词义消歧数据集，工具集。强烈推荐！http://t.cn/RvzajR2

Yoav Goldberg写了个测评文档，大致结论就是GloVe和word2vec如果正常比的话 效果差不多，没有宣称的11%这么大。。 链接：http://t.cn/RP0gMXB
@刘知远THU
斯坦福Richard Socher在EMNLP2014发表新作：GloVe: Global Vectors for Word Representation 粗看是融合LSA等算法的想法，利用global word co-occurrence信息提升word vector学习效果，很有意思，在word analogy task上准确率比word2vec提升了11%。 http://t.cn/RPohHyc


### 基于深度学习的图片语义分析
本节主要内容：cnn分类，cnn-sentences。

ImageNet的创始人、大规模图像分类竞赛ILSVRC的组织者、斯坦福CV组的Li Fei-Fei等人写了一篇比较详尽的ImageNet综述文章，总结了2010-2014年的成绩: http://t.cn/RhyGbGi


@沈复民 CVPR'13 best paper由google的论文fast detection of 100,000 categories on a single machine取得! 这个结果大家都猜到了！ 我在:http://t.cn/zHDS9R6

@陈博兴-NLP 最新的基于深度学习的机器翻译技术用RNN将源语言句子转换成一个vector representation，然后用另外一个RNN生成目标语言句子。Google借用这个idea，把第一个RNN换成Deep CNN，用来将图像翻译成文字，取得了很好的BLEU score。看看图片中例子，简直太赞了！http://t.cn/Rz7INhu

@王威廉 Google大神Jeff Dean月初在#CIKM2014#上关于谷歌深度学习的幻灯片，包括语音、NLP、视觉、MT以及多模态学习的最新进展: http://t.cn/RzPjoaD

@Jay_GraphLab 深度学习很强大，但训练起来非常昂贵，费时费力。然而并非每个问题都必须从头开始训练一个深度模型。这篇博客 http://t.cn/RzFAahl 详细介绍了为何以及如何使用深度学习做特征提取，并应用到简单机器视觉问题。在imagenet上训练好的模型，通过特征提取，应用到CIFAR10，可轻松达到94％以上测试准确率。

@丕子 Translating Videos to Natural Language http://t.cn/RzrJYnL 扫了一眼，做的还是image的，几乎没有video的信息嘛

@星空下的巫师 VGG已经放出他们ImageNet 2014模型的细节了， Very Deep Convolutional Networks for Large-Scale Image Recognition， see：http://t.cn/RhbZqQr

Jeff Hinton组把deep CNN(CovNets)在ImageNet上train好的模型放到网上了，试了下classification, retrieval, image2text的在线demo, amazing! http://t.cn/Rvs0Pvj 最重要的是他们的source code以及installation & documentation 也一并公布，超过Rob Fergus学生的Clarifai http://t.cn/8kL993u

斯坦福大学计算机系的大作业，利用深度学习技术（CNN、RNN等）做深度的图像分析，自动描述图片，图片物体识别等等。详细技术报告在这里 http://t.cn/RzLtOfp Project地址：http://t.cn/Rz7F0Fj

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

- [sequence to sequence learning with neural network]()
- [exploting similarities among language for machine translation]()


DataScientist   2014-11-10 13:30
Fast Randomized SVD http://t.cn/R71pgaC

https://github.com/memect/hao/blob/master/awesome/chinese-word-similarity.md