---
layout: post
title: "matrix similarity"
description: ""
category: 
tags: []
---
{% include JB/setup %}

### 矩阵相似度计算

- 在机器学习任务中，矩阵是一个很重要的表述。文档与词，用户与其购买的商品，用户与好友关系等都可以描述成一个矩阵。对这个矩阵，一个最基础的任务就是找到最相似的用户，最相似的文档，也就是k最近邻问题。

- 方法一是：minhash + lsh。参考[链接1](http://en.wikipedia.org/wiki/MinHash)，[链接2](http://blog.csdn.net/sunlylorn/article/details/7835411)

>我们便可以根据MinHash来计算两个集合的相似度了。一般有两种方法：
        
>第一种：使用多个hash函数。
为了计算集合A、B具有最小哈希值的概率，我们可以选择一定数量的hash函数，比如K个。然后用这K个hash函数分别对集合A、B求哈希值，对
每个集合都得到K个最小值。比如Min(A)k={a1,a2,...,ak}，Min(B)k={b1,b2,...,bk}。
那么，集合A、B的相似度为|Min(A)k ∩ Min(B)k| / |Min(A)k  ∪  Min(B)k|，及Min(A)k和Min(B)k中相同元素个数与总的元素个数的比例。

>第二种：使用单个hash函数。
第一种方法有一个很明显的缺陷，那就是计算复杂度高。使用单个hash函数是怎么解决这个问题的呢？请看：
前面我们定义过 hmin(S)为集合S中具有最小哈希值的一个元素，那么我们也可以定义hmink(S)为集合S中具有最小哈希值的K个元素。这样一来，
我们就只需要对每个集合求一次哈希，然后取最小的K个元素。计算两个集合A、B的相似度，就是集合A中最小的K个元素与集合B中最小的K个元素
的交集个数与并集个数的比例。

>对于每个doc，利用minhash计算后，则将其降维至K维向量。然后就该LSH出场了。

>lsh:local sensitive hash。将上面K维向量划分到n个桶，每个桶有K/n维。两个doc，只要有一个桶的元素是一样的，那么就认为他们是相似候选。这里有一个公式来衡量n的选值。请参考论文[find similar items]()

- 方法二是：通过map-reduce直接计算集合之间的相似性。
>假设矩阵U\*I，计算两两U之间的相关性，直接计算的话，即使是用map-reduce，实现复杂度也很高。
>这里讲述一种map-reduce实现，第一轮map-reduce过程，将U\*I矩阵转置为I\*U矩阵，输出每个item下，任意两个user之间在此item维度下的weight乘积。第二轮map-reduce，将两个user之间的第一轮得分聚合起来，就得到了这两个user之间的cosine相似度。具体请参考链接[大规模矩阵相似度计算]()

