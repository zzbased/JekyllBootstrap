---
layout: post
title: "推荐算法实战"
description: ""
category: 
tags: [machine learning]
---
{% include JB/setup %}

# 推荐算法实战

### 重要的参考资料

数据挖掘技术在推荐系统的应用;陈运文 

[陈运文: CIKM 数据挖掘竞赛夺冠算法介绍](http://www.52cs.org/?p=369)

[Tutorial: Recommender Systems;Dietmar Jannach](http://ijcai13.org/files/tutorial_slides/td3.pdf)

Google News Personalization: Scalable Online Collaborative Filtering。大规模cf的实现。

[Application of Dimensionality Reduction in Recommender System -- A Case Study](http://ai.stanford.edu/~ronnyk/WEBKDD2000/papers/sarwar.pdf)

[Up Next: Retrieval Methods for Large Scale Related Video Suggestion](http://vdisk.weibo.com/s/DaKXoKQC5TSH)

[推荐系统的常用算法对比](http://dataunion.org/bbs/forum.php?mod=viewthread&tid=835&extra=)

[学生强则国强，访天猫推荐算法大赛Top 9团队](http://www.csdn.net/article/2014-08-27/2821403-the-top-9-of-ali-bigdata-competition/9)

[alex-recommendation](http://alex.smola.org/teaching/berkeley2012/slides/8_Recommender.pdf)

[美团推荐算法实践](http://tech.meituan.com/mt-recommend-practice.html)

[Large scale recommendation in e-commerce -- qiang yan](http://www.slideshare.net/scmyyan/large-scale-recommendation-in-ecommerce-qiang-yan)。
//@严强Justin:回复@李沐M: http://t.cn/R7vCJGv //@李沐M:Share下slides？ //@严强Justin:online match + online learning works very well //@吴竑:@严强Justin 赞processing stack。我们最近在豆瓣fm上也做了online learning，improve10个点左右

[幻灯]《Recommender Systems: Super Overview》http://t.cn/R7WtFwY 来自Netflix的Xavier Amatriain在Summer School 2014 @ CMU上长达4小时的报告，共248页，是对推荐系统发展的一次全面综述，其中还包括Netflix在个性化推荐方面的一些经验介绍，强烈推荐! 云盘:http://t.cn/RZuLoSS

推荐点关于推荐系统的综述么? 答: 问答207 http://t.cn/RhCt7lc 强推KDD2014讲义 "the recommender problem revisited": 第一部分Xavier Amatriain的综述(135页, 2014机器学习夏季学校版有248页), 第二部分"Context Aware Recommendation" (64页) 谢 @小飞鱼_露 @明风Andy 推荐

翻了翻推荐系统的tutorial slides，目前发现的最好的两个：一是alex前年在berkeley上课用的，简练，清晰，重点都覆盖到了 http://t.cn/R7WtFwj 二是xavier在今年cmu夏季课程用的，4小时时长，很全面。http://t.cn/R7WtFwY @phunter_lau 你怎么看？

//@陈阿荣: 先收着吧 @我的印象笔记 //@永夜: 马
@丕子 遇到从公司里面出来的推荐经验，一定得好好看看。 (评论给 @美团技术团队 美团推荐算法实践 - 美团技术团队 http://t.cn/RZYdtMn )

如果还不熟悉Xavier，请看主页http://t.cn/RvZmne2 他刚刚作为VP Engineering加入Quora。也请参他在KDD 2014上的报告 http://t.cn/RZMK3gY 和cmu夏季课程上的4小时教程 http://t.cn/R7WtFwY
@小飞鱼_露 Xavier Amatriain 在 Quora 上关于目前推荐系统研究总结，涵盖了推荐系统的多样性，基于上下文环境推荐，社交信息的引入，评分预测已经不是主流，LTR的应用会更符合推荐的初衷等 http://t.cn/RZMx6Gu @好东西传送门
mlss2014- Recommender Systems.pdf 请参考微云中的文件。


### 推荐系统的资料分享
[link](http://blog.sina.com.cn/s/blog_804abfa70101btrv.html)

这个资料分享主要分享的都是非学术的Paper，都来自商业公司，Google, YouTube, Amazon, LinkedIn等等。
我个人非常喜欢这些文章，基本上，这些文章描述的都是在系统中的实际能工作的东西。

这个是Google的一篇论文http://t.cn/zl0zxPZ这个里面有很多有意思的想法。
推荐的结果是三个算法的融合，即MinHash, PLSI, covisitation. 
融合的方式是分数线性加权
一个主要的思想是“online”的进行更新，所以这个地方一定要减少规模，索引使用了User Clustering的算法，包括Min Hash和PLSI。
在新数据来的时候，关键是不要去更新User Cluster，而是直接更新所属的Cluster对于URL的点击数据
对于新用户，使用covisitation的方法进行推荐

这个是上一篇Paper的进阶paper。 http://t.cn/zl0zqDO
这篇Paper在上一篇的基础上增加了一些内容，主要包括Topic部分的内容，Google News是有Topic信息的。
这篇Paper通过用户喜欢的Topic这个信息以及Topic Trend这个信息一起进行分析。
热门的topic会被更多的展现给用户，其中用户只会看到他喜欢的Topic
这个方法和上面的方法相比，可能对于解决热门News的问题，有更大的帮助

这个是Youtube的文章 vdisk.weibo.com/s/fcbuu
这篇Paper的的方法更直观，它只使用了covisitation的信息，但是对于covisitation的方法做了N次扩展，即找一个Seed的多次邻居。
在这个的基础上，做了一些后处理的工作，尤其是Diversity的工作

 这个是对于Amazon商品推荐算法的一个Paper的翻译版
http://blog.sina.com.cn/s/blog_586631940100pduh.html
  这个Paper比较老了，但是是item-Based推荐的经典文章了。
这个是IBM的两位同学对于推荐的一个综述，属于入门级的，看看也不错。 
  http://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy1/index.html
这个比较有营养，是高级货，是LinkedIn的兄弟们在KDD2012上发布的，有用！进阶以后值得看看，尤其是搞真系统的。http://t.cn/zl0ZTN1
这个更是高级货了，Recommendations as a Conversation with the User 
这个的角度更多的是推荐系统的HCI设计，前面是一堆哲学，看不懂可以跳过，后面的例子还是比较给力的。有几个数字很给力：
Amazon: 35% of sales result from recommendations 
75% of Netflix views result from recommendations



