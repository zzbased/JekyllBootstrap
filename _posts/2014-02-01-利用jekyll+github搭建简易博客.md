---
layout: post
title: "利用jekyll+github搭建简易博客"
description: ""
category: 
tags: [blog, git]
---
{% include JB/setup %}

##利用jekyll+github搭建简易博客##

### 准备工作 ###
1. Github工作目录
搭建Github工作目录，需要先把ssh通道建立好，参看下面两篇文章。[产生ssh keys](https://help.github.com/articles/generating-ssh-keys)    [可能碰到的问题](https://help.github.com/articles/error-permission-denied-publickey)

2. markdown编辑器
在macbook上，我使用的编辑器是lightpaper. 引用图像存储链接服务是 [droplr](droplr.com)

### 步骤 ###
1. 我使用的是[jekyllbootstrap](http://jekyllbootstrap.com)。号称三分钟可以教会搭建github博客，事实就是如此。参考这篇入门指南即可。[入门指南](http://jekyllbootstrap.com/usage/jekyll-quick-start.html)

2. 需要注意的是，如果在上面准备工作里github的ssh设置没能成功。
	git remote set-url origin git@github.com:zzbased/zzbased.github.com.git
	可以更改为https地址:
	git remote set-url origin https://github.com/zzbased/zzbased.github.com.git
	
3. 安装好jekyll后，就可以本地调试。我们利用index.md，可以在原基础上做修改即可。

4. 然后在_post文件夹里，删除原来的example。利用rake post title="xxx"新增一个md文件。接下来就开始编辑了。

5. 如果不喜欢页面最下面的footer, 可以在“./_includes/themes/twitter/default.html”文件中，把footer屏蔽掉。不过建议还是留着，可以让更多的人接触到这项工具。

