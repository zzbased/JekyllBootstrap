---
layout: post
title: "git_operator"
description: ""
category: 
tags: [git]
---
{% include JB/setup %}

## git相关操作指南

###最基础的git操作
1. 创建新仓库。
> 创建新文件夹，打开，然后执行 git init 以创建新的 git 仓库。

2. 检出仓库。
> 执行如下命令以创建一个本地仓库的克隆版本：
git clone /path/to/repository 
如果是远端服务器上的仓库，你的命令会是这个样子：
git clone username@host:/path/to/repository

3. 工作流。
>你的本地仓库由 git 维护的三棵“树”组成。第一个是你的 工作目录，它持有实际文件；第二个是 缓存区（Index），它像个缓存区域，临时保存你的改动；最后是 HEAD，指向你最近一次提交后的结果。

4. 添加与提交。
>你可以计划改动（把它们添加到缓存区），使用如下命令：
git add <filename>
git add *
这是 git 基本工作流程的第一步；使用如下命令以实际提交改动：
git commit -m "代码提交信息"
现在，你的改动已经提交到了 HEAD，但是还没到你的远端仓库。

5. 推送改动。
> 你的改动现在已经在本地仓库的 HEAD 中了。执行如下命令以将这些改动提交到远端仓库：
git push origin master
可以把 master 换成你想要推送的任何分支。
如果你还没有克隆现有仓库，并欲将你的仓库连接到某个远程服务器，你可以使用如下命令添加：
git remote add origin <server>
如此你就能够将你的改动推送到所添加的服务器上去了。

6. 更新与合并。
> 要更新你的本地仓库至最新改动，执行：
git pull
以在你的工作目录中 获取（fetch） 并 合并（merge） 远端的改动。
要合并其他分支到你的当前分支（例如 master），执行：
git merge <branch>
两种情况下，git 都会尝试去自动合并改动。不幸的是，自动合并并非次次都能成功，并可能导致 冲突（conflicts）。 这时候就需要你修改这些文件来人肉合并这些 冲突（conflicts） 了。改完之后，你需要执行如下命令以将它们标记为合并成功：
git add <filename>
在合并改动之前，也可以使用如下命令查看：
git diff <source_branch> <target_branch>

7. 替换本地改动。
> 假如你做错事（自然，这是不可能的），你可以使用如下命令替换掉本地改动：
git checkout -- <filename>
此命令会使用 HEAD 中的最新内容替换掉你的工作目录中的文件。已添加到缓存区的改动，以及新文件，都不受影响。
假如你想要丢弃你所有的本地改动与提交，可以到服务器上获取最新的版本并将你本地主分支指向到它：
git fetch origin
git reset --hard origin/master

8. git与svn命令的区别。
> git pull -- svn up；
> git add -- svn add；
> git commit -- svn ci；
> git clone -- svn cp；
> git checkout -- svn co；
> git push -- 无；
> git status -- svn status；
> git revert -- svn revert；
> git diff --  svn diff；
> git merge -- svn merge；

### git 客户端
常见的git客户端有：[msysgit](https://msysgit.github.io/)，[TortoiseGit](https://code.google.com/p/tortoisegit/)

我常用的是msysgit。它自带一个git gui。下图中，缓存改动的命令为git add，提交的命令为git commit，上传的命令为git push。

![git_gui](https://raw.githubusercontent.com/zzbased/zzbased.github.com/master/_posts/images/git_gui.png)

在公司里用git，可能需要设置git proxy，命令如下面所示：
git config --global http.proxy https://web-proxyhk.oa.com:8080

git gui默认不保存用户名和密码，如果需要让它保存，可以设置credential.helper，
git config --global credential.helper "cache --timeout=3600"。具体请参考[skip-password](http://stackoverflow.com/questions/5343068/is-there-a-way-to-skip-password-typing-when-using-https-github)

### 其他参考链接

- [完整的git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
- [git使用简易指南](http://www.bootcss.com/p/git-guide/)
- [Git冲突：commit your changes or stash them before you can merge. 解决办法](http://www.letuknowit.com/post/144.html)
- [git与github的关联](http://blog.csdn.net/authorzhh/article/details/7533086)
