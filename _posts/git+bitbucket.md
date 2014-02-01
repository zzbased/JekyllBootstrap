##git + bitbucket 配置##

bitbuckte 相比于时下最流行的 github, 其最牛之处在于可以免费创建私人项目。这点相当吸引人。我打算把一些杂乱的项目放上去，一方面为个备份，另一方面也为跟踪开发过程。

mac电脑上默认有安装git, 需要做的只是一些简单的配置。

1. 首先申请一个bitbucket账号。创建一个repository。
2. 接着参考这篇文章，https://confluence.atlassian.com/display/BITBUCKET/Set+up+SSH+for+Git
3. ssh-keygen 生成rsa的私钥和公钥。
4. 创建ssh config.
vi ~/.ssh/config。在里面输入下面内容：
Host bitbucket.org
 IdentityFile ~/.ssh/id_rsa
5. 然后把public key设置到bitbucket里。
    设置成功后，执行“ssh -T git@bitbucket.org”测试是否OK。
6. 根据bitbucket的提示，在本地command shell里输入：
mkdir /path/to/your/project
cd /path/to/your/project
git init
git remote add origin ssh://git@bitbucket.org/zzbased/documents.git
echo "# This is my README" >> README.md
git add README.md
git commit -m "First commit. Adding a README."
git push -u origin master

7. over!