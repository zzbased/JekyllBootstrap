## mac电脑上eclipse设置 ##

1. 首先是下载最新的jdk. 下载安装后，并设置环境变量：
export JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/"
export PATH="$JAVA_HOME/bin:$PATH"
export CLASSPATH=".:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar"

2. 再下载eclipse，本来以为可以直接启动了。发现还不行，经搜索发现：
http://blog.csdn.net/qysh123/article/details/16930143
还要再下载一个java. 
http://support.apple.com/kb/DL1572?viewlocale=en_US&locale=en_US
暂时还不清楚这个和oracle官网的有啥不同的地方。