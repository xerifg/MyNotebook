# 笔记初衷

我会在这里将自己学习过程中的知识以笔记的形式在这里记录下来，会包含算法、数据结构、编程语言、常用工具的使用，以及经常会遇到的各种问题，还会包含平时自己的一些思考，或许还会根据需要再添加其他类的笔记，大家有好的想法的欢迎进行补充，这个笔记的初衷就是--共同进步。

# 开始之前

[学习如何学习](https://www.coursera.org/learn/learning-how-to-learn)

学习新知识很重要。但是，复习知识更重要，否则，时间一长你之前学习知识就会忘记，相当于你之前花的时间浪费掉了，这里有一个用于复习的抽认卡软件推荐给大家（[Anki](https://apps.ankiweb.net/)），类似于一些英语学习的软件学习单词的方式，让你复习你之前学过的各种知识。

# 数据结构

* #### 算法复杂度/Big-O/渐进分析法

  [渐进表示（视频）](https://www.youtube.com/watch?v=iOq5kSKqeR4)

  [Big-O记号  (视频)](https://www.youtube.com/watch?v=ei-A_wy5Yxw&list=PL1BaGV1cIH4UhkL8a9bJGG356covJ76qN&index=3)

  ## i  一级知识

  * #### 数组（Arrays）

  * #### 链表（Linked Lists）

  * #### 堆栈（Stack）

  * #### 队列（Queue）

  * #### 哈希表（Hash table）

  ## ii 二级知识

  * #### 二分查找
  
  * #### 树

# 经典算法

* [卡尔曼滤波](https://zhuanlan.zhihu.com/p/45238681)

* [马尔可夫链](https://www.bilibili.com/video/BV19b4y127oZ?from=search&seid=11829849833797645360&spm_id_from=333.337.0.0)

* [PCA降维](https://zhuanlan.zhihu.com/p/32412043)

  主要是对原始数据的协方差矩阵进行特征分解，提取前k个特征向量，则由这些特征向量组成的矩阵，就是将原数据降维到k维的转换矩阵

# 神经网络

神经网络可以分为两大类，一类是类似于卷积神经网络的多层神经网络，其中，BP网络是始祖；

另一类是类似于深度信念网络的相互连接型网络，其中，Hopfied网络是始祖

* 特征工程
  * 编码方式
    * 独热编码（one hot encoding）
    * [Embedding](https://blog.csdn.net/weixin_44493841/article/details/95341407)
* 损失(代价)函数
  1. 最小二乘误差函数（二次代价函数）
  2. [交叉熵代价函数](https://blog.csdn.net/u014313009/article/details/51043064)
  3. [Triplet loss](https://blog.csdn.net/u013082989/article/details/83537370)


* 激活函数
  1. Sigmoid函数
  2. tanh函数
  3. ReLU函数

* 似然函数
  1. Softmax函数

* 学习率
  1. AdaDrad
  2. AdaDelta
  3. 动量方法

## 1. 卷积神经网路

* [BP神经网络上](https://blog.csdn.net/weixin_42398658/article/details/83859131)、[BP神经网络中](https://blog.csdn.net/weixin_42398658/article/details/83929474)、[BP神经网络下](https://blog.csdn.net/weixin_42398658/article/details/83958133)
* [卷积网络的动态可视化](https://www.bilibili.com/video/BV1AJ411Q72b?p=3&spm_id_from=pageDriver)
* 卷积神经网络保持平移、缩放、变形不变性的原因：
  1. 局部感受野
  2. 权值共享
  3. 下采样（池化）：减少参数，防止过拟合

####   * 参数的设定

* 与神经网络相关的（排名越靠前，对结果影响越大）
  1. 网络的层结构
  2. 卷积层的卷积核的大小，个数
  3. 激活函数的种类
  4. 有无预处理
  5. Dropout的概率
  6. 池化的方法
  7. 全连接层的个数
  8. 有无归一化

* 与训练相关的参数
  1. Mimi-Batch的大小
  2. 学习率
  3. 迭代次数
  4. 有无预训练

## 2. 深度信念网络(DBN)

* Hopfield神经网络([学习资料(上)](https://blog.csdn.net/weixin_42398658/article/details/83991773)、[学习资料(下)](https://blog.csdn.net/weixin_42398658/article/details/84027012))
* 波尔兹曼机
* [受限波尔兹曼机](https://zhuanlan.zhihu.com/p/22794772)
* 深度信念网络

# 强化学习

* [基本概念(human-level control through deep reinforcement learning)](https://github.com/xerifg/MyNotebook/blob/main/materials/dqn-atari.pdf)

# 数据集

* [步态数据集](https://raw.githubusercontent.com/xerifg/MyNotebook/main/picture/%E6%AD%A5%E6%80%81%E6%95%B0%E6%8D%AE%E9%9B%86.bmp)

# 信号处理

* [自相关与互相关](https://zhuanlan.zhihu.com/p/77072803)

  自相关的特点，原信号的自相关信号，虽幅值改变，但保留了原信号的频率特征。常用应用：从杂乱的信号中提取有周期性的隐藏信号

# 工具


- [如何正确使用Git](https://blog.csdn.net/qq_43075378/article/details/120067900)
- [如何正确使用Markdown](https://www.bilibili.com/video/BV1Yb411c7Hi)
- [python使用者一定要会用的笔记本-Jupyter](https://www.bilibili.com/video/BV1Q4411H7fJ?spm_id_from=333.999.0.0)
- [如何正确使用GitHub](https://www.ixigua.com/6892223361208812043?wid_try=1)
- [如何正确使用“虚拟机”-docker](https://www.bilibili.com/video/BV1og4y1q7M4?p=1)
- [敲代码不会给变量起名字？用这个网站啊！](https://unbug.github.io/codelf/#position%20list)

# FAQ

* [如何利用GitHub搭建自己的博客网站？](https://www.toutiao.com/a6992456857474449934/?log_from=275cf7f05bdfc_1630748431310)
* [Python中常用的矩阵运算有哪些？](https://cs231n.github.io/python-numpy-tutorial/#numpy)
* [ubuntu终端如何实现科学上网](https://www.jianshu.com/p/3ea31fcca279)
* [代码是如何驱动硬件的？](https://www.cnblogs.com/zhugeanran/p/8605757.html)

# 代码/项目

[如何将论文中的深度学习方法用pytorch实现？](https://mp.weixin.qq.com/s/iQdRqxw7pjPMAa3suiJRVA)

* [使用pytorch生成GAN网络](https://github.com/xerifg/MyNotebook/blob/main/GAN_pytorch.ipynb)
* [使用pytorch搭建全连接神经网络识别MINIST数据集的手写数字](https://github.com/xerifg/MyNotebook/blob/main/pytorch%2BMINIST.ipynb)

# 思考

* [年轻时候做点什么投资自己，才能受益终身？](https://www.ixigua.com/6895365107195314701)





