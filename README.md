
# 笔记初衷

我会在这里将自己学习过程中的知识以笔记的形式在这里记录下来，会包含算法、数据结构、编程语言、常用工具的使用，以及经常会遇到的各种问题，还会包含平时自己的一些思考，或许还会根据需要再添加其他类的笔记，大家有好的想法的欢迎进行补充，这个笔记的初衷就是--共同进步。

# 开始之前

[学习如何学习](https://www.coursera.org/learn/learning-how-to-learn)

学习新知识很重要。但是，复习知识更重要，否则，时间一长你之前学习知识就会忘记，相当于你之前花的时间浪费掉了，这里有一个用于复习的抽认卡软件推荐给大家（[Anki](https://apps.ankiweb.net/)），类似于一些英语学习的软件学习单词的方式，让你复习你之前学过的各种知识。

# 目录

* [数据结构](#数据结构)
* [经典算法](#经典算法)
* [机器学习](#机器学习)
* [深度学习](#深度学习)
* [神经网络](#神经网络)
* [强化学习](#强化学习)
* [数据集](#数据集)
* [目标检测](#目标检测)
* [信号处理](#信号处理)
* [机器人](#机器人)
* [编程](#编程)
* [工具](#工具)
* [FAQ](#FAQ)
* [代码/项目](#代码/项目)
* [思考](#思考)

# 数据结构

一个来自b站up的《剑指offer》的[教程](https://github.com/Jack-Cherish/LeetCode)，包含了各种数据结构的讲解，以及代码实现

浙江大学的数据结构课程，[这里](https://www.bilibili.com/video/BV1JW411i731)

* #### 算法复杂度/Big-O/渐进分析法

  [渐进表示（视频）](https://www.youtube.com/watch?v=iOq5kSKqeR4)

  [Big-O记号  (视频)](https://www.youtube.com/watch?v=ei-A_wy5Yxw&list=PL1BaGV1cIH4UhkL8a9bJGG356covJ76qN&index=3)

   #### i  一级知识

  * 线性表

    由同类型**数据元素**构成**有序序列**的线性结构

  *  数组（Arrays）

  * 链表（Linked Lists）

    数组与链表是下面各种数据结构实现的物理结构类型

  *  堆栈（Stack）

  * 队列（Queue）

    循环队列

  * 哈希表（Hash table）（又称散列表、字典）

    ”键“>>>散列函数>>>对应的“值“所在的存储位置>>>”值“；

    数组与链表都是直接映射到内存，而散列表需要使用**散列函数**来确定元素的存储位置；

    平均情况（正常情况）下，散列表的读取速度和数组一样快，存储删除速度和链表一样快；

    几乎不可能有一个散列函数可以将不同的”键“映射到数组的不同位置，因此，常常会出现**冲突**的问题；

    解决冲突问题：

     1. 较低的**填装因子**

        通过增加散列表的长度来降低填装因子

     2. 良好的散列函数

        例如：SHA函数

  #### ii 二级知识

  * **树**

    结点的度：结点的子树个数

    树的度：树的所有结点中度最大的度数

    叶结点：度为0的结点

    由N个节点构成的有限集合

    1. 二叉树

       二叉树的表示方式（数组、链表、结构数组（静态链表））

       二叉树的遍历方式

       1. 递归遍历:(编程序利用递归实现)

          * 先序遍历
       
            根节点->左儿子->右儿子

          * 中序遍历

            左儿子->根节点->右儿子

          * 后序遍历

            左儿子->右儿子->根节点

       2. 非递归遍历
       
          * 层序遍历（利用队列实现）
          * 中序遍历（除了可以用递归实现，也可以用堆栈实现）
       
       按树的结构分类：
       
       * 斜二叉树
       * 完美（满）二叉树
       * 完全二叉树
       
       搜索、查找问题：
       
       * 二叉搜索树
       
         左右两个子节点小于父结点，左节点小于右结点
       
       * 平衡二叉树（特殊的二叉搜索树）
       
         任一结点左、右子树的高度差绝对值不超过1

  * **堆**

    结构性：用数组表示的**完全二叉树**

    最大堆：父节点比子树中的所有节点都大

    最小堆：与最大堆刚好相反

    性质：由最大堆、最小堆定义发现，堆从根节点到任意节点路径上的节点序列都是有序的，最大堆是从大到小，最小堆是从小到大

    重要操作：

    * 堆的插入

    * 堆的删除

      先将最后一个叶节点放到根节点的位置，然后循环调整树节点位置满足最大堆的要求

    * 堆的建立

      方法1：按照最大堆或者最小堆的规则，将所有数据一个个插入到空的堆中

      方法2：

      ​	（1）将N个节点按顺序存入，先满足完全二叉树的结构要求

      ​	（2）调整所有节点的位置，以满足最大堆或最小堆的有序特性
    
  * **哈夫曼树**

    问题：如何根据节点不同的查找频率构造更有效的**搜索树**？

    构造原理：每次把权值最小的两棵二叉树合并

    应用：哈夫曼编码

    注意：利用哈夫曼编码得到的编码一定是最优编码（长度最短），但最有编码不一定必须用哈夫曼编码得到

  * **集合**

    问题：解决在已知多对电脑与电脑之间是连接的，如何知道指定的某两个电脑之间是否有链接

    思路：将所有电脑假设为集合，电脑与电脑之间的链接就是集合与集合的合并，将合并以**树的根节点**链接实现

    按秩归并（改进集合合并部分）：解决两个集合合并时，两个根节点谁指向谁。

    ​				   按高度：矮树指向高树根节点

    ​				   按规模：低节点数的树指向高节点数的树

    路径压缩（改进集合父节点查找部分）

    ​				   在查找的过程中建立一个新树，把查找路径中的节点全部指向最后的根节点，这样，在下次查				   找时可以极大的缩短查找某节点到最终节点的路径的长度

  * **图**

    图在程序中的**表示方法**：

     1. 邻接矩阵

     2. 邻接表

        定义G[N]为一个数组，N为所有节点，数组中每个元素存储的是每个链表的头指针，每个链表存储对应节点的链接情况，即它与哪些节点相连。

    **遍历方法**

    ​	1. 深度优先搜索（DFS）

    ​		类似于树的先序遍历

    ​	2. 广度优先搜索（BFS）

    ​		类似于树的层序遍历

  ### iii 三级知识

  * 最短路径

    1. 单源最短路径

       * 无权图

         BFS算法（广度优先算法）

       * 有权图

         [Dijkstra(狄克斯特拉)算法](https://blog.csdn.net/qq_35644234/article/details/60870719)

         注意：其只适用于有向无环图，且权重为正

         每次从最短路径的数组选取最小的数值代表的节点；
         
         节点集合T每加入一个新的节点，代表从原点到该节点的最短路径已确定

    2. 多源最短路径

       * 对所有节点依次使用单源最短路径的方案
       * Floyd算法

  * 排序问题

    1. 选择排序

       思路：每次将数组最大或最小元素放入新数组

    2. 快速排序

       思路：随便找一个元素作为分割线，将数组分为两部分，一部分比分割线小，另一部分比分个线大，通过递归分而治之

  * 动态规划算法

    典型应用问题：背包问题

    将大问题分成**若干小问题**，小问题的解又服务于大问题，最终解决大问题。动态规划并不是你想象的那样高深，只是名字起的看起来高大上而以，其实本质类似于传统递归算法的优化而以，主要需要解决的是**边界条件**与**递归关系**，边界条件是最简单的情况是啥，递归关系是在当前已知的基础上再多给你一个，怎么办。进而可以实现从最简单的小问题逐步递推到最终的大问题。

    **注意**：动态规划算法的每个子问题都是离散的，即不依赖于其它子问题

    动态规划的核心部分（单元格的绘制方法）

    单元格可以防止重复计算

    1. 如何将这个问题划分为子问题？
    2. 网格的坐标轴是什么？
    3. 单元格中的值是什么？

  * 最小生成树

    包含所有结点（N个），一共N-1条边，没有连通子树

    典型应用问题：多村庄之间修路

    * Prim算法（一种贪心算法）

      每次寻找离当前树最近的结点，将其收入树中

    * Kruskal算法（一种贪心算法）

      每次寻找在满足最小生成树的约束的前提下的最小边（以及边链接的结点），加入树中

    

# 经典算法&思想

什么是一个好的算法？

你的算法可以避开一些无用的操作，例如，不用去遍历所有的情况就可以找到问题的最优解。即，最短的运行时间、最少的占用空间。

* [P问题、NP问题、NPC问题(NP完全问题)、NP难问题](https://blog.csdn.net/u014044032/article/details/91513982)

  * P问题：存在多项式算法的问题，即，可以找到一个算法可以在多项式的时间内找到问题的最优解；
  * NP问题：能在多项式时间内验证得出一个正确解的问题，但不一定是最优解，即，无法找到一个算法使其可以在多项式的时间内找到问题的最优解，例如旅行商问题，可以在有限时间内找到问题的一个解，但因为无法在有限时间内遍历所有城市，因此，无法在有限时间内找到最优解；
  * NPC问题：同一类的NP问题都可以约化到这个问题，注意，约化操作会让问题变得更复杂。其核心思想是如果我可以解决这个复杂问题，那它对应的简单问题也可以用同样的方法解决；
  * NP难问题：同一类的NP问题可以约化到这个问题，但这个问题不一定是NP问题，换句话说，他有可能是一个可以找到一个算法在有限时间找到最优解的问题。

  为什么要判断一个问题是不是NP问题或者NPC问题？

  因为当你发现它是一个NP问题时，你就会确定一定没有算法可以找到它的最优解，因此，你的注意力需要放在如何找它的近似解，例如：贪婪算法。

* **递归**（分而治之的思想）

  组成部分：1. 基线条件 2. 递归条件 

* 线性规划

  在一定的约束条件下使得目标最大化。这是一个很大的概念，前面的图问题也只是它的一个子集。

* [卡尔曼滤波](https://zhuanlan.zhihu.com/p/45238681)

* [马尔可夫链](https://www.bilibili.com/video/BV19b4y127oZ?from=search&seid=11829849833797645360&spm_id_from=333.337.0.0)

* [PCA降维](https://zhuanlan.zhihu.com/p/32412043)

  主要是对原始数据的协方差矩阵进行特征分解，提取前k个特征向量，则由这些特征向量组成的矩阵，就是将原数据降维到k维的转换矩阵

# 机器学习

[吴恩达的机器学习课程-b站转载](https://www.bilibili.com/video/BV164411b7dx)

一个b站up的机器学习笔记，[这里](https://github.com/Jack-Cherish/Machine-Learning)

* K最近邻（分类、回归）

* 朴素贝叶斯分类器（朴素指的是特征之间相互独立）

  利用贝叶斯公式的条件概率公式进行条件概率转换来完成预测

# 深度学习

[吴恩达的深度学习课程-b站转载](https://www.bilibili.com/video/BV1FT4y1E74V)

一个b站up的深度学习笔记，[这里](https://github.com/Jack-Cherish/Deep-Learning)

* #### 深度学习框架

  1. [Pytorch](https://www.bilibili.com/video/BV1Rv411y7oE)
  
     [Pytorch模型参数的存储、加载、初始化](https://zhuanlan.zhihu.com/p/48524007)
  
     [数据集的创建与加载， TensorDataset 、DataLoader 、Dataset、Variable](https://blog.csdn.net/zw__chen/article/details/82806900)
  
     * 从dataset类中提取的数据如果想放入计算图中进行传播与梯度计算需要先转换为Variable类型，才能送去网络模型中进行训练；
  
       神经网络在做运算的时候，需要先构建一个**计算图谱**，然后在里面进行前向传播和反向传播；
  
       Variable本质上和Tensor（张量）没有区别，Variable会放入一个计算图中进行**前向、反向传播和自动求导**，不懂的话可以看[这里](https://blog.csdn.net/weixin_44478378/article/details/104292622)
  
     * 当我们自己定义了一个继承与Dataset类的类时，必须重写 **len** 方法，该方法提供了dataset的大小； 	**getitem** 方法， 该方法支持从 0 到 len(self)的索引，也决定了调用数据集时返回哪些数。[博客](https://www.jianshu.com/p/2d9927a70594)
  
  2. [Tensorflow](https://www.bilibili.com/video/BV1kW411W7pZ)



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
  3. 对比损失(Contrastive loss)：多用于度量学习
  4. [三元组损失(Triplet loss)](https://blog.csdn.net/u013082989/article/details/83537370):多用于度量学习(Metric learning)
  5. 四元组损失(Quadruplet loss)：多用于度量学习


* 激活函数
  1. Sigmoid函数
  2. tanh函数
  3. ReLU函数
* 似然函数
  1. Softmax函数
* 梯度优化器
  1. AdaDrad
  2. AdaDelta
  3. 动量方法
  4. SGD(随机剃度下降)
* 防止网络过拟合
  1. Dropout
  2. DropConnect
  3. L1、L2正则化
  4. 提前结束训练
  5. 模型集成(训练多个模型，投票表决)
* 预处理
  1. 均值减法（将样本数据处理为均值为0）
  
  2. 均一化（标准化）（将样本数据约束为均值为0，方差为1的标准化数据）
  
     [为什么神经网络训练前需要进行标准化(Normalization)？](https://blog.csdn.net/keeppractice/article/details/105330513)
  
  3. 白化（消除样本数据间的相关性）

## 1. 卷积神经网路

* [BP神经网络上](https://blog.csdn.net/weixin_42398658/article/details/83859131)、[BP神经网络中](https://blog.csdn.net/weixin_42398658/article/details/83929474)、[BP神经网络下](https://blog.csdn.net/weixin_42398658/article/details/83958133)
* [卷积网络的动态可视化](https://www.bilibili.com/video/BV1AJ411Q72b?p=3&spm_id_from=pageDriver)
* 卷积神经网络保持平移、缩放、变形不变性的原因：
  1. 局部感受野
  2. 权值共享
  3. 下采样（池化）：减少参数，防止过拟合

####    参数的设定

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

## 3. 自编码器

自编码器网络是一种无监督学习，其功能有：1. 数据的降维，2. 提取特征表达，3. 利用自编码器可以训练其他神经网络的初始参数

* 降噪自编码器

  通过编码与解码，消除掉图像中的噪声

* 稀疏自编码器

  引入正则化项，解决中间层参数冗余问题

* 栈式自编码器

  只有编码没有解码，由多个2层的编码器组合而成，逐层进行训练

# 强化学习

* [基本概念(human-level control through deep reinforcement learning)](https://github.com/xerifg/MyNotebook/blob/main/materials/dqn-atari.pdf)

# 数据集

* [步态数据集](https://raw.githubusercontent.com/xerifg/MyNotebook/main/picture/%E6%AD%A5%E6%80%81%E6%95%B0%E6%8D%AE%E9%9B%86.bmp)

# 目标检测

### 2D目标检测

* [2D目标检测的发展历程及相关论文](https://github.com/hoya012/deep_learning_object_detection)

### 3D目标检测

* [3D目标检测综述CSDN](https://blog.csdn.net/wqwqqwqw1231/article/details/90693612)

# 信号处理

* [自相关与互相关](https://zhuanlan.zhihu.com/p/77072803)

  自相关的特点，原信号的自相关信号，虽幅值改变，但保留了原信号的频率特征。常用应用：从杂乱的信号中提取有周期性的隐藏信号

# 数据处理

* 普式分析（Procrustes analysis）[博客](https://www.cnblogs.com/nsnow/p/4745730.html)

  Procrustes analysis是一种用来分析形状分布的统计方法。从数学上来讲，普氏分析就是利用最小二乘法寻找形状A到形状B的仿射变换。

# 机器人

* **坐标系旋转**的相关知识

  * 旋转的几种表示方式：欧拉角、[轴角](https://zh.wikipedia.org/wiki/%E8%BD%B4%E8%A7%92)、旋转矩阵、四元数

  * [旋转矩阵的推导](https://www.cnblogs.com/zhoug2020/p/7842808.html)

    3维空间中，旋转矩阵不带位移是3乘3大小，带位移是4乘4大小；

    旋转矩阵的乘积表示两次旋转的叠加

  * 四元数（[博客1](https://www.jianshu.com/p/7aa0fd8503c5)、[博客2](https://blog.csdn.net/shenshen211/article/details/78492055)、[博客3](http://www.wy182000.com/2012/07/17/quaternion%E5%9B%9B%E5%85%83%E6%95%B0%E5%92%8C%E6%97%8B%E8%BD%AC%E4%BB%A5%E5%8F%8Ayaw-pitch-roll-%E7%9A%84%E5%90%AB%E4%B9%89/)）

    用四个数表示旋转；

    两个四元数的乘积满足复数乘积规则，且表示两次旋转的叠加；

    最大的优点是容易进行旋转的插值

  * [指数映射](https://blog.csdn.net/weixin_33801856/article/details/94562415)

    指数映射用来将轴角表示转化为旋转矩阵形式，对应的转换公式叫做**罗德里格斯公式**（[Rodrigues’ formula](https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula)）

    推导过程：一阶微分方程-》指数形式的旋转表示-》利用泰勒展开求解出指数形式的旋转-》旋转矩阵

  * 李群和李代数

    [博客上](https://zhuanlan.zhihu.com/p/76959511)、[博客下](https://zhuanlan.zhihu.com/p/33156814)

* 机器人仿真平台

  1. Webots(免费开源)
  2. coppeliasim
  3. Gazebo(ROS平台)
  4. MuJoCo(多用于强化学习的机器人仿真)
  5. PyBullet(python环境下的仿真模块)

# 编程

[编程的命名规则](https://blog.csdn.net/qq_43075378/article/details/120717118)

# 工具


- [如何正确使用Git](https://blog.csdn.net/qq_43075378/article/details/120067900)
- [如何正确使用Markdown](https://www.bilibili.com/video/BV1Yb411c7Hi)
- [python使用者一定要会用的笔记本-Jupyter](https://www.bilibili.com/video/BV1Q4411H7fJ?spm_id_from=333.999.0.0)
- [如何正确使用GitHub](https://www.ixigua.com/6892223361208812043?wid_try=1)
- [如何正确使用“虚拟机”-docker](https://www.bilibili.com/video/BV1og4y1q7M4?p=1)
- [敲代码不会给变量起名字？用这个网站啊！](https://unbug.github.io/codelf/#position%20list)
- [Unity使用手册](https://docs.unity3d.com/Manual/index.html)

# FAQ

* [如何利用GitHub搭建自己的博客网站？](https://www.toutiao.com/a6992456857474449934/?log_from=275cf7f05bdfc_1630748431310)
* [Python中常用的矩阵运算有哪些？](https://cs231n.github.io/python-numpy-tutorial/#numpy)
* [ubuntu终端如何实现科学上网](https://www.jianshu.com/p/3ea31fcca279)
* [代码是如何驱动硬件的？](https://www.cnblogs.com/zhugeanran/p/8605757.html)
* [什么是行人重识别技术？](https://blog.csdn.net/qq_30121457/article/details/108918512)
* [如何快速入门人工智能](https://www.bilibili.com/video/BV1Ry4y1h7Kd)
* [什么是feature scaling，什么时候需要feature scaling？](https://mp.weixin.qq.com/s/ehnoWIg8vK7dX_vFmg4zNQ?scene=25#wechat_redirect)
* [如何将tensorflow1.0的代码升级到v2.0？](https://www.tensorflow.org/guide/upgrade);[中文博客](https://blog.csdn.net/xovee/article/details/93402172)

# 代码/项目

[如何将论文中的深度学习方法用pytorch实现？](https://mp.weixin.qq.com/s/iQdRqxw7pjPMAa3suiJRVA)

* [使用pytorch生成GAN网络](https://github.com/xerifg/Myipynb/blob/main/GAN_pytorch.ipynb)
* [使用pytorch搭建全连接神经网络识别MINIST数据集的手写数字](https://github.com/xerifg/Myipynb/blob/main/pytorch%2BMINIST.ipynb)
* [Github-字节跳动的视频人像抠图技术](https://github.com/PeterL1n/RobustVideoMatting)

# 思考

* [年轻时候做点什么投资自己，才能受益终身？](https://www.ixigua.com/6895365107195314701)





