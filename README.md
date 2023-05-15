
# 笔记初衷

我会在这里将自己学习过程中的知识以笔记的形式在这里记录下来，会包含算法、数据结构、编程语言、常用工具的使用，以及经常会遇到的各种问题，还会包含平时自己的一些思考，或许还会根据需要再添加其他类的笔记，大家有好的想法的欢迎进行补充，这个笔记的初衷就是--共同进步。

# 开始之前

[学习如何学习](https://www.coursera.org/learn/learning-how-to-learn)

学习新知识很重要。但是，复习知识更重要，否则，时间一长你之前学习知识就会忘记，相当于你之前花的时间浪费掉了，这里有一个用于复习的抽认卡软件推荐给大家（[Anki](https://apps.ankiweb.net/)），类似于一些英语学习的软件学习单词的方式，让你复习你之前学过的各种知识。

# 目录

* [基础数学](#基础数学)
* [数据结构](#数据结构)
* [经典算法](#经典算法)
* [机器学习](#机器学习)
* [深度学习](#深度学习)
* [神经网络](#神经网络)
* [强化学习](#强化学习)
* [信号处理](#信号处理)
* [数据处理](#数据处理)
* [计算机视觉](#计算机视觉)
* [渲染](#渲染)
* [机器人](#机器人)
* [自动驾驶](#自动驾驶)
* [编程](#编程)
* [工具](#工具)
* [FAQ](#FAQ)
* [项目收藏](#项目收藏)
* [已读论文](#已读论文)

# 基础数学

* [常见的最优化求解算法总结，如SGD、牛顿法等](https://blog.csdn.net/qq_43075378/article/details/126943292?spm=1001.2014.3001.5502)

# 数据结构

一个来自b站up的《剑指offer》的[教程](https://github.com/Jack-Cherish/LeetCode)，包含了各种数据结构的讲解，以及代码实现

浙江大学的数据结构课程，[这里](https://www.bilibili.com/video/BV1JW411i731)

[代码随想录](https://programmercarl.com/)，一个超级全的数据结构刷题解析和找工作的心得笔记

* #### 算法复杂度/Big-O/渐进分析法

  [渐进表示（视频）](https://www.youtube.com/watch?v=iOq5kSKqeR4)

  [Big-O记号  (视频)](https://www.youtube.com/watch?v=ei-A_wy5Yxw&list=PL1BaGV1cIH4UhkL8a9bJGG356covJ76qN&index=3)

  <b><details><summary>i  一级知识</summary></b>

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

  </details>

  <b><details><summary>ii 二级知识</summary></b>

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

         a.左节点小于父结点；b.右节点大于父结点；c.左右子树仍然满足二叉搜索树

         **注意**：左子树所有节点都小于根节点，右子树所有节点都大于根节点

       * 平衡二叉树（特殊的二叉搜索树）

         任一结点左、右子树的高度差绝对值不超过1

       * [红黑树](https://www.bilibili.com/video/BV1zU4y1H77f?from=search&seid=13804250434115165457&spm_id_from=333.337.0.0)

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

    注意：利用哈夫曼编码得到的编码一定是最优编码（长度最短），但最优编码不一定必须用哈夫曼编码得到

  * **集合**

    问题：解决在已知多对电脑与电脑之间是连接的，如何知道指定的某两个电脑之间是否有链接

    思路：将所有电脑假设为集合，电脑与电脑之间的链接就是集合与集合的合并，将合并以**树的根节点**链接实现

    按秩归并（改进集合合并部分）：解决两个集合合并时，两个根节点谁指向谁。

    ​				   按高度：矮树指向高树根节点

    ​				   按规模：低节点数的树指向高节点数的树

    路径压缩（改进集合父节点查找部分）

    ​				   在查找的过程中建立一个新树，把查找路径中的节点全部指向最后的根节点，这样，在下次查找时可以极大的缩短查找某节点到最终节点的路径的长度

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

  * **动态规划算法**

    典型应用问题：背包问题

    将大问题分成**若干小问题**，小问题的解又服务于大问题，最终解决大问题。动态规划并不是你想象的那样高深，只是名字起的看起来高大上而以，其实本质类似于传统递归算法的优化而以，主要需要解决的是**边界条件**与**递归关系**，边界条件是最简单的情况是啥，递归关系是在当前已知的基础上再多给你一个，怎么办。进而可以实现从最简单的小问题逐步递推到最终的大问题。

    **注意**：动态规划算法的每个子问题都是离散的，即不依赖于其它子问题

    动态规划的核心部分（单元格的绘制方法）

    ​	单元格可以防止重复计算

    ​	1. 如何将这个问题划分为子问题？

    ​	2. 网格的坐标轴是什么？

    ​	3. 单元格中的值是什么？

  </details>

  <b><details><summary>iii 三级知识</summary></b>

  * **最短路径**

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

  * **排序问题**

    1. 选择排序

       思路：每次将数组最大或最小元素放入新数组

    2. 快速排序

       思路：随便找一个元素作为分割线，将数组分为两部分，一部分比分割线小，另一部分比分个线大，通过递归分而治之

    3. 拓扑排序

       思路：每次输出入度为0的结点

       应用问题：排课问题（AOV问题）

    4. 冒泡排序

       改进：当发现在一次循环中没有元素进行交换时，提前跳出循环
  
    5. 插入排序

       思路：类似于打牌时给手中的牌排序
       
       冒泡排序与插入排序，从本质上说都是进行了两两的逆序对的交换，所有速度取决于原序列的逆序对的数量

    6. 希尔排序

       思路：原序列数量为N，先进行N/2间隔的排序，再进行N/4间隔的排序，最后进行1间隔的排序

       改进： 将增量序列改为Hibbard增量序列，即间隔分别取2^k - 1，这样可以防止出现像8，4，2间隔导致可能出现的多次无用排序

    7. 堆排序

       一种特殊的选择排序，为了解决选择排序中每次从剩余序列中提取最大值的复杂度，将序列变更为堆，利用堆的特点解决上述问题

    8. 归并排序

       核心思路：有序子列的归并

       实现方式：1. 递归实现 ；2. 非递归实现  

    9. 表排序

       问题：当需要排序的元素很大，例如是一个电影的时候，元素的移动成本就会很大

       核心思想：移动元素对应的地址，而不进行物理排序

    10. 桶排序

        问题：已知所需排列的元素范围，例如在0-100之间

        核心思路：在0-100之间对应位置放一个桶，所有待排元素对号入座，最后依次输出

    11. 基数排序

        问题：桶的个数远远大于待排元素的个数

        核心思想：依次按照个位、十位、百位的顺序进行排序，每次桶的个数是10

        方法：次位优先排序

        补充：与次位优先排序对应的另一种考虑多关键字的是主位优先排序

  * **查找问题**

    1. 顺序查找（静态查找）

    2. 二分查找（静态查找）

       需要先对原序列进行排序

    3. 二叉搜索树（动态查找）

    4. 利用散列表 ”计算出” 查找位置（动态查找）

       （1）以关键字key为自变量，通过设计一种散列函数，计算出对应的函数值，即数据对象的存储位置

       （2）可能不同的关键字会映射到同一个存储位置上，所以还需要一种解决 “冲突“ 的策略

  * **散列表查找问题**

    一种以空间换时间的查找方法

    * 散列函数的设计

      1. 关键字是数字

         核心思想：计算简单，散列函数计算的地址空间分布均匀

         直接定址法（线性映射）、除留余数法（求余数）、数字分析法（观察数字的特点，例如身份证的前几位可能都一样，直接忽略）、折叠法（将数字分为几段然后相加）、平方取中法（将原数字取平方然后取结果中间的几个数字作为地址）
  
      2. 关键字是字符
  
         1. 将所有字符对应的ASCII码相加
         2. 将ASCII码利用进制权重进行相加

    * 解决 “冲突” 问题

      1. 开放地址法（换个地址存）

         若发生第i次冲突，就在原来的计算；出来的地址基础上加上di，根据di的取值不同，分为：

         1）线性探测（di = i）

         2）平方探测（di = +(-)i*i）

         3）双散列（偏移量di利用另一个三列表计算出来）

      2. 链地址法（同一位置的冲突对象组织在一起）

         分离链接法（将有冲突的元素放在链表里面）

  * **最小生成树问题**

    包含所有结点（N个），一共N-1条边，没有连通子树

    典型应用问题：多村庄之间修路

    * Prim算法（一种贪心算法）

      每次寻找离当前树最近的结点，将其收入树中

    * Kruskal算法（一种贪心算法）
  
      每次寻找在满足最小生成树的约束的前提下的最小边（以及边链接的结点），加入树中
  

</details>

# 经典算法

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

* [卡尔曼平滑](https://blog.csdn.net/weixin_42647783/article/details/106035691)

* [马尔可夫链](https://www.bilibili.com/video/BV19b4y127oZ?from=search&seid=11829849833797645360&spm_id_from=333.337.0.0)

* [PCA降维](https://zhuanlan.zhihu.com/p/32412043)

  主要是对原始数据的协方差矩阵进行特征分解，提取前k个特征向量，则由这些特征向量组成的矩阵，就是将原数据降维到k维的转换矩阵。[视频讲解](https://www.bilibili.com/video/BV1E5411E71z/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)

* [SVD分解](https://www.bilibili.com/video/BV16A411T7zX/?spm_id_from=333.788&vd_source=8c065934da63850a7afd383a2017d263)

* [匈牙利匹配](https://www.bilibili.com/video/BV16K4y1X7Ph/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)

* [VAE(变分自编码器)](https://www.bilibili.com/video/BV1f34y1e7EK/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)

* [HMM隐马尔可夫](https://www.bilibili.com/video/BV14R4y1N7iH/?spm_id_from=333.788&vd_source=8c065934da63850a7afd383a2017d263)

* [信息量、熵、KL散度、交叉熵](https://www.bilibili.com/video/BV15V411W7VB/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)

* [SIFT算法](https://www.bilibili.com/video/BV1Hb411r7n8/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)-图片的特征点提取

* [Loftr](https://www.bilibili.com/video/BV19e4y1z7C7?p=1&vd_source=8c065934da63850a7afd383a2017d263)-end to end的特征点提取与匹配（使用了transformer）

# 机器学习

[吴恩达的机器学习课程-b站转载](https://www.bilibili.com/video/BV164411b7dx)

一个b站up的机器学习笔记，[这里](https://github.com/Jack-Cherish/Machine-Learning)

<b><details><summary>常见的机器学习算法</summary></b>

* **监督学习**

  *分类与回归问题都是有标签的监督学习*

  * [K近邻算法(KNN)](https://zhuanlan.zhihu.com/p/25994179)

  * [决策树](https://www.bilibili.com/video/BV1ar4y137GD/?spm_id_from=333.788)

  * [Logistic回归](https://www.bilibili.com/video/BV17r4y137bW/?spm_id_from=333.788&vd_source=36f81f373ea58937a8b5ec640201633c)

  * [朴素贝叶斯分类器](https://www.bilibili.com/video/BV1eT411V7jM/?spm_id_from=333.788)（朴素指的是特征之间相互独立）

    利用贝叶斯公式的条件概率公式进行条件概率转换来完成预测

  * [SVM(支持向量机)](https://www.bilibili.com/video/BV16T4y1y7qj/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)

  * [集成算法](https://zhuanlan.zhihu.com/p/30035094)

    将多个分类器组合在一起的分类、回归算法

    * Bagging

      有放回的随机采样

    * Boosting

      给样本增加权重

      * [AdaBoost]([Adaboosting介绍_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1dk4y117cW/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263))（是Boosting思想的一种具体实现算法）

* **半监督学习**

* **无监督学习**

  * [EM算法(高斯混合模型)](https://www.bilibili.com/video/BV1RT411G7jJ/?spm_id_from=333.788)
  * [K均值(KMeans)](https://zhuanlan.zhihu.com/p/78798251)

</details>

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

* #### 模型的部署与加速

  * **torch2trt**：python中的使用tensorRT进行网络推理加速的模块，1.进行了kernel融合，2.提升GPU利用率，直接提升网络的推理速度，网络中有不支持的算子则无法使用；经过torch2trt处理后的模型仍然是torch的网络模型。torch2trt可以直接将torch模型转换为tensorRT的加速推理模型。[演示视频](https://www.bilibili.com/video/BV1AU4y127Uo/?spm_id_from=333.788&vd_source=8c065934da63850a7afd383a2017d263)
  * **trtorch**：可以实现将加载的网络模型转换为tensorRT的加速推理模型，torch模型的保存、加载、使用trtorch进行部署加速，[演示视频](https://www.bilibili.com/video/BV1ky4y167TG/?spm_id_from=333.788.recommend_more_video.4&vd_source=8c065934da63850a7afd383a2017d263)
  * [模型的剪枝、量化、蒸馏](https://zhuanlan.zhihu.com/p/282777488)

# 神经网络

<b><details><summary>神经网络的基础知识</summary></b>神经网络可以分为两大类，一类是类似于卷积神经网络的多层神经网络，其中，BP网络是始祖；

另一类是类似于深度信念网络的相互连接型网络，其中，Hopfied网络是始祖

* 特征工程
  * 编码方式
    * 独热编码（one hot encoding）
    * [Embedding](https://blog.csdn.net/weixin_44493841/article/details/95341407)
  
* 损失(代价)函数

  [损失函数是如何被设计出来的？](https://www.bilibili.com/video/BV1Y64y1Q7hi/?spm_id_from=333.788&vd_source=8c065934da63850a7afd383a2017d263)

  1. 最小二乘误差函数（二次代价函数）
  2. [交叉熵代价函数](https://zhuanlan.zhihu.com/p/38241764)     [视频](https://www.bilibili.com/video/BV15V411W7VB/?spm_id_from=333.788&vd_source=8c065934da63850a7afd383a2017d263)
  3. 对比损失(Contrastive loss)：多用于度量学习
  4. [三元组损失(Triplet loss)](https://blog.csdn.net/u013082989/article/details/83537370):多用于度量学习(Metric learning)
  5. 四元组损失(Quadruplet loss)：多用于度量学习
  6. [smooth L1损失](https://blog.csdn.net/weixin_41940752/article/details/93159710?ops_request_misc=%7B%22request%5Fid%22%3A%22160525394119724842952021%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=160525394119724842952021&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-93159710.pc_first_rank_v2_rank_v28p&utm_term=smooth L1&spm=1018.2118.3001.4449)


* 激活函数
  1. Sigmoid函数
  2. tanh函数
  3. ReLU函数
  
* 似然函数
  1. [Softmax函数](https://blog.csdn.net/lz_peter/article/details/84574716)
  
* 梯度优化器
  
  [视频](https://www.bilibili.com/video/BV1Vx411j7pA?from=search&seid=4270052544885788174&spm_id_from=333.337.0.0)
  
  1. SGD(随机剃度下降)
  2. Momentum(动量方法)
  3. AdaGrad
  4. AdaDelta
  5. Adam(Momentum + AdaGrad)
  
* 防止网络过拟合
  1. Dropout
  
  2. DropConnect
  
  3. [L1、L2正则化](https://www.bilibili.com/video/BV1aE411L7sj?p=3)
  
     数学角度：为什么正则化可以降低模型复杂度？
  
     降低了给模型参数的求解**加入了约束条件**，利用最优化的KKT条件求解，推导出的最终结论
  
  4. 提前结束训练
  
  5. 模型集成(训练多个模型，投票表决)
  
  6. BN层
  
* 预处理
  1. 均值减法（将样本数据处理为均值为0）
  
  2. 均一化（标准化）（将样本数据约束为均值为0，方差为1的标准化数据）
  
     [为什么神经网络训练前需要进行标准化(Normalization)？](https://blog.csdn.net/keeppractice/article/details/105330513)
  
  3. 白化（消除样本数据间的相关性）

* 神经网络的基本组成

  <b><details><summary>网络中的各种层</summary></b>

  * 卷积层：普通卷积、空洞卷积

    <details><summary>空洞卷积</summary>
        优点：不改变特征图大小，不增加参数量的情况下，增大感受野。<br>
        缺点：<br>
        1.网格效应：由于空洞卷积的稀疏采样方式，当多个空洞卷积叠加时，有些象素根本没有被用到，会损失信息的连续性与相关性；
        2.深层的特征图信息与前面的没有相关性：空洞的稀疏采样，导致远距离的卷积得到的结果缺乏相关性；
        3.不同尺度的物体的关系：大的dilation rate对大物体的检测与分割有利，但对于小物体不利。
    </details>

  * 激活函数层：Sigmoid、Softmax

  * 池化层：最大值池化、平均值池化

  * Dropout层：防止过拟合
  
  * BN层：缓解梯度消失，模型稳定

    从数据分布的角度避免参数陷入饱和区，其核心是对输入数据进行去均值与方差的操作，为使修改后的数据尽可能恢复原数据的表达能力，在上一步操作后添加了线性操作。

    <b><details><summary>优点</summary></b>

    * 缓解梯度消失，加速网络收敛。让激活函数的输入数据落在了非饱和区，缓解了梯度消失的问题。
    * 简化调参，网络更稳定。BN层抑制了参数的微小变化随网络加深二被放大的问题
    * 防止过拟合。BN层将每个batch均值与方差引入到网络中，相当于给训练过程增加了随机噪声，可以起到一定的正则效果，防止过拟合。
  
    </details>
  
  * 全连接层：分类
  
  </details>

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

* 加速神经网络的推理速度的工具：Openvino(CPU)、TensorRT(GPU)

* <b><details><summary>梯度弥散与梯度爆炸问题</summary></b>loss下降到一定程度不动了，和loss先下降一定程度又出现了增长的现象

  * 梯度弥散产生原因：
    1. 激活函数的“饱和”（使用ReLu作为中间层激活函数）
    2. 样本中有奇异样本，引起模型无法收敛（使用Batch Normalization）
    3. 反向传播传播梯度时，随着网络的深度加深，梯度的幅度会急剧减小，导致浅层神经元的权重更新非常缓慢（使用ResNet思想）
    4. 从数学角度，梯度的连乘，导致了梯度的逐渐消失
    5. 学习率过大导致模型振荡无法收敛
  
  * 梯度爆炸产生原因：
    1. 神经网络的初始权重过大，导致每层网络的反向求导的结果都大于1，这样，梯度相乘就会变得更大
    2. 学习率非常大而导致
  
  </details>

</details>

<b><details><summary>经典神经网络</summary></b>

* CNN

* AlexNet

* VGGNet

  首次使用更小的卷积核和更深的网络结构

* Inception(GoogLeNet)

  首次提出Inception模块(使用不同的卷积核进行卷积，然后进行通道concat)；

  首次提出用1*1的卷积核进行降维

* ResNet

  首次提出残差连接，解决网络加深带来的梯度消失等问题

* DenseNet

  提出了Dense Block概念(每层的输入由前面所有卷积层的输出组成【使用通道的concat进行的拼接】)

* FPN(特征金字塔)

  提出了通过融合不同层的特征来改善多尺度的检测问题，包括四部分：自下而上、自上而下、横向连接、卷积融合

* DetNet

  在ResNet的结构的基础上提出了空洞卷积，使得模型兼顾较大的感受野与较高的分辨率

* Faster RCNN

  经典的两阶段物体检测

* Mask RCNN

  在Faster RCNN的基础上进行改进的一种分割网络

* SSD

  经典的单阶段物体检测

* YOLO

  经典的单阶段物体检测

* SqueezeNet

  轻量型网络

* MobileNet

  轻量型网络

* CornerNet

  free anchor，基于角点的检测

* CenterNet

  free anchor，基于中心点的检测

* [Transformer](https://www.bilibili.com/video/BV1MY41137AK/)

</details>

# 强化学习

* [基本概念(human-level control through deep reinforcement learning)](https://github.com/xerifg/MyNotebook/blob/main/materials/dqn-atari.pdf)

# 信号处理

* [自相关与互相关](https://zhuanlan.zhihu.com/p/77072803)

  自相关的特点，原信号的自相关信号，虽幅值改变，但保留了原信号的频率特征。常用应用：从杂乱的信号中提取有周期性的隐藏信号

# 数据处理

* 普式分析（Procrustes analysis）[博客](https://www.cnblogs.com/nsnow/p/4745730.html)

  Procrustes analysis是一种用来分析形状分布的统计方法。从数学上来讲，普氏分析就是利用最小二乘法寻找形状A到形状B的仿射变换。

# 计算机视觉

* ## 图像处理

  <b><details><summary>常用的图像处理算法</summary></b>
  
  * 图像滤波
  
    均值滤波、中值滤波、高斯滤波、双边滤波、低通滤波、高通滤波
  
  * 图像增强
  
    直方图均衡化
  
  * 图像边缘检测（锐化）
  
    梯度锐化、Sobel算子、Laplace算子
  
  * 图像分割
  
    阈值分割（固定阈值、自适应阈值等等）、Canny边缘检测、Hough变换、语义分割
  
  * 图像形态学
  
    腐蚀、膨胀、开运算、闭运算
  
  * 图像特征
  
    角点（Harris算法）、几何特征、直方图特征（均值方差等）、颜色特征
  
  * 图像复原
  
  * 图像压缩
  
    行程编码、霍尔曼编码
  
  </details>
  
* ## 感知与重建

  * ### 2D感知

    [2D目标检测的发展历程及相关论文](https://github.com/hoya012/deep_learning_object_detection)

    <b><details><summary>相关算法</summary></b>

    * [RetinaNet](https://www.bilibili.com/video/BV1g3411z72f/?spm_id_from=333.337.search-card.all.click&vd_source=8c065934da63850a7afd383a2017d263)

    </details>

  * ### 3D感知

    [3D目标检测综述CSDN](https://blog.csdn.net/wqwqqwqw1231/article/details/90693612)

    [PointNet++的作者亲自讲解3D检测的综述](https://www.bilibili.com/video/BV1wA411p7FZ?spm_id_from=333.999.0.0)

    [视差图计算深度](https://blog.csdn.net/fb_help/article/details/83339092)

    [单目的深度估计综述](https://zhuanlan.zhihu.com/p/111759578)

    <b><details><summary>主要方法</summary></b>

    * 被动感知
      * 单目3D
        * 反变换（需要知道必要的先验知识，如目标的大小已知）
        * 关键点和3D模型（2D模型感知关键点，然后于3D模型做匹配）
          * [Deep MANTA](https://zhuanlan.zhihu.com/p/25996617)（2017）
        * 几何约束（通过几何约束确定目标的3D位置，使用网络预测目标大小与方向）
          * [MonoGRNet](https://blog.csdn.net/abrams90/article/details/98484420)（2019）
        * 直接预测3D模型（纯网络方法）
      * 双目3D
        * [PSMNet](https://blog.csdn.net/weixin_45614254/article/details/110708111)（2018）
        * STereo TRansformers
      * 多目3D
        * FSD
    * 主动感知
      * 结构光法
      * TOF飞行时间法
      * 激光三角测距法

    </details>

  * ### 3D重建

    [基于多视图几何的三维重建](https://apposcmf8kb5033.pc.xiaoe-tech.com/live_pc/l_5ea4fa067411e_oJ22mYIW)

    **基础概念**

    TSDF：3D重建结果的表示方法,mesh重建算法的表示形式

    * #### 场景重建

      <b><details><summary>相关算法</summary></b>

      * [NeuralRecon](https://github.com/zju3dv/NeuralRecon)
      * BundleFusion
      * [Possion Reconstruction(泊松重建)](https://groups.google.com/group/seminair-si350/attach/46ed857dec1bb9e5/泊松表面重建(英译汉版本).pdf?part=0.1)：目前较为成熟的光滑mesh重建经典算法

      </details>

* ## 人体方向

  <b><details><summary>相关文章、论坛</summary></b>

  *  [人体姿态估计(Human Pose Estimation)经典方法整理](https://zhuanlan.zhihu.com/p/104917833)
  *  [2020CVPR人体姿态估计论文盘点](http://www.360doc.com/content/21/0602/13/61825250_980111489.shtml)
  *  [【CVPR 2021】PRTR：基于transformer的2D Human Pose Estimation](https://zhuanlan.zhihu.com/p/368067142)

  </details>

  * ### 2D感知

  * ### 3D感知

  * ### 3D重建

    **基础概念**

    SMPL：人体3维模型

    GHUM：谷歌提出的一种可训练产生的人体3D模型

    * #### 人体重建

      <b><details><summary>相关算法</summary></b>

      * [MeshTransformer](https://github.com/microsoft/MeshTransformer)
      * GLAMR

      </details>

* ## 数据集

  * [数据集搜索地址](https://paperswithcode.com/datasets?mod=images)
  * [步态数据集](https://raw.githubusercontent.com/xerifg/MyNotebook/main/picture/%E6%AD%A5%E6%80%81%E6%95%B0%E6%8D%AE%E9%9B%86.bmp)

* ## 评价指标

  [ROC曲线与AUC值](https://blog.csdn.net/dujiahei/article/details/87932096)

  [Precision, Recall, Accuracy, F1-score, confidence score, IoU, AP, mAP，ROC曲线，P-R曲线](https://www.jianshu.com/p/fd9b1e89f983)

  

# 渲染

视频教程：[games-101](https://www.bilibili.com/video/BV1X7411F744?p=1)

<b><details><summary>教程笔记</summary></b>

* **变换**

  * 正交变换
  * 透视变换（可以产生近大远小）

* **光栅化(Rasterrization)**

  用于将空间中的物体投影到相机平面中，即将空间中的物体在相机平面中“画”出来的操作叫光栅化

  * Triangles

  * 反走样

    如何去除成像图面上的锯齿

    1直接增大成像图面分辨率

    2从频域角度考虑，有一种方案是：先进行低通滤波去除高频信息（让图像变模糊）再进行采样

    2从频域角度考虑，有一种方案是：先进行低通滤波去除高频信息（让图像变模糊）再进行采样

* **着色(shading)**

  * 解决遮挡问题（前后景距离不同）

    * 画家画法（先画远的再画近的）
    * Z-Buffer（从成像图面的像素角度，即每个像素记录距离最近的深度度，将目标深度最小的目标画在该像素上）

  * 着色（光源，材质，角度）

    * [漫反射](https://github.com/xerifg/MyNotebook/blob/main/picture/diffuse.png)
    * [高光](https://github.com/xerifg/MyNotebook/blob/main/picture/specular.png)
    * [环境光](https://github.com/xerifg/MyNotebook/blob/main/picture/ambient.png)

    将漫反射、高光、环境光叠加，即可构成着色模型，[效果图](https://github.com/xerifg/MyNotebook/blob/main/picture/blinn.png)

    * 着色频率

      并不一定要对图片中物体的所有像素依次利用着色模型进行着色，可以对其一“块”进行统一着色，按照“块”的划分，可以分为三类

      * Flat shading
      * Gouraud shading
      * Phong shading

  * Texture Mapping

    用于给物体进行“贴图”，或者说在控制每个点的属性，其本质上是在控制每个点在着色时的漫反射的kd系数，进而控制该点的颜色。

    * MIPMAP

    Texture的其他用途:

    * 记录环境光
    * 凹凸贴图

将上述的投影变换、光栅化、着色三个模块合在一起，便可构成渲染pipeline，[示意图](https://github.com/xerifg/MyNotebook/blob/main/picture/render_pipeline.png)

* **几何**(Geometry)

  不是所有的物体都适合用三角形进行表示的

  * 几何的表示方式
    * implicit
      * 代数表示：x^2+y^2+z^2=1
      * 基础形状的相互组合
      * Distance Functions
      
    * explicit
    
      * Point Cloud
    
      * Polygon Mesh
    
      * Curves
    
        * 贝塞尔曲线
          * de Casteljau algorithm(如何画出贝塞尔曲线)
        * Spline(样条)：可以控制的曲线
    
      * Surfaces
    
        * 贝塞尔曲面
    
        * Mesh Subdivision(曲面细分)
    
          * Loop Subdivision(先细分再调整,只针对三角形)
          * Catmull-Clark Subdivision(可以应用在不止三角形面)
    
        * Mesh Simplification(曲面简化(减少曲面个数))
    
          * Edge Collapse
    
            如何选择那个边进行collapse,通过最小化Quadric Error
  
* **光线追踪**(Ray Tracing)

  如何渲染出阴影？可以使用Shadow Mapping
  
  * Recursive Ray Tracing(发生了多次反射折射)
  * Ray-Intersection
    * Ray-Intersection With Box(简化求光线相交的计算量)
    * Ray-Scene Intersection(简化求光线与Box中的那个object相交)
    * Spatial Partition
      * Oct-Tree
      * KD-Tree
      * BSP-Tree
  * 辐射度量学(Radiometry)(更准确的描述光线)
  
* **Materials and Appearances**

  决定了object的反射相关的系数，例如镜面反射、漫反射、折射

  Materials==BRDF

  * Microfacet Material

    从远处看看到的是材质，近处看是很多的微小镜面

  * 材质分类

    * Isotropic(微表面没有方向)
    * Anisitropic(微表面有方向，如金属)
  
* **Color and Perception**

  * Light Field(or Lumigraph)(光场)

    记录物体表面任一点沿任意方向的光的强度。
  
* **Animation**

  * 物理模型

    * Mass Spring System(质点弹簧系统)

      复杂的object可以由一系列的质点弹簧系统组成

    * Particle System(粒子系统)

  * Kinematices

    * Forward Kinematices
    * Inverse Kinematices

  * Rigging(对模型的动作pose的控制，类似于提线木偶)


</details>



# 机器人

* **坐标系旋转**的相关知识

  * 旋转的几种表示方式：欧拉角、[轴角](https://zh.wikipedia.org/wiki/%E8%BD%B4%E8%A7%92)、旋转矩阵、四元数

  * 欧拉角：会产生万向节死锁的现象

  * [旋转向量](https://www.bilibili.com/video/BV1W3411E7XR/?spm_id_from=333.788.recommend_more_video.1&vd_source=8c065934da63850a7afd383a2017d263)

    轴角的一种表示方式，其向量的模代表旋转的角度，向量方向代表所绕的轴的方向

  * [旋转矩阵的推导](https://www.cnblogs.com/zhoug2020/p/7842808.html)

    3维空间中，旋转矩阵不带位移是3乘3大小，带位移是4乘4大小；

    旋转矩阵的乘积表示两次旋转的叠加

  * [四元数](https://www.bilibili.com/video/BV1bq4y1F7Yp/?spm_id_from=autoNext&vd_source=8c065934da63850a7afd383a2017d263)（[博客1](https://www.jianshu.com/p/7aa0fd8503c5)、[博客2](https://blog.csdn.net/shenshen211/article/details/78492055)、[博客3](http://www.wy182000.com/2012/07/17/quaternion%E5%9B%9B%E5%85%83%E6%95%B0%E5%92%8C%E6%97%8B%E8%BD%AC%E4%BB%A5%E5%8F%8Ayaw-pitch-roll-%E7%9A%84%E5%90%AB%E4%B9%89/)）

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


# 自动驾驶
[UdaCity联合百度的Apollo推出的自动驾驶入门讲解视频](https://apollo.baidu.com/community/online-course/1)

自动驾驶汽车的四大组成部分：参考车辆、硬件平台、软件系统、云平台  
主要技术组成部分：高精度地图、定位、感知、预测、规划、控制
### 高精度地图
首先，车载激光雷达和摄像头可以扫描道路和周围环境，获取高精度的三维点云数据和图像数据。  
然后，这些数据可以通过计算机视觉技术进行处理，提取出道路、车道线、交通标志等地图信息。此外，GPS和惯性导航系统可以提供车辆的位置和方向信息，进一步提高地图的精度和准确性。 
最后，这些数据经过处理和整合后，可以生成高精度地图。这些地图包含了道路的几何形状、车道线、交通标志、路口、交叉口等详细信息，可以为自动驾驶车辆提供精确的位置和导航信息，从而实现自主驾驶。和单纯使用GPS定位的以m为单位的误差相比，高精度地图可以将误差控制在cm级别。

### 定位
这里的定位是需要做到自动驾驶可以使用的cm级别的定位
* GPS+RTK
* IMU
* 激光雷达  
需要最新的高精度地图
  1.  直方图滤波定位
  2. 卡尔曼滤波定位
* 视觉定位  
需要最新的高精度地图
  1. 粒子滤波定位

### 感知
通过利用车辆上的所有传感器（激光雷达、摄像头、毫米波雷达、超声波传感器、IMU）感知周围环境，例如：车辆周围的交通情况、道路状况、障碍物等信息。  
感知的四大任务：检测(Detection)、分类(Classification)、跟踪(Tracking)、分割(Segmentation)   
不同的传感器有各自的优势：摄像头擅长检测与分类，激光雷达擅长障碍物检测尤其是夜晚，雷达擅长处理恶劣天气情况，目前业内认为将上述所有传感器进行融合是最优的感知方案。常用的融合算法为卡尔曼滤波，利用卡尔曼的预测与测量状态融合更新部分实现多传感器的感知结果融合。



# 编程

[编程的命名规则](https://blog.csdn.net/qq_43075378/article/details/120717118)

**编程解决问题的四大步**：1. 问题的定义 ---> 2. 算法设计 ---> 3. 数据结构的选择 ---> 4. 优化代码

**注意1：** 发现具体的代码实现困难时，可以先实现其伪代码来构建程序框架，再将其翻译成具体的代码

**注意2：** 当在一个大项目中添加新功能时，先不要把新写好的程序直接放到大程序里进行调试，先编写一个简单的测试**脚手架**来测试新函数，没有问题后再放入大程序中

### 编程的五大法则

1. 能用小程序实现的就不要用大程序
2. 重复的代码使用通用的函数表示
3. 数据结构决定程序结构
4. 使用数组重新编写重复代码
5. 封装复杂结构，例如：封装成结构体或类

### 系统执行效率优化的四大方向

* **粗略估算**

  在宽松的假设条件下，粗略估计一下系统执行所需要的时间或者空间。

  通过一些小实验来获得一些关键参数，如计算机完成一次乘法大概需要多长时间等等

  **安全系数**：因为对系统中有些问题的不了解或者估算参数的错误，为了补偿这个引入安全系数，即在原先估计的时间的基础上乘以一个系数（2，4，6）

  **Little定律**：队列中物体的平均数量 = 平均进入速率 * 平均逗留时间

* **算法设计与数据结构**

  1. 保存状态，避免重复计算：通过使用一些空间来保存中间的计算结果，避免花时间对其进行重复计算
  2. 将信息预处理至数据结构中：例如如果需要计算数组元素的求和，可以在最外层循环之前将求和信息存储在一个数组中
  3. 分而治之
  4. 扫描算法：假设已经解决了[0,i-1]的问题，如何将其拓展为解决[0,i]的问题
  5. 下界：即算法最快的运行时间也不会超过这个下界，用来确定算法是否已达到最优

* **代码调优**

  通过测试找到代码中最消耗时间的几个函数，分别去调整这几个函数的效率

* **节省空间**

### 增加代码安全性

1. 程序中在可能出现错误的位置加入**断言（assert）**,当程序执行到此处出现预料之外的结果，就会直接输出AssertionError
2. 在可能出现异常的地方添加异常处理模块，例如python中的try...except..

## 编程语言

* #### C/C++

​		[什么是条件编译？如：#if, #ifndef, #else, #endif](https://blog.csdn.net/qq_36662437/article/details/81476572)

​		[智能指针](https://www.bilibili.com/video/BV1fK411H7CA?from=search&seid=18162903492104110019&spm_id_from=333.337.0.0)、[右值引用](https://www.bilibili.com/video/BV1Vq4y1K7ut?from=search&seid=14280358547772966820&spm_id_from=333.337.0.0)、[函数指针](https://www.bilibili.com/video/BV1uz41187DQ?from=search&seid=221618835148746806&spm_id_from=333.337.0.0)

* #### Python

  python中万物皆**对象**

  <b><details><summary>常问问题</summary></b>

  * 可变对象：list、dict、set
  * python装饰器
  * python中的序列化与反序列化
  * 垃圾回收机制：1.引用计数；2.标记-清除；3.分代回收
  * 列表与元组的区别：元组元素不可变
  * lambda表达式：不用起名字的函数
  * python迭代器、生成器
  * python中的docstring：用一对三个单引号括起来的，对函数的功能的说明文字
  * python中的包与模块，\__init__.py
  * global关键字：在函数内对函数外的变量进行操作
  * python中的三元运算符：a if a>b else b

  </details>

  

# 工具


- [如何正确使用Git](https://blog.csdn.net/qq_43075378/article/details/120067900)
- [如何正确使用Markdown](https://www.bilibili.com/video/BV1Yb411c7Hi)
- [python使用者一定要会用的笔记本-Jupyter](https://www.bilibili.com/video/BV1Q4411H7fJ?spm_id_from=333.999.0.0)
- [如何正确使用GitHub](https://www.ixigua.com/6892223361208812043?wid_try=1)
- [如何正确使用“虚拟机”-docker](https://www.bilibili.com/video/BV1og4y1q7M4?p=1)
- [敲代码不会给变量起名字？用这个网站啊！](https://unbug.github.io/codelf/#position%20list)
- [Unity3D使用手册](https://docs.unity3d.com/Manual/index.html)
- [远程链接助手-tmux，将前端与后端分离](https://zhuanlan.zhihu.com/p/98384704)
- [windows下的全能终端神器-mobaXterm](https://blog.csdn.net/xuanying_china/article/details/120080644)
- github代码阅读方式：直接按“。”；代码运行调试工具：gitpod

# FAQ

* [如何利用GitHub搭建自己的博客网站？](https://www.toutiao.com/a6992456857474449934/?log_from=275cf7f05bdfc_1630748431310)
* [Python中常用的矩阵运算有哪些？](https://cs231n.github.io/python-numpy-tutorial/#numpy)
* [ubuntu终端如何实现科学上网](https://www.jianshu.com/p/3ea31fcca279)
* [代码是如何驱动硬件的？](https://www.cnblogs.com/zhugeanran/p/8605757.html)
* [什么是行人重识别技术？](https://blog.csdn.net/qq_30121457/article/details/108918512)
* [如何快速入门人工智能](https://www.bilibili.com/video/BV1Ry4y1h7Kd)
* [什么是feature scaling，什么时候需要feature scaling？](https://mp.weixin.qq.com/s/ehnoWIg8vK7dX_vFmg4zNQ?scene=25#wechat_redirect)
* [如何将tensorflow1.0的代码升级到v2.0？](https://www.tensorflow.org/guide/upgrade);[中文博客](https://blog.csdn.net/xovee/article/details/93402172)

# 项目收藏

[如何将论文中的深度学习方法用pytorch实现？](https://mp.weixin.qq.com/s/iQdRqxw7pjPMAa3suiJRVA)

* [使用pytorch生成GAN网络](https://github.com/xerifg/Myipynb/blob/main/GAN_pytorch.ipynb)
* [使用pytorch搭建全连接神经网络识别MINIST数据集的手写数字](https://github.com/xerifg/Myipynb/blob/main/pytorch%2BMINIST.ipynb)
* [Github-字节跳动的视频人像抠图技术](https://github.com/PeterL1n/RobustVideoMatting)
* [一个临床步态分析网站](http://www.clinicalgaitanalysis.com/)
* [微软亚洲研究院和北大联合开发的一个多模态算法-女娲](https://github.com/microsoft/NUWA)

# 已读论文

* [单目的绝对深度估计，2022，《Depth Map Decomposition for Monocular Depth Estimation》](https://arxiv.org/pdf/2208.10762.pdf)

# 大佬主页

* [于涛](https://ytrock.com/)（3D人体重建方向）