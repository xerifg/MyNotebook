# 笔记初衷

我会在这里将自己学习过程中的知识以笔记的形式在这里记录下来，会包含算法、数据结构、编程语言、常用工具的使用，以及经常会遇到的各种问题，还会包含平时自己的一些思考，或许还会根据需要再添加其他类的笔记，大家有好的想法的欢迎进行补充，这个笔记的初衷就是--共同进步。

# 经典算法

* [卡尔曼滤波](https://zhuanlan.zhihu.com/p/45238681)

# 神经网络

神经网络可以分为两大类，一类是类似于卷积神经网络的多层神经网络，其中，BP网络是始祖；

另一类是类似于深度信念网络的相互连接型网络，其中，Hopfied网络是始祖

* 损失(代价)函数
  1. 最小二乘误差函数（二次代价函数）
  2. [交叉熵代价函数](https://blog.csdn.net/u014313009/article/details/51043064)


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
* 受限波尔兹曼机
* 深度信念网络

# 强化学习

* [基本概念(human-level control through deep reinforcement learning)](https://github.com/xerifg/MyNotebook/blob/main/materials/dqn-atari.pdf)

# 数据集

* 步态数据集

# 工具


- [如何正确使用Git](https://blog.csdn.net/qq_43075378/article/details/120067900)
- [如何正确使用Markdown](https://www.bilibili.com/video/BV1Yb411c7Hi)
- [python使用者一定要会用的笔记本-Jupyter](https://www.bilibili.com/video/BV1Q4411H7fJ?spm_id_from=333.999.0.0)

# FAQ

* [如何利用GitHub搭建自己的博客网站？](https://www.toutiao.com/a6992456857474449934/?log_from=275cf7f05bdfc_1630748431310)
* [Python中常用的矩阵运算有哪些？](https://cs231n.github.io/python-numpy-tutorial/#numpy)

# 思考

* [年轻时候做点什么投资自己，才能受益终身？](https://www.ixigua.com/6895365107195314701)





