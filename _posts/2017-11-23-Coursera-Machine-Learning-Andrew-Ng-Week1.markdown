---
layout:     post
title:      "Coursera吴恩达机器学习Week1"
subtitle:   "Machine Learning on Coursera by Andrew Ng Week1"
date:       2017-11-23 01:11:00
author:     "MBB"
header-img: "img/in-post/real-time-rendering/the-graphics-rendering-pipeline/background.png"
header-mask: 0.3
catalog:    true
tags:
    - 机器学习
    - 学习笔记
---

## 介绍
### 什么是机器学习

1. Arthur Samuel：“不用显示编程而让计算机具备学习能力的研究领域。”
2. Tom Mitchell：“ 一个程序被认为能从经验E中学习，解决任务 T，达到性能度量值P，当且仅当，有了经验E后，经过P评判，程序在处理T时的性能有所提升。”

通常来说机器学习问题可以被划分为两大类：监督学习与无监督学习。

### 监督学习
在监督学习中，我们给定一个数据集并且已知正确的输出应该是什么样的，要记住在输入与输出之间有一种关系。
监督学习问题被归类为“回归”与“分类”问题。在回归问题中，我们尝试在连续的输出中预测结果，也就是说我们尝试将输入变量映射到某个连续函数中。在分类问题中，我们尝试在离散输出中预测结果。换句话说，我们尝试将输入变量映射到离散分类中。

**例子1：**

在房地产市场中，给定房子的大小来预测房价。房价相对于房子大小来说是一个连续函数，所以这是个回归问题。
我们也可以将这个问题转化成为一个分类问题：房子的售价比提出的价格高还是低。这样我们就将房子在基于价格的基础上分成了两个离散的类别（比提出的价格低/比提出的价格高）。

**例子2：**

1. 回归-给出一张人的照片，在已给定的照片的基础上预测他们的年龄。

2. 分类-给定一位肿瘤患者，预测肿瘤是良性还是恶性。

### 无监督学习
无监督学习可以让我们在不知道结果具体是什么样的情况来处理问题。我们可以在不需要知道变量影响的情况下从数据中得到结果。

我们可以通过数据中变量之间的关系来聚类数据从而得到结果。

无监督学习的预测结果是没有反馈的。

**例子**

聚类：假设有1,000,000个不同的基因，根据不同的变量，例如寿命，位置，角色等等，将这些基因进行分组，使每组基因在某些方面相似或者相关。

非聚类：“[鸡尾酒晚会算法](https://en.wikipedia.org/wiki/Cocktail_party_effect)”，让你可以在混乱的环境中得到分类（例如从人声和音乐的混音中独立识别出人声和音乐）。

## 模型与代价函数
### 模型表示
为了统一公式，我们用\\(x^{(i)}\\)表示“输入”变量，也叫做输入特征，用\\(y^{(i)}\\)表示我们要预测的输出或者叫做目标变量。一对\\((x^{(i)},y^{(i)})\\)叫做一个训练样本，我们用于学习的数据集——\\((x^{(i)},y^{(i)});i=1,...,m\\)——叫做一个训练集。注意上标“\\((i)\\)”仅仅表示训练集中的所以，跟幂没有关系。我们同时也用\\(X\\)表示输入值空间，\\(Y\\)表示输出值空间。

为了更加正式地描述监督学习问题，我们的目标是，给定一个训练集，学习得到一个函数\\(h:X\to Y\\)，\\(h(x)\\)是相对应\\(y\\)值的“良好”预测器。由于历史原因，这个函数\\(h\\)被称作“**假设**”。如下图所示，，这个过程大概是这样的:![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\model_representation.png)

当我们所要预测的目标变量的值是连续的，例如我们所提到的房价预测例子，我们将这种学习问题称作回归问题。当Y只能是几个少量的离散值时，例如给定一个生活区，预测一个住宅是房屋还是公寓，我们称这种学习问题为分类问题。

### 代价函数
我们可以用**代价函数**来测量我们假设函数(hypothesis function)\\(h(x)\\)的精度，其实就是对所有输入x经过假设函数h得到的结果y的所有结果求平均差(其实是一种更高级的平均值)，可以由如下公式表示：

\\[J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x_i)-y_i)^2\\]

将之拆分开来看就是\\(\frac{1}{2}\bar{x}\\)，其中\\(\bar{x}\\)就是\\(h_\theta(x_i)-y_i\\)的方差，也就是预测值与实际值之间的差。

这个函数也叫做“平方差函数”，或者叫做“平均方差”。平均值取($\frac{1}{2}$)是为了梯度下降计算的方便，平方的导数可以与$\frac{1}{2}$抵消掉。下图简要说明了代价函数做了什么:![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\what_cost_function_does.png)

### 代价函数-直观感受I
如果就我们所见的来讲，训练数据集离散地分布在x-y平面上，我们尝试找出一条直线（由$h_\theta(x)$定义），让这条直线穿过这些离散的数据点。
我们的目的就是得到一条最佳的直线。那么怎么定义这条最佳直线呢？也就是这些离散的点在垂直方向上到这条直线的距离的平方的平均值最小。最理想的情况就是这条直线穿过所有这些点，在这种情况下$J(\theta_0,\theta_1)$等于0。下图展示了理想情况下代价函数为0的例子：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\cost_function_of_1.png)</center>

当$\theta_1=1$时，我们得到一条斜率为1的穿过所有数据点的直线。相反地，当$\theta_1=0.5$，我们可以看到数据点到直线的垂直距离增加了。<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\cost_function_of_0.5.png)</center>

这使得我们的代价函数增长到了0.58.计算其他几个点可以得到下图：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\cost_function_plotting.png)</center>

因此作为目标，我们应该最小化代价函数。在本例中，$\theta_1=1$就是我们的目标最小值。

### 代价函数-直观感受II
等值线图是由许多等值线组成的图。有两个变量的函数的等值线在在同一条线上的所有点对应的值都是相等的。如下图右图所示：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\contour_plot.png)</center>

同一颜色的线所得到的代价函数的值都相同。例如图中同一条绿色线上的三个点的$J(\theta_0,\theta_1)$值都相同。右图中绿色带圈的X点表示左图中直线($\theta_0=800,\theta_1=-0.15$)所对应的代价函数值。取另一个$h(x)$并绘制其等值线图，得到如下图所示：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\contour_plot_2.png)</center>

我们可以看到，当$\theta_0=360,\theta_1=0$时，等值线图离中心更近了，这样就减少了代价函数的误差。现在给我们的假设函数一个稍微正数一点的斜率，我们可以得到更好的数据：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\contour_plot_better.png)</center>

上图尽可能地最小化代价函数，结果$\theta_1$大约等于0.12，$\theta_0$大约等于250。最终再将左图中的这些值绘制到右边的等值图中，这些绘制的点趋近于聚集在右图中最内层圈的中心。

## 参数学习
### 梯度下降
我们知道了什么是假设函数以及对于给定数据的最优解的判定，现在我们需要估算假设函数中参数的值，于是我们引入了梯度下降。

以$\theta_0$为x轴，$\theta_1$为y轴，代价函数为z轴，在给定的$\theta_0,\theta_1$范围内绘制图像，如下图所示：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\gradient_descent.png)</center>

当代价函数的值达到上图中极低点时，我们就找到了代价函数的最小值，如上图中的红色箭头所示。

我们通过不断对代价函数求导（求代价函数在某点的切线）迭代来找到最小值。具体做法：首先在图像中初始化$\theta_0,\theta_1$，在初始化点对代价函数求导（求切线），然后沿切线方向前进一定步长（由参数$\alpha$决定）得到新的点$\theta_0^{'},\theta_1^{'}$，然后再在新的点对代价函数求导并进行迭代，直到$\theta_0,\theta_1$收敛，即：

$$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$$

收敛，其中j为特征索引（j=0,1）。

在每一次迭代中，应该同步更新参数$\theta_0,\theta_1,,...,\theta_n$。如果先更新前一个参数然后代入更新下一个参数将会出错，如图所示：<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\simultaneously_update.png)</center>

### 梯度下降直观感受
为了对梯度下降有更直观地感受，我们忽略参数$\theta_0$，只保留$\theta_1$，于是我们可以得到如下方程，并尝试对其进行迭代直到收敛：

$$\theta_1:=\theta_1-\alpha\frac{d}{d\theta_1}J(\theta_1)$$

忽略$\frac{d}{d\theta_1}J(\theta_1)$的符号，最终$\theta_1$收敛于其最小值。从下图中可以看到，当斜率为负时，$\theta_1$的值不断增大，当斜率为正时，$\theta_1$的值不断减小。<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\single_parameter_gradient_descent.png)</center>

从另一方面说明，我们应该调整我们参数$\alpha$使得梯度下降算法在合理的时间内收敛。收敛失败或者收敛时间太长都表明$\alpha$的不合理。<center>![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\gradient_descent_about_alpha.png)</center>

#### 所以在步进值$\alpha$固定的情况下我们的梯度下降是如何收敛的呢？
收敛的直观感受就是$\frac{d}{d\theta_1}J(\theta_1)=0$，也就是说我们的凸函数$J(\theta_1)$达到最小值（$J(\theta_1)$始终为凸函数）。在最小值处导数始终为0，于是我们得到：

$$\theta_1:=\theta_1-\alpha*0$$

![](img\in-post\machine_learning\coursera_stanford_andrew_ng\week1\gradient_descent_converge_at_bottom.png)

## 线性回归的梯度下降
当具体应用到线性回归中时，我们可以得到新的梯度下降方程。带入到具体的代价方程与假设函数中，我们可以得到方程：

$$\theta_0:=\theta_0-\alpha\frac{1}{m}\sum^{m}_{i=1}(h_0(x_i)-y_i)$$
$$\theta_1:=\theta_1-\alpha\frac{1}{m}\sum^{m}_{i=1}((h_0(x_i)-y_i)x_i)$$

其中m为训练集的大小，$\theta_0$为与$\theta_1$同步变化的常量，$x_i,y_i$为训练集中的样本值。

这种方法对训练集中的每一个样本进行迭代计算，又被称作批量梯度下降。由于梯度下降会受局部最小值的影响，为了简化问题，我们提出了线性回归只有一个全局小值，没有局部最小值，因此梯度下降必定收敛于全局最小值（假设学习率$\alpha$不是太大）。实际上，函数$J$是一个凸二次函数。

## 线性代数复习
找本线性代数的书复习下，主要复习知识点：
1. 矩阵与向量
2. 矩阵加法与标量乘法
3. 矩阵与向量乘法
4. 矩阵与矩阵乘法
5. 矩阵乘法性质
6. 矩阵的逆与转置