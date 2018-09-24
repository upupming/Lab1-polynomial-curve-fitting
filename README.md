# 多项式拟合正弦函数

## 〇、数学符号格式规范

本文格式参考了 [Bishop 2006]，若不加特殊说明，定义向量时均是列向量。

|数据类型|格式规范|MathJax 写法|实际效果|
|-|-|-|-|
|向量|小写加粗罗马字母|`\mathrm{\mathbf{w}}`|$\mathrm{\mathbf{w}}$|
|转置|右上标 T|`(w_0, w_1, ..., w_M)^T`|$(w_0, w_1, ..., w_M)^T$|
|随机变量分布类型|书法字母 (calligraphic letters)|`\mathcal{N}`|$\mathcal{N}$|
|矩阵|大写加粗罗马字母|`\mathrm{\mathbf{X}}`|$\mathrm{\mathbf{X}}$|

## 一、实验目的

+ 掌握最小二乘法求解（无惩罚项的损失函数）
+ 掌握加惩罚项（2 范数）的损失函数优化
+ 梯度下降法、共轭梯度法
+ 理解过拟合、克服过拟合的方法(如加惩罚项、增加样本)

## 二、实验要求及实验环境

### 实验要求

+ [x] 生成数据，加入噪声
+ [x] 用高阶多项式函数拟合曲线
+ [x] 用解析解求解两种 loss 的最优解（无正则项和有正则项）
+ [ ] 优化方法求解最优解（梯度下降，共轭梯度）
+ [x] 用你得到的实验数据，解释过拟合
+ [x] 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果

<!-- 7. 语言不限，可以用 matlab，python。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如 pytorch，tensorflow 的自动微分工具。 -->

### 实验环境

#### 硬件

+ Windows 10 64-bit
+ Python 3.7.0

#### 软件

+ Matplotlib
  
  Python 2D 绘图库

+ NumPy
  
  矩阵运算

## 三、设计思想

本次实验的目标是对正弦函数曲线 $sin(2\pi x)$ 进行拟合。简单来讲，就是根据已有数据集 $(x_i, y_i)$ 找到一条曲线，使其能够最好地预测真实情况下给定 $x$ 后计算出的 $y$ 值。正弦函数只是一个具体的例子，理解了算法的思想之后，我们可以对任意曲线进行拟合。

### 算法原理

向量、矩阵均采用数据结构均是 Numpy 中提供的数组结构。

#### 生成数据

编写一个函数，根据用户传入的函数生成指定数量的数据，利用 Numpy 库提供的 `numpy.random.normal` 加入以 0 为均值、用户指定方差的噪声。以字典 `{'xArray': array, 'tArray': array}` 的形式返回给用户。

#### 最小二乘法

##### 误差函数 $E(\mathrm{\mathbf{w}})$

最小二乘法通过**最小化误差的平方和**寻找数据的最佳函数匹配。

我们使用下面的多项式函数来拟合数据:

$$
y(x, \mathrm{\mathbf{w}}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j = 0}^Mw_jx^j
$$(1)

其中 $M$ 为多项式阶数，多项式系数向量 $\mathrm{\mathbf{w}} = (w_0, w_1, ..., w_M)^T$。

显然

$$
\begin{aligned}
  y(x, \mathrm{\mathbf{w}}) 
  &= 
    \begin{pmatrix}  
      1 & x & ... & x^M
    \end{pmatrix}
    \mathrm{\mathbf{w}}
\end{aligned}
$$(2)

其中 

为了求得最优的解析解，我们需要最小化 $N$ 个数据误差的平方和，也就是最小化下面的误差函数：

$$
E(\mathrm{\mathbf{w}}) = \frac{1}{2}\sum_{n = 1}^N\{y(x_n, \mathrm{\mathbf{w}}) - t_n\}^2
$$(3)

误差函数中的 $\frac{1}{2}$ 是利用高斯分布进行最大似然估计得来的。假定在 $x$ 已知的情况下，相应的 $t$ 服从以 $y(x, \mathrm{\mathbf{w}})$ 为均值、以 $\beta^{-1}$ 为方差的高斯分布，即：

$$
\begin{aligned}
  p(t|x, \mathrm{\mathbf{w}}, \beta)
  &= \mathcal{N}(t|y(x, \mathrm{\mathbf{w}}), \beta^{-1}) \\
  &= \frac{\exp{-\frac{[t - y(x, \mathrm{\mathbf{w}})]^2}{2\beta^{-1}}}}{(2\pi \beta^{-1})^{\frac{1}{2}}}
\end{aligned}
$$(4)

由于每个数据都是独立同分布的，根据乘法公式，对于训练数据集 $\mathrm{\mathbf{x}} = (x_1, x_2, ..., x_N)^T$ 和 $\mathrm{\mathbf{t}} = (t_1, t_2, ..., t_N)^T$ 则有：

$$
\begin{aligned}
  p(\mathrm{\mathbf{t}}|\mathrm{\mathbf{x}}, \mathrm{\mathbf{w}}, \beta) 
  &= \prod_{n = 1}^N\mathcal{N}(t_n|y(x_n, \mathrm{\mathbf{w}}), \beta^{-1}) \\
  &= \prod_{n = 1}^N\frac{\exp-{\frac{[y(x_n, \mathrm{\mathbf{w}}) - t_n )]^2}{2\beta^{-1}}}}{(2\pi \beta^{-1})^{\frac{1}{2}}}
\end{aligned}
$$(5)

这就是似然函数，即在 $\mathrm{\mathbf{w}}$ 已知的条件下，给定 $\mathrm{\mathbf{x}}$ 利用拟合曲线得到的估计值正好是 $\mathrm{\mathbf{t}}$ 的概率。我们的任务就是最大化似然函数，由于⼤量⼩概率的乘积很容易下溢，于是我们转而计算概率的对数和：

$$
\begin{aligned}
\ln p(\mathrm{\mathbf{t}}|\mathrm{\mathbf{x}}, \mathrm{\mathbf{w}}, \beta) 
&= \sum_{n = 1}^N\{-\frac{\beta}{2}[y(x_n, \mathrm{\mathbf{w}}) - t_n )]^2 - \frac{1}{2}\ln \frac{2\pi}{\beta}\}  \\
&= -\frac{\beta}{2}\sum_{n = 1}^N{[y(x_n, \mathrm{\mathbf{w}}) - t_n )]^2} + \frac{N}{2}\ln \beta - \frac{N}{2}\ln 2\pi
\end{aligned}
$$(6)

现假设系数向量 $\mathrm{\mathbf{w}}$ 取 $\mathrm{\mathbf{w_{ML}}}$ 时似然函数达到最大值。为了求解 $\mathrm{\mathbf{w_{ML}}}$，可以忽略后两项，同时将第一项中的 $-\frac{\beta}{2}$ 替换为 $\frac{1}{2}$ 也不会影响计算结果。也就是要最大化：

$$
-\frac{1}{2}\sum_{n = 1}^N{[y(x_n, \mathrm{\mathbf{w}}) - t_n )]^2}
$$(7)

这等价于最小化误差函数 (3)。

##### 最小化误差函数求得最优解 $\mathrm{\mathbf{w_{ML}}}$

下面对 (3) 做一些变形：

$$
\begin{aligned}
E(\mathrm{\mathbf{w}})
&= \frac{1}{2}\sum_{n = 1}^N{[y(x_n, \mathrm{\mathbf{w}}) - t_n )]^2} \\
&= \frac{1}{2}\left\|
\begin{pmatrix}
  1 & x_1 & x_1^2 & ... & x_1^M \\
  1 & x_2 & x_2^2 & ... & x_2^M \\
  ... & ... & ... & ... & ... \\
  1 & x_N & x_N^2 & ... & x_N^M \\
\end{pmatrix}_{N\times (M+1)}
\mathrm{\mathbf{w}}_{(M+1) \times 1} - \mathrm{\mathbf{t}_{N \times 1}}
\right\|^2
\end{aligned}
$$

令
$$
\mathrm{\mathbf{X}} = \begin{pmatrix}
  1 & x_1 & x_1^2 & ... & x_1^M \\
  1 & x_2 & x_2^2 & ... & x_2^M \\
  ... & ... & ... & ... & ... \\
  1 & x_N & x_N^2 & ... & x_N^M \\
\end{pmatrix}_{N\times (M+1)}
$$(8)

则

$$
\begin{aligned}
E(\mathrm{\mathbf{w}})
&= \frac{1}{2}\left\|
\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{t}}
\right\|^2 \\
&= \frac{1}{2}(\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{t}})^T(\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{t}}) \\
&= \frac{1}{2}(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{t}} - \mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} + \mathrm{\mathbf{t}}^T\mathrm{\mathbf{t}})
\end{aligned}
$$(9)

接下来我们需要求导，在下面的计算过程中要十分注意矩阵求导是该采用分子布局还是分母布局，下面引用一段 [Matrix calculus | Wikipedia] 中的原话：

> 1. If we choose numerator layout for $\frac {\partial \mathbf {y} }{\partial \mathbf {x} }$, we should lay out the gradient $\frac {\partial y}{\partial \mathbf {x} }$ as a row vector, and $\frac {\partial \mathbf {y} }{\partial x}$ as a column vector.
> 2. If we choose denominator layout for $\frac {\partial \mathbf {y} }{\partial \mathbf {x} }$, we should lay out the gradient $\frac {\partial y}{\partial \mathbf {x} }$ as a column vector, and $\frac {\partial \mathbf {y}}{\partial x}$ as a row vector.
> 3. In the third possibility above, we write $\frac {\partial y}{\partial \mathbf {x} '}$ and $\frac {\partial \mathbf {y} }{\partial x}$, and use numerator layout.

我们在接下来的计算中选择第 1 种规约 (Consistent numerator layout)，$\frac {\partial \mathbf {y} }{\partial x}$ 以 $\mathbf{y}$ 布局，$\frac{\partial y}{\partial \mathbf{x}}$ 以 $\mathbf{x}^T$ 布局，那么就有

$$
\begin{aligned}
  \frac{\partial\mathrm{\mathbf{w}}^T\mathrm{\mathbf{A}}\mathrm{\mathbf{w}}}{\partial\mathrm{\mathbf{w}}} = \mathrm{\mathbf{w}}^T\mathrm{\mathbf{A}} + \mathrm{\mathbf{w}}^T\mathrm{\mathbf{A}}^T  \\ \\
  \frac{\partial\mathrm{\mathbf{w}}^T\mathrm{\mathbf{A}}}{\mathrm{\mathbf{w}}} = \frac{\partial\mathrm{\mathbf{w}}\mathrm{\mathbf{A}^T}}{\mathrm{\mathbf{w}}} = \mathrm{\mathbf{A}}^T
\end{aligned}
$$(10)

对 $\mathrm{\mathbf{w}}$ 求导，得：

$$
\begin{aligned}
  \frac{\partial E(\mathrm{\mathbf{w}})}{\partial \mathrm{\mathbf{w}}}
  &= \frac{\partial \frac{1}{2}(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{t}} - \mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} + \mathrm{\mathbf{t}}^T\mathrm{\mathbf{t}})}{\partial \mathrm{\mathbf{w}}} \\
  &= \frac{1}{2}(2\mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}} - 2\mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}}) \\
  &= \mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}} - \mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}}
\end{aligned}
$$(11)

令导数为 $0$ 解得：

$$
\mathrm{\mathbf{w_{ML}}} = (\mathrm{\mathbf{X}}^{T}\mathrm{\mathbf{X}})^{-1}(\mathrm{\mathbf{X}}^{T}\mathrm{\mathbf{t}})
$$(12)

总结：对于已知训练数据集 $\mathrm{\mathbf{x}}、\mathrm{\mathbf{t}}$，规定一个多项式次数 $M$，根据相应的范德蒙德矩阵 (8) 以及式 (12) 就可以解出最优解 $\mathrm{\mathbf{w_{ML}}}$。再根据 (2) 即可计算估计值从而获得拟合曲线。

然而，当多项式次数 $M$ 刚好等于 $N-1$ 时，能够使训练数据集的误差函数取值为 0（例：2 次函数可以完全拟合 3 个点），但对测试数据集却不能很好地拟合，这就是**过拟合**。经过查阅资料，主要有这几种解决方法：

1. 增大数据量
2. 使用贝叶斯方法，根据数据集的规模自动调节有效参数数量
3. 修正误差函数，加入惩罚项，对其进行正则化 (regularization)

下面讨论加入惩罚项的最小二乘法。

#### 带惩罚项的最小二乘法

修正误差函数，在 (3) 式的基础上加入正则项：

$$
\begin{aligned}
\widetilde{E}(\mathrm{\mathbf{w}})
  &= \frac{1}{2}\sum_{n = 1}^N\{y(x_n, \mathrm{\mathbf{w}}) - t_n\}^2 + \frac{\lambda}{2}\|\mathrm{\mathbf{w}}\|^2 \\
  &= \frac{1}{2}(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{t}} - \mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} + \mathrm{\mathbf{t}}^T\mathrm{\mathbf{t}}) + \frac{\lambda}{2}(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{w}})
\end{aligned}
$$(13)

其中 $\lambda$ 调节正则项、平方和两者之间的比例。

将其对 $\mathrm{\mathbf{w}}$：

$$
\begin{aligned}
\frac{\partial\widetilde{E}(\mathrm{\mathbf{w}})}{\partial\mathrm{\mathbf{w}}}
  &= \mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}} - \mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}} + \frac{\lambda}{2}\frac{\partial(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{w}})}{\partial \mathrm{\mathbf{w}}}
\end{aligned}
$$

这里同样需要用到 [Matrix calculus | Wikipedia] 中的矩阵求导的一个 (scalar-by-vector) 公式：

$$
\frac{\partial(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{w}})}{\partial \mathrm{\mathbf{w}}} = 2\mathrm{\mathbf{w}}^T
$$(14)

于是得到：

$$
\begin{aligned}
\frac{\partial\widetilde{E}(\mathrm{\mathbf{w}})}{\partial\mathrm{\mathbf{w}}} = \mathrm{\mathbf{w}}^T\mathrm{\mathbf{X}}^T\mathrm{\mathbf{X}} - \mathrm{\mathbf{t}}^T\mathrm{\mathbf{X}} + \lambda\mathrm{\mathbf{w}}^T
\end{aligned}
$$(15)

令导数为 $0$，解得：

$$
\mathrm{\mathbf{w_{ML}}} = (\mathrm{\mathbf{X}}^{T}\mathrm{\mathbf{X}} + \lambda\mathrm{\mathbf{I}}_{(M+1)(M+1)})^{-1}(\mathrm{\mathbf{X}}^{T}\mathrm{\mathbf{t}})
$$(16)

总结：带惩罚项的最小二乘法解析解与普通最小二乘法相比，相差不大，只是在左边乘积项中多了一个单位矩阵的 $\lambda$ 倍。

#### 梯度下降法

说到梯度下降法，我们可以先来看看**导数下降法**。对于下面的函数 $f$：

<img src="https://ctmakro.github.io/site/on_learning/gd_plot_2.svg" style="float:left; padding-right:30px">

<div style="margin-top:60px">

我们可以用这种方法求解 $f(x) = 0$ 的解：
    
  1. 初始化 $x = 0$ （任意值都可以），给定精确度 $\alpha = 0.2$
  2. $f'(0) < 0$，令 $x = x + \alpha = 0.2$
  3. $f'(0.2) < 0$，令 $x = x + \alpha = 0.4$
  4. $f'(0.4) > 0$，令 $x = x + \alpha = 0.6$
  5. $f'(0.6) > 0$，令 $x = x - \alpha = 0.4$
</div>

<div style="clear:both">

可以看到，按照这个计算过程 $x$ 会一直在 $0.4$ 和 $0.6$ 两个数之间变化。实际计算时应该减去导数值与 $\alpha$（学习率）的乘积，如果我们将 $\alpha$ 取的足够小，并让这个迭代过程在可接受的误差范围内停止的话，我们最终将得到方程的解。当然，$\alpha$ 也不宜太小，否则会导致误差下降缓慢，计算耗时太长。

</div>

**梯度下降法**则是对向量函数进行类似的处理。梯度的方向是函数上升最快的方向，其反方向则是函数下降最快的方向。我们的目的就是找到误差函数 $\widetilde{E}(\mathrm{\mathbf{w}})$ 取最小值的点，我们可以从任意点开始沿着其梯度反方向以学习率为步长进行迭代，最终在误差许可范围内停止。

对误差函数 $\widetilde{E}(\mathrm{\mathbf{w}})$ 求梯度得：

$$
\begin{aligned}
  \nabla \widetilde{E}(\mathrm{\mathbf{w}}) 
  &= \nabla [\frac{1}{2}\left\|
\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{t}}
\right\|^2 + \frac{\lambda}{2}(\mathrm{\mathbf{w}}^T\mathrm{\mathbf{w}})] \\
  &= \mathrm{\mathbf{X}}^T(\mathrm{\mathbf{X}}\mathrm{\mathbf{w}} - \mathrm{\mathbf{t}}) + \lambda \mathrm{\mathbf{w}}
\end{aligned}
$$(17)



## 三、实验结果与分析

#### 最小二乘法

运行命令：

```
$ make least_squares
```

##### N = 40, M = 10

![least-squares-40-10.png](./images/least-squares-40-10.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="100px" data="./training_results/least-squares-40-10.txt"></object>


拟合效果不太好，很多点相距曲线较远。同时相对正弦函数有一些误差。

##### N = 40, M = 20

![least-squares-40-20.png](./images/least-squares-40-20.png)

$\mathrm{\mathbf{w_{ML}}} =$
<object width="100%" height="180px" data="./training_results/least-squares-40-20.txt"></object>


对训练数据拟合度比较高，同时拟合曲线比较符合正弦曲线的轨迹。

##### N = 40, M = 39

![least-squares-40-39.png](./images/least-squares-40-39.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="240px" data="./training_results/least-squares-40-39.txt"></object>

可以看到几乎所有点都被拟合了，但是图像与正弦函数相比有较大的误差，有些地方曲线波动非常大。这就是**过拟合**现象。

##### N = 20, M = 19

![least-squares-20-19.png](./images/least-squares-20-19.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="180px" data="./training_results/least-squares-20-19.txt"></object>

跟上一种情况类似，几乎所有点都被拟合了，但是图像与正弦函数相比有较大的误差，有许多波动大的地方。同样出现了过拟合现象。

### 带惩罚项的最小二乘法

运行命令：

```
$ make least_squares_regularization
```

##### N = 40, M = 10

![least-squares-regularization-40-10.png](./images/least-squares-regularization-40-10.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="100px" data="./training_results/least-squares-regularization-40-10.txt"></object>

对训练数据的拟合效果较差，同时相比于正弦函数误差较大，需要增加多项式系数来更好地拟合。

##### N = 40, M = 20

![least-squares-regularization-40-20.png](./images/least-squares-regularization-40-20.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="160px" data="./training_results/least-squares-regularization-40-20.txt"></object>

拟合效果比较好，有一些点都没有落在拟合曲线上。拟合曲线比较符合正弦的轨迹。

##### N = 40, M = 39

![least-squares-regularization-40-39.png](./images/least-squares-regularization-40-39.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="240px" data="./training_results/least-squares-regularization-40-39.txt"></object>

拟合效果比较好，有一些点都没有落在拟合曲线上。拟合曲线也比较符合正弦的轨迹，但不如 $M = 20$ 时，可以考虑增大 $\lambda$。

##### N = 20, M = 19

![least-squares-regularization-20-19.png](./images/least-squares-regularization-20-19.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="140px" data="./training_results/least-squares-regularization-20-19.txt"></object>

拟合效果很好，基本都落在拟合曲线上。拟合曲线有些许波动，不过比不加正则项要好许多。可以考虑进一步增大 $\lambda$ 来降低误差。

#### 梯度下降法

通过实验，我可算是理解到了梯度下降的本质：不断地调节学习率，既不能太大以确保收敛，又不能太小让计算机能在有效时间内给出结果。

##### N = 4, M = 2

![gradient-descent-4-2.png](./images/gradient-descent-4-2.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="80px" data="./training_results/gradient-descent-4-2.txt"></object>

因为数据点比较少，拟合效果一般，也不太符合正弦曲线的特征。

#### N = 10, M = 3

![gradient-descent-10-3.png](./images/gradient-descent-10-3.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="80px" data="./training_results/gradient-descent-10-3.txt"></object>

拟合效果不太好，也不太符合正弦曲线的特征。

#### N = 10, M = 9

![gradient-descent-10-9.png](./images/gradient-descent-10-9.png)

$\mathrm{\mathbf{w_{ML}}} =$

<object width="100%" height="80px" data="./training_results/gradient-descent-10-9.txt"></object>

拟合效果比较好，同时曲线较好地吻合了正弦曲线。

## 四、结论

1. 最小二乘法需要计算矩阵的逆效率较慢。
2. 最小二乘法中多项式阶数 $M$ 越大，拟合效果越好，但可能会出现过拟合现象，主要的解决办法有：
    + 增加数据数量
    + 采用贝叶斯方法自动调节有效参数数量
    + 加入惩罚项 :heavy_check_mark:
3. 梯度下降法避免了矩阵求逆过程，不过梯度的选取需要反复测试。当学习率较小的时候，运行的时候速度与最小二乘法相比慢得多；当学习率较大的时候，又可能出现不收敛的情况。

## 五、参考文献

1. **[Bishop 2006]** Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006.
2. **[[Least squares | Wikipedia](https://en.wikipedia.org/wiki/Least_squares)]**
3. **[[Linear least squares | Wikipedia](https://en.wikipedia.org/wiki/Linear_least_squares)]**
4. **[[Numpy and Scipy Documentation](https://docs.scipy.org/doc/)]**
5. **[[Matrix calculus | Wikipedia](https://en.wikipedia.org/wiki/Matrix_calculus)]**
6. **[[ctmakro 梯度下降](https://ctmakro.github.io/site/on_learning/gd.html)]** Gradient Descent 梯度下降法
7. **[[Gradient descent | Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)]**
8. **[[Gradient descent](https://en.wikipedia.org/wiki/Gradient)]**
9. **[[MATT NEDRICH 2014](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)]**

## 七、附录：源代码（带注释）
