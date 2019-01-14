##**Bellman方程**

- 定义value fuction，期望的回报越高，价值越大

- 这里的回报指的是多种动作对应的reward的期望值（？）

- $$
  \begin{split}
  v(s) &= E(G_t|S_t=s)\\
  &=E(R_{t+1} + \lambda R_{t+2} + \lambda^2R_{t+3} + ...|S_t = s)\\
  &=E(R_{t+1} + \lambda v(S_{t+1})\ |\ S_t = s)
  \end{split}
  $$

- value fuction可以迭代计算得到



## Action-Value functio动作价值函数

- 用reward进行定义，该reward指的是执行完某个动作得到的回报。

![image-20181218091234938](/Users/pp/pp_git/typora_images/image-20181218091234938-5095554.png)

- $\pi$为当前的策略，该公式表示的是当前动作执行之后得到的reward与之后的动作产生的reward的和的期望。



## Optimal value function 最优价值函数

- value based：我们需要找到最优策略，现在求解最优策略等价于求解最优的value function，找到最优的value function
- 最优价值函数：所有策略下，动作价值函数的最大值。

$$
\begin{split}
  Q^*(s,a) &= max_{\pi}Q^{\pi}(s,a)\\
  &=E(r\ +\lambda max_{a'} Q^*(s^{'},a^{'})\ |s,a )
  \end{split}
$$

##value iteration

- 通过bellman方程得到

![image-20181218093204101](/Users/pp/pp_git/typora_images/image-20181218093204101-5096724.png)

- 算法

![image-20181218093705997](/Users/pp/pp_git/typora_images/image-20181218093705997-5097026.png)

- 理解：每次用最优的方式更新value，最终得到一个最优的policy。
- ![image-20181218094617933](/Users/pp/pp_git/typora_images/image-20181218094617933-5097577.png)

- 由于没有办法遍历所有状态和动作，只能对有限的样本进行操作。

  一种新的更新Q值的方法：类似梯度下降，每次

![image-20181218094916821](/Users/pp/pp_git/typora_images/image-20181218094916821-5097756.png)

## DQN

### 价值函数近似Value Function Approximation

使用函数来近似表示Q价值函数：
$$
Q(s,a)=f(s,a)
$$

- 神经网络训练：
  - loss function：让估计值与目标值越来越接近。

![image-20181218101135460](/Users/pp/pp_git/typora_images/image-20181218101135460-5099095.png)

- Experience Replay