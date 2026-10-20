---
title: 概率论基础：LLM 里真正用到的那些
---

# 概率论基础：LLM 里真正用到的那些

> **一句话**：RL 的目标写成"期望"、GRPO 对一个 prompt 采一组样本是"蒙特卡洛估计"、PPO 里的 ratio 是"重要性采样"、SFT 的损失是"极大似然"——这些词在正文里天天出现。本页把它们一次讲清楚，每个概念都挂到站内真正用到它的地方。
>
> 读完这页再去看 [信息论基础](/guide/info-theory)（熵 / KL / 互信息）与 [统计学基础](/guide/ab-testing)（置信区间 / 显著性），三页正好是一条线：**概率描述随机，信息论度量不确定性，统计从样本反推真值**。

本站二十多个页面在用"方差""重要性采样""蒙特卡洛"这些词，但它们从没在任何一页被定义过。这一页补上这个空缺。不堆公式，只讲每个概念在 LLM 训练里对应的是哪件事。

## 一、随机变量与分布：LLM 本身就是一个条件分布

**随机变量**就是"结果还没定下来的那个量"，而**分布**描述它取各个值的概率。LLM 里最重要的随机变量只有一个：下一个 token。

生成第 $t$ 个 token 时，模型吐出的是一组实数 **logits** $z \in \mathbb{R}^{|\mathcal{V}|}$（词表每个词一个分数，不是概率）。softmax 把它变成一个合法分布：

$$
\pi_\theta(v \mid x, y_{<t}) = \frac{\exp(z_v / T)}{\sum_{v' \in \mathcal{V}} \exp(z_{v'} / T)}, \qquad \sum_{v} \pi_\theta(v \mid x, y_{<t}) = 1 .
$$

所谓"模型"，本质上就是这个**条件分布** $\pi_\theta(\cdot \mid x, y_{<t})$——给定上文，词表上每个词有多大机会。训练在改这个分布，解码在从这个分布里取样。

温度 $T$ 和截断策略都只是在**改分布的形状**：

| 做法 | 对分布做了什么 | 极端情况 |
| --- | --- | --- |
| 温度 $T$ | logits 除以 $T$ 再归一化 | $T \to 0$ 退化成 $\arg\max$（贪心）；$T$ 越大越平、[熵](/guide/info-theory)越高 |
| top-k | 只留概率最高的 $k$ 个词，重新归一化 | $k=1$ 即贪心 |
| top-p（核采样） | 按概率从高到低累加到 $p$ 为止，截断后归一化 | $p=1$ 即原分布 |

> 工程上两个常识：一是永远在 **log 空间**算（`log_softmax` 先减去最大值，避免 $\exp$ 溢出）；二是训练日志里的 logprob 指的就是 $\log \pi_\theta(y_t \mid x, y_{<t})$，RL 的各种比值都建立在它上面。

## 二、条件概率与链式法则：自回归分解是怎么来的

条件概率 $p(b \mid a) = p(a, b) / p(a)$ 变形一下就是**链式法则** $p(a, b) = p(a)\,p(b \mid a)$。把它反复套在一个 token 序列上，就得到了整条回答的概率：

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{|y|} \pi_\theta(y_t \mid x, y_{<t}) .
$$

这不是某种近似，而是**恒等式**——自回归模型"只预测下一个词"却能给整句话打分，靠的就是它。取对数后乘积变求和：

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) .
$$

几百个介于 0 和 1 之间的数相乘会直接下溢到 0，而相加不会——这是训练和打分一律在 log 空间进行的第一个理由；第二个理由是求和对每一项的梯度互不干扰，可以整条序列并行算。符号体系见 [符号约定](/guide/notation)。

还有一个常被忽略的操作：**边缘化**（把不关心的变量按概率加掉）

$$
\bar{p}(y_t) = \sum_r p(y_t \mid r)\, p(r) .
$$

[OPD 自蒸馏](/opd/)里"老师的边缘分布"说的就是这个——把老师在各种参考答案 $r$ 下的预测按 $r$ 的概率平均掉，得到一个不依赖 $r$ 的分布。学生能学到的上限正是这个边缘分布，剩下的差额是[互信息](/guide/info-theory#互信息-两个变量之间共享了多少信息)。

## 三、期望与方差：RL 的目标，和它的噪声

**期望**是"按概率加权的平均值"，**方差**是"偏离这个平均值有多厉害"：

$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_x p(x) f(x), \qquad \mathrm{Var}[f] = \mathbb{E}\big[(f - \mathbb{E}[f])^2\big] .
$$

两条每天都在用的性质：期望是**线性**的（$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$，不要求独立）；方差不是（$\mathrm{Var}[aX] = a^2 \mathrm{Var}[X]$，独立时才能相加）。

所有 RL 算法的目标都写成一个期望（见 [RLHF 总览](/rlhf/)）：

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[r(x, y)\big] .
$$

注意这里的 $y \sim \pi_\theta$：**期望是对模型自己的输出分布取的**，而这个分布随参数变化。它没有闭式解，只能采样估计——这正是 RL 比 SFT 贵得多的根源。梯度靠**对数求导技巧**（REINFORCE，Williams, 1992）拿到：

$$
\nabla_\theta \, \mathbb{E}_{y \sim \pi_\theta}\big[r(y)\big]
= \sum_y r(y)\, \nabla_\theta \pi_\theta(y)
= \sum_y \pi_\theta(y)\, r(y)\, \frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)}
= \mathbb{E}_{y \sim \pi_\theta}\big[\, r(y)\, \nabla_\theta \log \pi_\theta(y) \,\big] .
$$

一行推导，三个要点：**奖励不需要可导**（它只是个权重）；更新方向永远是 $\nabla\log\pi_\theta$，即"把采到的这条回答的概率推高或压低"；推高多少由 $r$ 决定。从 [PPO](/rlhf/ppo) 到 [GRPO](/rlhf/grpo) 到 [CISPO](/rlhf/cispo)，改的全是这个系数（即[后训练总览](/post-training/)里的梯度系数 $\mathrm{GC}$），骨架没变过。

## 四、蒙特卡洛估计：为什么 GRPO 要"采一组"

期望算不出来，就用样本平均代替——这就是**蒙特卡洛（Monte Carlo）估计**：

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} f(x_i), \qquad \mathbb{E}[\hat{\mu}] = \mu, \qquad \mathrm{Var}[\hat{\mu}] = \frac{\sigma^2}{n} .
$$

它**无偏**（平均意义上不偏不倚），标准误是 $\sigma / \sqrt{n}$。$\sqrt{n}$ 这个分母是整页最该记住的事实：

> **想把误差减半，要采 4 倍的样本；减到 1/10，要 100 倍。**

站内处处是它的后果：

- [GRPO](/rlhf/grpo) 对同一个 prompt 采一组 $G$ 条回答，用组内均值当基线——这是对"该 prompt 的期望回报"的蒙特卡洛估计，$G$ 太小基线就噪声大。
- 评测里 temperature > 0 时，**一次跑分本身就是一个随机变量**。AIME 这类只有 30 题的基准，单次分数抖动可以到十几个点，所以现在普遍报 avg@k（跑 $k$ 次取平均）而不是单次成绩，细节见 [评测分数的误差棒](/guide/ab-testing#十、离线评测分数的误差棒)。
- [test-time scaling](/reasoning/test-time-scaling) 里的多数投票（self-consistency），本质也是用多次采样把单次解码的方差压下去。

## 五、基线与方差缩减：减掉一个常数，为什么答案不变

策略梯度无偏但方差极大。标准解法是从奖励里减一个**基线** $b$（与当前动作无关即可）。它不改变期望，因为：

$$
\mathbb{E}_{y \sim \pi_\theta}\big[b\, \nabla_\theta \log \pi_\theta(y)\big]
= b \sum_y \pi_\theta(y) \frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)}
= b\, \nabla_\theta \sum_y \pi_\theta(y)
= b\, \nabla_\theta 1 = 0 .
$$

概率加起来恒等于 1，对常数求导为零——所以 $r$ 换成 $r - b$ **一定无偏**，而只要 $b$ 接近 $\mathbb{E}[r]$，方差就能大幅下降。整个 PPO 之后的算法演化史，有一半是在回答"$b$ 取什么"：

| 算法 | 基线 $b$ 怎么来 | 代价 |
| --- | --- | --- |
| [PPO](/rlhf/ppo) | 训一个 critic 估计状态价值 $V(s)$ | 多一个模型要训、要占显存 |
| [GRPO](/rlhf/grpo) | 同一 prompt 采一组，用组内均值 | 多采 $G$ 条样本 |
| [RLOO](/rlhf/rloo) | 留一法：其余 $k-1$ 条的均值 | 同上，但估计更干净 |
| [REINFORCE++](/rlhf/reinforce-plus-plus) | 全局 batch 的均值 | 最省，但基线更粗 |

一个细节：GRPO 在减去组内均值之后还会**除以组内标准差**。这一步不再是"减常数"那种无偏变换，它会给优势估计引入偏差——属于拿一点偏差换方差的工程取舍，和上表那些严格无偏的基线不是一回事。

## 六、重要性采样：用旧策略的样本更新新策略

训练时常常遇到：样本是用策略 $q$ 采的，但想算的是策略 $p$ 下的期望。**重要性采样（importance sampling）**给出等式：

$$
\mathbb{E}_{x \sim p}\big[f(x)\big] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right], \qquad \text{要求 } q(x) > 0 \ \text{ 凡 } p(x) > 0 .
$$

权重 $w = p/q$ 的作用是"把按 $q$ 采到的样本掰回 $p$ 的口径"。RL 里 $q = \pi_{\text{old}}$（这批数据被采出来时的策略），$p = \pi_\theta$（已经更新过几步的当前策略），于是出现那个熟悉的比值：

$$
\rho_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})} .
$$

这就是为什么一批数据能做多轮更新（见 [训练循环机制](/rlhf/training-loop)）。但重要性采样有个硬毛病：**$p$ 和 $q$ 差得越远，权重分布越重尾，方差越大**，极端时少数几个样本的权重主导了整个梯度。衡量"还剩多少有效样本"的常用指标是有效样本量

$$
\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2} \le n .
$$

压住这个方差的办法就是裁剪，而站内三条路线正是裁在了不同位置：

| 算法 | 裁什么 | 副作用 |
| --- | --- | --- |
| [PPO](/rlhf/ppo) / [GRPO](/rlhf/grpo) | token 级裁 $\rho_t$ | 被裁掉的 token 直接失去梯度 |
| [GSPO](/rlhf/gspo) | 序列级的重要性比 | 粒度粗，但避开了 token 级比值的噪声 |
| [CISPO](/rlhf/cispo) | 裁 IS 权重本身、保留全部 token 的梯度 | 低概率的反思 token 不再被踢出更新 |

## 七、贝叶斯：先验、后验与"证据比"

贝叶斯公式把"看到证据后怎么更新信念"写成一行：

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)} \;\propto\; \underbrace{p(D \mid \theta)}_{\text{似然}} \cdot \underbrace{p(\theta)}_{\text{先验}} .
$$

LLM 训练里最直白的一个应用在 [OPD 的自蒸馏](/opd/)：RLSD 让同一个模型分别以"看不到参考答案"（学生，先验）和"看到参考答案"（老师，后验）的身份给同一个 token 打分，两者的比值

$$
w_t = \frac{P_T(y_t)}{P_S(y_t)}
$$

被称为 **evidence ratio**——它量化的正是"这条证据让我对这个 token 的信念变强了多少倍"，论文附录给出了它作为一次贝叶斯信念更新的正式推导。注意它和上一节的重要性比值在**结构上完全相同**（都是两个分布在同一点的概率比），用途却不同：一个控制更新步长，一个控制功劳在 token 之间怎么分配。

## 八、极大似然：预训练和 SFT 在做同一件事

**极大似然估计（MLE）**：挑一组参数，让已经观察到的数据在模型下的概率最大。

$$
\hat{\theta} = \arg\max_\theta \sum_{(x, y) \in \mathcal{D}} \log \pi_\theta(y \mid x)
= \arg\min_\theta \Big( - \sum_{(x,y)} \sum_t \log \pi_\theta(y_t \mid x, y_{<t}) \Big) .
$$

右边这个负对数似然（NLL）就是 [SFT 的损失](/sft/)，也是预训练的损失——两者的区别只在数据和哪些 token 参与算 loss（见 [Loss Masking](/sft/loss-masking)）。而 NLL 等价于[交叉熵](/guide/info-theory)，交叉熵取指数就是困惑度：**"最大化似然""最小化交叉熵""降低 PPL"是同一件事的三种说法**。

## 九、大数定律与中心极限定理：样本均值能当真值用吗

两条定理回答了"用有限样本估计期望，到底靠不靠谱"：

- **大数定律（LLN）**：$n \to \infty$ 时样本均值 $\hat{\mu}$ 收敛到真实期望 $\mu$。保证了"采得够多就准"。
- **中心极限定理（CLT）**：不管原分布长什么样（只要方差有限），样本均值近似服从正态分布 $\hat{\mu} \sim \mathcal{N}(\mu,\ \sigma^2/n)$。

CLT 才是实用的那条：它意味着**误差的形状是已知的**，于是可以给任何一个平均值配上区间——"$\hat{\mu} \pm 1.96 \times$ 标准误"覆盖真值的概率约 95%。评测分数的误差棒、A/B 实验的置信区间，都是从这里长出来的，接着读 [统计学基础](/guide/ab-testing)。

## 一张速查表

| 概念 | 一句话 | 站内哪里用 |
| --- | --- | --- |
| 条件分布 $\pi_\theta(\cdot\mid x,y_{<t})$ | 模型本身就是它 | 所有损失函数 |
| 链式法则 | 整句概率 = 各 token 条件概率连乘 | 自回归生成、[符号约定](/guide/notation) |
| 期望 $\mathbb{E}$ | 按概率加权的平均 | [RL 目标](/rlhf/) |
| 方差 $\mathrm{Var}$ | 偏离均值的程度，即梯度噪声 | 基线、裁剪、采样预算 |
| 对数求导技巧 | $\nabla\mathbb{E}[r]=\mathbb{E}[r\nabla\log\pi]$ | 一切策略梯度算法 |
| 蒙特卡洛估计 | 样本均值代替期望，误差 $\sigma/\sqrt{n}$ | [GRPO](/rlhf/grpo) 采组、avg@k 评测 |
| 基线 $b$ | 减与动作无关的常数：无偏、降方差 | [PPO](/rlhf/ppo) / [GRPO](/rlhf/grpo) / [RLOO](/rlhf/rloo) |
| 重要性采样 $p/q$ | 用 $q$ 的样本算 $p$ 的期望 | [PPO](/rlhf/ppo) 的 ratio、[GSPO](/rlhf/gspo)、[CISPO](/rlhf/cispo) |
| 贝叶斯 / 证据比 | 看到证据后信念变了多少倍 | [RLSD 自蒸馏](/opd/) |
| 极大似然 | 让观察到的数据概率最大 | [SFT](/sft/)、预训练 |
| CLT | 样本均值近似正态，误差可量化 | [置信区间](/guide/ab-testing) |

## 参考

- Williams, 1992. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.*（REINFORCE 与对数求导技巧）
- Sutton & Barto, 2018. *Reinforcement Learning: An Introduction*, 2nd ed.（期望、基线与方差缩减）
- Schulman et al., 2015. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438（偏差—方差权衡）
- Owen, 2013. *Monte Carlo Theory, Methods and Examples.*（蒙特卡洛与重要性采样、ESS）
- Bishop, 2006. *Pattern Recognition and Machine Learning.*（概率建模与贝叶斯）
