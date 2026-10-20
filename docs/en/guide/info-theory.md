---
title: "Information Theory Basics: Entropy / Cross-Entropy / KL Divergence"
translation: synced
---

# Information Theory Basics: Entropy / Cross-Entropy / KL Divergence

> **In one sentence**: SFT's loss is cross-entropy, the core constraint of DPO and RLHF is KL divergence, and distillation is essentially KL minimization—these terms are really one and the same language of "how many bits it takes to describe a distribution." This page strings them together with "bits" in plain words; once you get it, revisiting the loss functions in the main text becomes much clearer.

A great many loss functions across this site are built on these few concepts, yet they are often glossed over as "already known." Instead of piling on formulas, let's first make clear what each quantity actually measures, why its unit is the "bit," and how they relate to one another.

## A unified perspective: information = "how surprising"

Information theory starts from a plain intuition: **the less likely an event is, the more information it carries when it actually happens**.

- "The sun rises as usual tomorrow"—nearly certain, saying it is as good as saying nothing, information ≈ 0.
- "A meteorite lands on our city tomorrow"—extremely improbable, but once it happens the information explodes.

To quantify this intuition: an event with probability $p$ has an **information content (self-information)** defined as

$$
I = \log_2 \frac{1}{p} = -\log_2 p .
$$

We take the $\log$ so that "the information of two independent events happening together = the sum of their individual information" (multiplying probabilities → adding logarithms). When the base is 2, the unit is called the **bit**.

**The real meaning of a bit**: one bit = the uncertainty a single yes/no question can eliminate. An event with probability $p=\tfrac{1}{2}$ has exactly $-\log_2\tfrac12=1$ bit of information—guessing heads or tails takes exactly one yes/no question. An event with probability $\tfrac{1}{8}$ is 3 bits, because it takes 3 yes/no questions to pin it down (1 out of $2^3=8$).

> In machine learning, loss functions often use the natural logarithm $\ln$ (base $e$), in which case the unit is the **nat**. Bits and nats differ only by the constant $\ln 2 \approx 0.693$; the conversion changes no optimization conclusion, so in the main text $\log$ does not specify a base.

## Entropy: how uncertain a distribution is "on average"

A single event has information content, so how much information does **an entire probability distribution** carry on average? That is the **entropy**—the expectation of self-information weighted by probability:

$$
H(p) = \mathbb{E}_{x\sim p}\big[-\log p(x)\big] = -\sum_x p(x)\log p(x).
$$

Intuitively, entropy measures **how "messy" / how hard to guess** the distribution is:

- A fair coin, $p=(0.5, 0.5)$, entropy = 1 bit (most uncertain, hardest to guess).
- A biased coin, $p=(0.99, 0.01)$, entropy ≈ 0.08 bits (almost always heads, easy to guess).
- A coin that always lands heads, $p=(1, 0)$, entropy = 0 (no suspense, zero information).

**From an engineering angle**: entropy is "with the optimal coding scheme, the minimum number of bits needed on average to store each sample." The more certain a distribution, the more compressible, and the lower its entropy.

## Cross-entropy: how many extra bits it costs to encode with the "wrong" distribution

Now two distributions appear:

- $p$: the **true distribution** (the actual distribution of tokens in the data; during training it is the one-hot label).
- $q$: the **model's predicted distribution** (the softmax probabilities output by $\pi_\theta$).

**Cross-entropy** measures: **the true data comes from $p$, but you encode it according to $q$'s "belief"; how many bits does each sample cost on average**:

$$
H(p, q) = \mathbb{E}_{x\sim p}\big[-\log q(x)\big] = -\sum_x p(x)\log q(x).
$$

Note the distinction: the expectation is taken over the true distribution $p$ (how events actually occur), but each event's "code length" uses the model's $q$ (how surprising you think it is). The more accurately the model guesses ($q$ closer to $p$), the smaller the cross-entropy.

**This is precisely SFT's loss**. During training the true label is one-hot (the correct token has probability 1, the rest 0), so the cross-entropy $-\sum_x p(x)\log q(x)$ leaves only the correct-token term $-\log q(\text{correct token})$, and the per-token cross-entropy loss reduces to the negative log-likelihood (NLL):

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x, y) \sim \mathbb{D}} \left[ \sum_{t=1}^{\lvert y \rvert} \log \pi_\theta\big(y_t \mid x, y_{<t}\big) \right].
$$

Minimizing cross-entropy = making the model assign as high a probability as possible to the correct token. See [SFT Overview](/en/sft/).

## KL divergence: how far apart two distributions are

Cross-entropy actually mixes two parts: the data's inherent uncertainty (entropy $H(p)$, independent of the model and unchangeable), plus the "extra overhead" from the model guessing inaccurately. Pulling out the latter on its own gives the **KL divergence (Kullback–Leibler divergence, relative entropy)**:

$$
\mathrm{KL}(p \,\|\, q) = \mathbb{E}_{x\sim p}\!\left[\log \frac{p(x)}{q(x)}\right] = \sum_x p(x)\log\frac{p(x)}{q(x)}.
$$

It measures "**how many bits per sample are wasted on average by passing off $q$ as $p$**." Three key properties:

1. **Non-negative**: $\mathrm{KL}(p\|q)\ge 0$, and equals 0 if and only if $p=q$. The more alike the two distributions, the smaller the KL.
2. **Asymmetric**: $\mathrm{KL}(p\|q)\neq\mathrm{KL}(q\|p)$. That is why it is called a "divergence," not a "distance," and you must keep straight which one comes first.
3. **It is exactly cross-entropy minus entropy**—the core identity that ties all three together.

## Tying it together: a one-line identity

$$
\underbrace{H(p, q)}_{\text{cross-entropy}} = \underbrace{H(p)}_{\text{entropy of the true distribution}} + \underbrace{\mathrm{KL}(p \,\|\, q)}_{\text{model bias}}.
$$

In plain words:

> **The total cost of encoding the true data with model $q$ (cross-entropy) = the data's intrinsic, incompressible cost (entropy) + the extra penalty paid because the model is inaccurate (KL divergence).**

This explains why these losses look the way they do:

- **Why does simply minimizing cross-entropy work when training SFT?** Because $H(p)$ is intrinsic to the data, a constant independent of the parameters $\theta$, so minimizing the cross-entropy $H(p,q)$ is exactly equivalent to minimizing $\mathrm{KL}(p\|q)$—that is, making the model distribution approach the data distribution.
- **Why does distillation use KL rather than cross-entropy?** In distillation the "true distribution" is the teacher model's soft labels (not one-hot), so $H(p)$ is no longer a constant; writing it directly as $\mathrm{KL}(\text{teacher}\|\text{student})$ is cleaner and clearer in meaning: make the student distribution match the teacher distribution. See [Distillation Overview](/en/distillation/).

## Mutual information: how much two variables share

The quantities above all compare **two distributions** (truth vs. model, teacher vs. student). There is another question with a different shape: **how much are two random variables related**—once you know $Y$, how much of the uncertainty about $X$ goes away? The quantity that measures this is **mutual information**.

First, one more piece: the **conditional entropy** $H(X \mid Y)$, i.e. how much uncertainty about $X$ remains once $Y$ is known (averaged over all possible $y$ by probability):

$$
H(X \mid Y) = \mathbb{E}_{y \sim p(y)}\big[H(X \mid Y = y)\big] = -\sum_{x, y} p(x, y)\log p(x \mid y).
$$

Mutual information is simply the difference between the entropy "before knowing $Y$" and "after knowing $Y$":

$$
I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y).
$$

In one sentence: **how many bits of average surprise about $X$ are removed once you know $Y$**. It also has an equivalent form that connects straight back to the previous section—how far the joint distribution is from "pretending the two are independent":

$$
I(X; Y) = \mathrm{KL}\big(p(x, y) \,\|\, p(x)\,p(y)\big).
$$

In other words, **mutual information is just a special case of KL divergence**, from which its properties follow directly:

1. **Non-negative**: $I(X;Y)\ge 0$, and $=0$ if and only if $X$ and $Y$ are independent (then $p(x,y)=p(x)p(y)$ and the KL is 0)—no relationship means "knowing it makes you no wiser."
2. **Symmetric**: $I(X;Y)=I(Y;X)$. Unlike KL, where you must keep straight which comes first.
3. **Bounded**: $I(X;Y)\le\min\{H(X),\,H(Y)\}$. When $Y$ fully determines $X$, $H(X\mid Y)=0$ and the mutual information reaches $H(X)$.
4. **Conditional version** $I(X;Y\mid Z)$: given that you already know $Z$, how much more $Y$ can still tell you about $X$. That is what the example below uses.

**A language-model example**: let $X$ be the prompt and $Y$ the response; $I(X;Y)$ is then "how many bits of uncertainty about the answer disappear once you see the question." A model that ignores the prompt entirely has zero mutual information; the more tightly the prompt constrains the answer (a math problem with a single correct answer, say), the larger it gets. It asks a different question than perplexity: PPL measures how much uncertainty is left, mutual information measures how much of it $X$ explains away.

**A place it genuinely matters: information leakage in self-distillation.** Self-distillation (OPSD) in [OPD](/en/opd/) is the cleanest example: the teacher is "yourself, having peeked at the reference answer $r$," while the student cannot see $r$. Theorem 1 of RLSD (*Self-Distilled RLVR*, arXiv:2604.03128) decomposes the OPSD loss into exactly the same shape as the identity on this page:

$$
\mathcal{L}_{\text{OPSD}} = \underbrace{\mathcal{L}^{*}}_{\text{what should be learned: match the teacher's marginal}} + \underbrace{I(Y_t;\, R \mid X,\, Y_{<t})}_{\text{what cannot be learned: extra information from the reference}}.
$$

That second term is precisely a **conditional mutual information**: given the prompt and the prefix generated so far, how much extra information the reference answer still carries about the next token. It is determined entirely by the teacher's distributions and contains no student parameter $\theta$, so no amount of training can remove it—it is a strictly positive lower bound on the loss. The paper further shows that while the expected gradient is benign, per-sample gradients carry an $r$-specific deviation whose variance grows monotonically with this mutual information; an optimizer walking sample by sample along those deviations can only end up encoding $x \to r$ correlations into the parameters—which shows up as the model citing a reference answer it never had access to. See [OPD Overview](/en/opd/).

Mutual information also shows up in contrastive learning (the InfoNCE loss is a lower bound on it), in RL exploration as an information-gain intrinsic reward, and in data selection as a measure of redundancy between samples.

## Perplexity: the "exponential version" of cross-entropy

The **perplexity (PPL)** you hear about most often when evaluating language models is not actually anything new—it is just cross-entropy exponentiated:

$$
\text{PPL}(p, q) = e^{H(p, q)} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N}\log q(y_t \mid y_{<t})\right).
$$

Why bother taking an exponent? Because it **switches to a more intuitive unit**:

- **Cross-entropy** lives on a logarithmic scale, with units of bits / nats, answering "how many bits each token costs on average";
- **Perplexity** exponentiates back to a linear scale, answering "at each step, the model is hesitating as if among **how many equally likely options**"—i.e., the "effective branching factor."

For example: a model with PPL = 20 on a test set means that its uncertainty when predicting the next word is **equivalent to choosing among 20 equally likely words at each step**. Lower is better: a perfect model has PPL = 1 (certain at every step, not perplexed at all), and a purely random 1-out-of-$N$ model has PPL = $N$.

So the several quantities on this page form a chain: **self-information → entropy / cross-entropy (take the expectation) → perplexity (then take the exponent)**. The three correspond monotonically, so minimizing SFT's cross-entropy loss is exactly equivalent to minimizing perplexity—which is why PPL is the most common intrinsic evaluation metric for language models.

## It is everywhere in post-training

KL divergence shows up repeatedly in the alignment stage, always playing the same role: **acting as a "don't stray too far" rein**—optimizing the objective while forbidding the new model from drifting too far from the reference model $\pi_{\text{ref}}$, to avoid collapse or reward hacking.

| Where it appears | What KL is doing |
| --- | --- |
| [SFT](/en/sft/) | The loss is cross-entropy / NLL, equivalent to minimizing $\mathrm{KL}(\text{data}\|\pi_\theta)$ |
| [PPO / RLHF](/en/rlhf/ppo) | A $-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ penalty is added to the reward, keeping the policy from straying too far from the reference model |
| [DPO](/en/dpo/dpo) | The $\beta$ in the preference loss is exactly the KL constraint strength; the log-probability ratio in the formula is the manifestation of the KL term |
| [Distillation](/en/distillation/) | Directly minimize the KL between the student's and teacher's output distributions |

This also explains why the [Notation Conventions](/en/guide/notation) define $\beta$ uniformly as "KL constraint strength / preference-optimization temperature coefficient"—across different algorithms it tunes the tightness of the same rein.

## A quick-reference table

| Quantity | Formula | One-line meaning |
| --- | --- | --- |
| Self-information | $-\log p(x)$ | How "surprising" a single event is, in bits |
| Entropy $H(p)$ | $-\sum_x p\log p$ | How uncertain a distribution is on average / minimum bits to store it |
| Cross-entropy $H(p,q)$ | $-\sum_x p\log q$ | Bits needed on average to encode data from $p$ using $q$ |
| KL divergence $\mathrm{KL}(p\|q)$ | $\sum_x p\log\frac{p}{q}$ | Bits wasted by passing off $q$ as $p$ (≥0, asymmetric) |
| Conditional entropy $H(X\mid Y)$ | $-\sum_{x,y} p(x,y)\log p(x\mid y)$ | Uncertainty about $X$ left once $Y$ is known |
| Mutual information $I(X;Y)$ | $H(X)-H(X\mid Y)=\mathrm{KL}(p(x,y)\|p(x)p(y))$ | Bits of uncertainty about $X$ removed by knowing $Y$ (≥0, symmetric) |
| Perplexity $\text{PPL}$ | $e^{H(p,q)}$ | Cross-entropy exponentiated; how many "options" the model hesitates among on average |
| Core relationship | $H(p,q)=H(p)+\mathrm{KL}(p\|q)$ | Total cost = intrinsic cost + model bias |

Once you understand this table, revisiting the loss functions of [SFT](/en/sft/), [DPO](/en/dpo/dpo), [PPO](/en/rlhf/ppo), and [Distillation](/en/distillation/) will no longer leave you stuck on words like "cross-entropy" and "KL."
