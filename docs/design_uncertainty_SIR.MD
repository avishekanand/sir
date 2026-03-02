# U-SIR: Uncertainty-Driven Scheduling & Ranking with LLM Feedback

## 1. Overview

We replace user/click feedback with **LLM uncertainty–derived feedback**.

Key ideas:

- A lightweight **Uncertainty Estimator** predicts LLM score and uncertainty.
- A **bandit-style utility** schedules documents for expensive LLM evaluation.
- **LLM feedback (score + uncertainty)** updates posterior relevance estimates.
- Rankings are continuously refined using Bayesian updates (SUNAR-style).

This yields an **active, uncertainty-aware reranking loop**.

---

## 2. Problem Setup

For a query $q$ and candidate documents:

$$
D = \{d_1, \dots, d_N\}
$$

Each document maintains:

- $\mu_i$ — posterior mean relevance  
- $\sigma_i^2$ — posterior variance  
- $n_i$ — number of LLM evaluations  
- $\mathcal{H}_i$ — evaluation history  

Current ranking:

$$
\pi = \text{argsort}_i(-\mu_i)
$$

---

## 3. Components

### 3.1 Uncertainty Estimator (Pre-LLM)

A lightweight model $E$ predicts:

$$
(\hat{\mu}_i, \hat{\sigma}_i) = E(q, d_i)
$$

Inputs may include:

- Query embedding  
- Document embedding  
- Cross-encoder score  
- Prior posterior $(\mu_i, \sigma_i)$  
- Historical LLM signals  

This avoids expensive LLM calls.

---

### 3.2 Utility Function (Bandit Acquisition)

We prioritize documents likely to change ranking.

#### UCB-style acquisition

$$
U_i = \hat{\mu}_i + \alpha \hat{\sigma}_i
$$

where $\alpha$ controls exploration.

#### Rank-Boundary Utility (Top-k Focus)

To stabilize top-$k$ ranking:

$$
U_i = \hat{\sigma}_i \cdot g(\text{rank}_i)
$$

where $g(\cdot)$ emphasizes documents near the decision boundary.

---

### 3.3 Scheduler

Select a batch:

$$
B = \operatorname{TopK}(U_i)
$$

Only $B$ is evaluated by the LLM.

---

### 3.4 LLM Evaluator

For each $d_i \in B$:

$$
(s_i, \sigma^{LLM}_i) = \text{LLM}(q, d_i)
$$

Where:

- $s_i$ = relevance score  
- $\sigma^{LLM}_i$ = uncertainty (token entropy, logit variance, SUNAR head)

We treat this as a noisy observation of true relevance.

---

## 4. Bayesian Posterior Update (Ranking Update)

Assume latent true relevance:

$$
r_i \sim \mathcal{N}(\mu_i, \sigma_i^2)
$$

LLM observation:

$$
s_i \sim \mathcal{N}(r_i, (\sigma^{LLM}_i)^2)
$$

### Posterior Mean

$$
\mu_i' =
\frac{\sigma_i^2}{\sigma_i^2 + (\sigma^{LLM}_i)^2} s_i
+
\frac{(\sigma^{LLM}_i)^2}{\sigma_i^2 + (\sigma^{LLM}_i)^2} \mu_i
$$

### Posterior Variance

$$
\sigma_i'^2 =
\left(
\frac{1}{\sigma_i^2} + \frac{1}{(\sigma^{LLM}_i)^2}
\right)^{-1}
$$

Properties:

- Confident LLM outputs (small $\sigma^{LLM}_i$) shift ranking more.
- Uncertain outputs have limited influence.
- Variance decreases after evaluation.

---

## 5. Ranking Update

After updating posteriors:

$$
\pi \leftarrow \text{argsort}_i(-\mu_i')
$$

Optional instability tracking:

$$
\Delta \text{Rank}_i =
\left|
\text{rank}_{old}(i) - \text{rank}_{new}(i)
\right|
$$

---

## 6. Reformulation Trigger (Optional)

If uncertainty in top-$k$ remains high:

$$
\max_{i \in \text{top-}k} \sigma_i > \tau
$$

Then:

- Reformulate query $q'$
- Expand candidate pool
- Restart scheduling loop

---

## 7. Full Algorithm

```python
Input: query q, documents D, budget B

Initialize posterior μ_i, σ_i using estimator E

while budget not exhausted:

    # Utility computation
    U_i = μ_i + α * σ_i

    # Select documents to evaluate
    S = top_k(U_i)

    for d_i in S:
        s_i, σ_llm = LLM(q, d_i)

        # Bayesian update
        μ_i = (
            (σ_i**2 / (σ_i**2 + σ_llm**2)) * s_i +
            (σ_llm**2 / (σ_i**2 + σ_llm**2)) * μ_i
        )

        σ_i = 1 / (1/σ_i**2 + 1/σ_llm**2)

    rank = sort_by(μ_i)

    if max_topk(σ_i) > τ:
        q = reformulate(q)

return rank
```

---

## 8. Architecture (Logical View)

```
                +------------------+
                |      Query q     |
                +--------+---------+
                         |
                         v
                +------------------+
                | Candidate Store  |
                +--------+---------+
                         |
                         v
      +--------------------------------------+
      | Uncertainty Estimator (μ_i, σ_i)     |
      +----------------+---------------------+
                       |
                       v
              +------------------+
              | Utility Module   |
              | U_i = μ_i+ασ_i   |
              +--------+---------+
                       |
                       v
                +--------------+
                |  Scheduler   |
                +------+-------+
                       |
                       v
                +----------------+
                |   LLM Eval     |
                |  s_i, σ_llm    |
                +--------+-------+
                         |
                         v
                +----------------+
                | Bayesian       |
                | Posterior      |
                | Update         |
                +--------+-------+
                         |
                         v
                     Ranking π
```

---

## 9. Complexity

Without scheduling:

$$
O(N) \text{ LLM calls}
$$

With U-SIR:

$$
O(K), \quad K \ll N
$$

Estimator is cheap; LLM is selectively invoked.

---

## 10. Interpretation

This framework implements:

**Bayesian Active Ranking with LLM Uncertainty Feedback**

It unifies:

- Bandit acquisition  
- Uncertainty calibration  
- Bayesian posterior updates  
- Ranking stabilization under budget constraints  
