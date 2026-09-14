# Trellis-Based Working Language Analysis for a Single Transformer Layer

## Problem Interpretation

Given logit lens predictions at a fixed layer $j$ across $N$ token positions, each position $i$ yields a top-$K$ set of predicted tokens with associated probabilities and per-token language distributions. The existing approach (see `notes.md`) computes a **marginal** language distribution at each position independently:

$$P^*(L_a \mid j, i) = \frac{\sum_{k=1}^K w(t_k^{(i)}) \cdot P(L_a \mid t_k^{(i)}) \cdot p(t_k^{(i)} \mid j, i)}{\sum_{k=1}^K w(t_k^{(i)}) \cdot p(t_k^{(i)} \mid j, i)}$$

This marginal ignores **sequential coherence**: if position $i$ is strongly English-dominant, position $i+1$ is more likely to also be English (because natural language exhibits local consistency). The trellis formulation captures this by modeling transitions between adjacent positions based on **language similarity** between their respective top-$K$ tokens.

**Problem type:** Probabilistic inference on a structured (trellis) graphical model — specifically, a Hidden Markov Model (HMM) variant where hidden states are token identities and emissions are lens probabilities.

---

## Notation and Definitions

### Sets and Indices

| Symbol | Definition |
|--------|-----------|
| $N \in \mathbb{Z}^+$ | Number of token positions |
| $K \in \mathbb{Z}^+$ | Number of top-$K$ predicted tokens per position |
| $M \in \mathbb{Z}^+$ | Number of languages |
| $i \in \{0, 1, \ldots, N-1\}$ | Position index |
| $k \in \{0, 1, \ldots, K-1\}$ | Token (state) index within a position |
| $m \in \{0, 1, \ldots, M-1\}$ | Language index |

### Data Structures (per fixed layer $j$)

**Definition 1 (Lens Probability Vector).** For each position $i$, the lens probability vector is:

$$\mathbf{p}_i = \big(p_i[0],\; p_i[1],\; \ldots,\; p_i[K-1]\big)^T \in \Delta^{K-1}$$

where $p_i[k] = p(t_k^{(i)} \mid j, i)$ is the normalized lens probability of the $k$-th predicted token at position $i$, layer $j$. Here $\Delta^{K-1} = \{\mathbf{x} \in \mathbb{R}^K_{\geq 0} : \sum_k x_k = 1\}$ is the probability simplex.

**Definition 2 (Language Matrix).** For each position $i$, the language matrix is:

$$T_i \in \mathbb{R}^{K \times M}, \quad T_i[k, m] = P(L_m \mid t_k^{(i)})$$

where row $k$ is the language distribution of the $k$-th predicted token. Each row satisfies $T_i[k, :] \in \Delta^{M-1}$ (i.e., $\sum_m T_i[k,m] = 1$).

**Definition 3 (Marginal Language Distribution).** The marginal language distribution at position $i$ (the existing approach) is:

$$\boldsymbol{\lambda}_i^{\text{marg}} = \mathbf{p}_i^T T_i \in \Delta^{M-1}$$

This is a simple weighted average of the token language distributions, with no sequential structure.

**Definition 4 (Entropy Weight).** For each token $t_k^{(i)}$ at position $i$, the entropy weight is:

$$w_i[k] = 1 - \frac{H(\mathcal{L} \mid t_k^{(i)})}{\log_2 M}$$

where $H(\mathcal{L} \mid t_k^{(i)}) = -\sum_m T_i[k,m] \log_2 T_i[k,m]$ is the Shannon entropy of the token's language distribution. This weight satisfies $w_i[k] \in [0, 1]$, with $w_i[k] = 1$ for language-unambiguous tokens and $w_i[k] = 0$ for completely ambiguous tokens. The weighted lens vector is $\tilde{\mathbf{p}}_i = \mathbf{w}_i \odot \mathbf{p}_i$ (element-wise product), which can replace $\mathbf{p}_i$ throughout the formulation.

---

## 1. Trellis Structure

**Definition 5 (Language Trellis).** For a fixed layer $j$, the **language trellis** is a layered directed acyclic graph (DAG) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:

- **Nodes (States):** $\mathcal{V} = \{(i, k) : i \in \{0,\ldots,N-1\},\; k \in \{0,\ldots,K-1\}\}$. Each node $(i,k)$ represents the hypothesis that the model's internal "working token" at position $i$ is the $k$-th predicted token.

- **Edges (Transitions):** $\mathcal{E} = \{((i,k), (i+1,k')) : i \in \{0,\ldots,N-2\},\; k,k' \in \{0,\ldots,K-1\}\}$. The graph is **fully connected between adjacent layers** — every state at position $i$ has an edge to every state at position $i+1$.

- **Node weights (Emissions):** Each node $(i,k)$ carries an emission weight $b_i[k] = p_i[k]$ (or $\tilde{p}_i[k] = w_i[k] \cdot p_i[k]$ if entropy weighting is used).

- **Edge weights (Transitions):** Each edge $((i,k),(i+1,k'))$ carries a transition weight $A_i[k, k']$ derived from language similarity (defined in Section 2).

A **path** through the trellis is a sequence $\pi = (k_0, k_1, \ldots, k_{N-1})$ with $k_i \in \{0,\ldots,K-1\}$, representing the hypothesis that the model's working token at position $i$ is token $k_i$.

**Path probability:**

$$P(\pi) = b_0[k_0] \cdot \prod_{i=0}^{N-2} A_i[k_i, k_{i+1}] \cdot b_{i+1}[k_{i+1}]$$

This factorizes as: initial emission $\times$ (transition $\times$ emission) at each step.

---

## 2. Transition Matrix

**Definition 6 (Language Similarity Matrix).** The **language similarity matrix** between positions $i$ and $i+1$ is:

$$S_i \in \mathbb{R}^{K \times K}, \quad S_i[k, k'] = \langle T_i[k,:],\; T_{i+1}[k',:] \rangle = \sum_{m=0}^{M-1} T_i[k,m] \cdot T_{i+1}[k',m]$$

In numpy: `S_i = T_i @ T_{i+1}.T`

**Interpretation.** Since each row of $T_i$ and $T_{i+1}$ is a probability distribution over $M$ languages, $S_i[k,k']$ is the **inner product** of two probability mass functions. This equals the probability that two independent random variables $X \sim \text{Categorical}(T_i[k,:])$ and $Y \sim \text{Categorical}(T_{i+1}[k',:])$ take the same value:

$$S_i[k,k'] = \Pr[X = Y] = \sum_m \Pr[X = m] \cdot \Pr[Y = m]$$

**Properties of $S_i$:**

1. **Non-negativity:** $S_i[k,k'] \geq 0$ for all $k, k'$.
2. **Bounded:** $0 \leq S_i[k,k'] \leq 1$. The maximum $1$ is achieved iff $T_i[k,:] = T_{i+1}[k',:]$ (identical language distributions).
3. **Range for uniform distributions:** If both rows are uniform ($1/M$ each), then $S_i[k,k'] = M \cdot (1/M)^2 = 1/M$.
4. **Diagonal dominance (typical):** When tokens at adjacent positions share language identity, the matrix tends to be diagonally dominant.

**Definition 7 (Transition Matrix).** The **transition matrix** $A_i \in \mathbb{R}^{K \times K}$ between positions $i$ and $i+1$ is obtained by row-normalizing $S_i$:

$$A_i[k, k'] = \frac{S_i[k, k']}{\sum_{k''=0}^{K-1} S_i[k, k'']}$$

In numpy:
```python
S = T_i @ T_next.T          # (K, K)
A = S / S.sum(axis=1, keepdims=True)
```

**Properties of $A_i$:**

1. **Row-stochastic:** $\sum_{k'} A_i[k, k'] = 1$ for all $k$, and $A_i[k, k'] \geq 0$. Thus $A_i$ is a valid Markov transition matrix.
2. **Interpretation:** $A_i[k, k']$ is the probability of transitioning from token $k$ at position $i$ to token $k'$ at position $i+1$, conditioned on language similarity. Tokens with similar language distributions receive higher transition weights.

**Alternative normalization (Softmax):** For sharper transitions, one can use:

$$A_i^{\text{softmax}}[k, k'] = \frac{\exp(S_i[k, k'] / \tau)}{\sum_{k''} \exp(S_i[k, k''] / \tau)}$$

where $\tau > 0$ is a temperature parameter. As $\tau \to \infty$, transitions become uniform; as $\tau \to 0^+$, transitions become deterministic (argmax). The row-normalized version corresponds to $\tau = 1$ without the exponential nonlinearity.

**Alternative similarity measures:**

| Measure | Formula | Range | Properties |
|---------|---------|-------|------------|
| Inner product (default) | $\sum_m T_i[k,m] T_{i+1}[k',m]$ | $[0, 1]$ | Simple, fast, probabilistic interpretation |
| Bhattacharyya coefficient | $\sum_m \sqrt{T_i[k,m] T_{i+1}[k',m]}$ | $[0, 1]$ | Hellinger-related, symmetric |
| Cosine similarity | $\frac{\sum_m T_i[k,m] T_{i+1}[k',m]}{\|T_i[k,:]\|_2 \|T_{i+1}[k',:]\|_2}$ | $[0, 1]$ | Scale-invariant (but distributions are already normalized) |
| Jaccard on supports | $\frac{|\text{supp}(T_i[k]) \cap \text{supp}(T_{i+1}[k'])|}{|\text{supp}(T_i[k]) \cup \text{supp}(T_{i+1}[k'])|}$ | $[0, 1]$ | Discrete, ignores probabilities |

The inner product is recommended as the default due to its clean probabilistic interpretation and efficient vectorized computation.

---

## 3. Forward Algorithm

The forward algorithm computes, for each position $i$ and state $k$, the probability of being in state $k$ at position $i$ given all emissions (lens probabilities) up to and including position $i$, marginalized over all paths through the trellis.

**Definition 8 (Forward Variable).** The forward variable at position $i$ is:

$$\boldsymbol{\alpha}_i \in \mathbb{R}^K, \quad \alpha_i[k] = P(\text{state}_i = k \mid b_0, b_1, \ldots, b_i)$$

where $b_i[k] = p_i[k]$ (or $w_i[k] \cdot p_i[k]$ with entropy weighting).

**Initialization** ($i = 0$):

$$\alpha_0[k] = \frac{b_0[k]}{\sum_{k'} b_0[k']}$$

In numpy: `alpha = p_0 / p_0.sum()`

**Recursion** ($i = 0, 1, \ldots, N-2$):

$$\alpha_{i+1}[k'] = \frac{b_{i+1}[k'] \cdot \sum_{k=0}^{K-1} \alpha_i[k] \cdot A_i[k, k']}{\sum_{k''=0}^{K-1} b_{i+1}[k''] \cdot \sum_{k=0}^{K-1} \alpha_i[k] \cdot A_i[k, k'']}$$

**Derivation.** By the HMM forward decomposition:

$$P(\text{state}_{i+1} = k',\; b_0, \ldots, b_{i+1}) = \underbrace{b_{i+1}[k']}_{\text{emission}} \cdot \underbrace{\sum_k P(\text{state}_i = k,\; b_0, \ldots, b_i) \cdot A_i[k, k']}_{\text{prediction from previous step}}$$

The prediction step computes $\sum_k \alpha_i[k] \cdot A_i[k, k']$, which is the probability of arriving at state $k'$ at position $i+1$ given all emissions up to position $i$. Multiplying by $b_{i+1}[k']$ incorporates the new evidence (the lens probability at position $i+1$). Normalization ensures $\sum_k \alpha_{i+1}[k] = 1$.

**Vectorized form.** Let $\mathbf{b}_{i+1} = (b_{i+1}[0], \ldots, b_{i+1}[K-1])^T$. The recursion is:

$$\boldsymbol{\alpha}_{i+1} = \frac{\mathbf{b}_{i+1} \odot (A_i^T \boldsymbol{\alpha}_i)}{\mathbf{1}^T \big(\mathbf{b}_{i+1} \odot (A_i^T \boldsymbol{\alpha}_i)\big)}$$

where $\odot$ denotes element-wise (Hadamard) product and $A_i^T \boldsymbol{\alpha}_i$ is a matrix-vector product.

In numpy:
```python
pred = A_i.T @ alpha_i          # (K,) — prediction step
alpha_next = b_next * pred       # (K,) — element-wise with emission
alpha_next /= alpha_next.sum()  # normalize
```

**Complexity:** $\mathcal{O}(N \cdot K^2)$ time (dominated by the $K \times K$ matrix-vector product), $\mathcal{O}(N \cdot K)$ space. No Python loops over states — only over positions.

**Relationship to the marginal approach.** The marginal approach uses $\boldsymbol{\alpha}_i^{\text{marg}} = \mathbf{p}_i$ (the lens probabilities directly). The trellis replaces this with the forward variable, which incorporates transition structure:

$$\boldsymbol{\alpha}_i^{\text{trellis}} \propto \mathbf{b}_i \odot \underbrace{(A_{i-1}^T \boldsymbol{\alpha}_{i-1})}_{\text{temporal context}}$$

When $A_i$ is uniform ($A_i[k,k'] = 1/K$ for all $k,k'$), the prediction step gives a constant vector, and the trellis reduces to the marginal (up to normalization). When $A_i$ is strongly diagonal (tokens preserve language identity), the trellis enforces temporal coherence, smoothing out noise in individual lens predictions.

---

## 4. Language-Level Aggregation

### Per-Position Language Probabilities

**Definition 9 (Trellis Language Distribution).** At each position $i$, the trellis language distribution is obtained by projecting the forward state probabilities through the language matrix:

$$\boldsymbol{\lambda}_i = \boldsymbol{\alpha}_i^T T_i \in \Delta^{M-1}$$

where $\lambda_i[m] = \sum_k \alpha_i[k] \cdot T_i[k, m]$ is the probability of language $m$ at position $i$.

In numpy: `lambda_i = alpha_i @ T_i  # (M,)`

**Validity:** Since $\boldsymbol{\alpha}_i \in \Delta^{K-1}$ and each row of $T_i$ is in $\Delta^{M-1}$, the product $\boldsymbol{\alpha}_i^T T_i$ is a convex combination of probability distributions, hence itself a probability distribution: $\sum_m \lambda_i[m] = 1$.

### Layer-Level Working Language

**Definition 10 (Layer Working Language).** The layer's working language distribution is the time-averaged trellis language distribution:

$$\bar{\boldsymbol{\lambda}} = \frac{1}{N} \sum_{i=0}^{N-1} \boldsymbol{\lambda}_i \in \Delta^{M-1}$$

In numpy: `working_lang = lambdas.mean(axis=0)  # (M,)`

**Derived metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Dominant working language | $\hat{m} = \arg\max_m \bar{\lambda}[m]$ | Most probable language across the layer |
| Cross-lingual entropy | $H = -\sum_m \bar{\lambda}[m] \log_2 \bar{\lambda}[m]$ | Language ambiguity (high = multilingual, low = monolingual) |
| Per-position language trajectory | $\boldsymbol{\lambda}_0, \boldsymbol{\lambda}_1, \ldots, \boldsymbol{\lambda}_{N-1}$ | How language identity evolves across positions |
| Language shift (position-level) | $D_{\text{KL}}(\boldsymbol{\lambda}_i \| \boldsymbol{\lambda}_{i-1})$ | KL divergence between adjacent positions |

### Log-Likelihood

The normalization constants from the forward pass yield the trellis log-likelihood:

$$\log P(b_0, b_1, \ldots, b_{N-1}) = \log \sum_k b_0[k] + \sum_{i=0}^{N-2} \log \left( \sum_{k''} b_{i+1}[k''] \cdot \sum_k \alpha_i[k] \cdot A_i[k, k''] \right)$$

This measures how well the trellis model (with language-similarity transitions) explains the observed lens probabilities. Higher values indicate more coherent language flow.

---

## 5. Viterbi Path (Most Likely Token Sequence)

While the forward algorithm marginalizes over all paths, the Viterbi algorithm finds the single most likely path $\pi^* = (k_0^*, k_1^*, \ldots, k_{N-1}^*)$:

$$\pi^* = \arg\max_{\pi} P(\pi) = \arg\max_{\pi} \left[ b_0[k_0] \cdot \prod_{i=0}^{N-2} A_i[k_i, k_{i+1}] \cdot b_{i+1}[k_{i+1}] \right]$$

**Recursion:**

$$\delta_0[k] = b_0[k], \qquad \delta_{i+1}[k'] = b_{i+1}[k'] \cdot \max_k \big(\delta_i[k] \cdot A_i[k, k']\big)$$

with backpointers $\psi_{i+1}[k'] = \arg\max_k \big(\delta_i[k] \cdot A_i[k, k']\big)$.

In numpy:
```python
scores = deltas[i][:, None] * A_i    # (K, K): scores[k, k'] = delta_i[k] * A[k, k']
bp[i+1] = np.argmax(scores, axis=0)  # (K,): best predecessor for each k'
deltas[i+1] = b_next * scores.max(axis=0)  # (K,)
```

The Viterbi path's language trajectory is $\boldsymbol{\lambda}_i^{\text{Viterbi}} = T_i[k_i^*, :]$, which shows the language identity of the single most likely token sequence.

---

## 6. Numpy Implementation Sketch

```python
import numpy as np

def trellis_layer_analysis(
    p_lens: list[np.ndarray],    # N arrays of shape (K,)
    lang_mats: list[np.ndarray], # N arrays of shape (K, M)
    entropy_weights: list[np.ndarray] | None = None,  # N arrays of shape (K,), optional
    transition_mode: str = "row_norm",  # or "softmax"
    temperature: float = 1.0,
):
    """
    Trellis-based working language analysis for a single layer.
    
    Parameters
    ----------
    p_lens : list of N np.ndarray, each (K,)
        Normalized lens probabilities at each position.
    lang_mats : list of N np.ndarray, each (K, M)
        Language distribution matrix at each position.
        Row k = P(L | token_k).
    entropy_weights : list of N np.ndarray, each (K,), optional
        Entropy info weights w(t_k). If provided, emissions become
        b_i[k] = w_i[k] * p_i[k] (then renormalized).
    transition_mode : str
        "row_norm" or "softmax".
    temperature : float
        Temperature for softmax mode.
    
    Returns
    -------
    dict with keys:
        alphas : (N, K)     — forward state probabilities
        lambdas : (N, M)    — per-position language distributions
        transitions : list  — (N-1) transition matrices, each (K, K)
        working_lang : (M,) — layer working language
        viterbi_path : (N,) — most likely token path
        viterbi_langs : (N, M) — language distributions along Viterbi path
        log_likelihood : float
    """
    N = len(p_lens)
    K = p_lens[0].shape[0]
    M = lang_mats[0].shape[1]
    
    # --- Apply entropy weighting to emissions if provided ---
    b_list = []
    for i in range(N):
        if entropy_weights is not None:
            b = entropy_weights[i] * p_lens[i]
        else:
            b = p_lens[i].copy()
        s = b.sum()
        b = b / s if s > 0 else np.full(K, 1.0 / K)
        b_list.append(b)
    
    # --- Compute transition matrices ---
    transitions = []
    for i in range(N - 1):
        # Similarity: inner product of language distributions
        S = lang_mats[i] @ lang_mats[i + 1].T  # (K, K)
        
        if transition_mode == "row_norm":
            A = S / S.sum(axis=1, keepdims=True)
        elif transition_mode == "softmax":
            S_scaled = S / temperature
            S_scaled = S_scaled - S_scaled.max(axis=1, keepdims=True)
            A = np.exp(S_scaled)
            A = A / A.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown transition_mode: {transition_mode}")
        transitions.append(A)
    
    # --- Forward pass ---
    alphas = np.zeros((N, K))
    lambdas = np.zeros((N, M))
    log_z = 0.0
    
    # Initialization
    alphas[0] = b_list[0]
    lambdas[0] = alphas[0] @ lang_mats[0]
    
    for i in range(N - 1):
        A = transitions[i]
        # Prediction: (A^T @ alpha_i)  — matrix-vector product
        pred = A.T @ alphas[i]           # (K,)
        # Update: multiply by emission, normalize
        alpha_unnorm = b_list[i + 1] * pred  # (K,)  element-wise
        z = alpha_unnorm.sum()
        alphas[i + 1] = alpha_unnorm / z
        log_z += np.log(z)
        # Language projection
        lambdas[i + 1] = alphas[i + 1] @ lang_mats[i + 1]  # (M,)
    
    # --- Working language ---
    working_lang = lambdas.mean(axis=0)  # (M,)
    
    # --- Viterbi path ---
    deltas = np.zeros((N, K))
    backpointers = np.zeros((N, K), dtype=int)
    deltas[0] = b_list[0]
    
    for i in range(N - 1):
        A = transitions[i]
        # scores[k, k'] = delta_i[k] * A[k, k']  — outer-like product
        scores = deltas[i][:, None] * A     # (K, K)
        backpointers[i + 1] = np.argmax(scores, axis=0)  # (K,)
        deltas[i + 1] = b_list[i + 1] * scores.max(axis=0)  # (K,)
    
    # Backtrack
    path = np.zeros(N, dtype=int)
    path[-1] = np.argmax(deltas[-1])
    for i in range(N - 2, -1, -1):
        path[i] = backpointers[i + 1, path[i + 1]]
    
    viterbi_langs = np.array([lang_mats[i][path[i]] for i in range(N)])
    
    return {
        "alphas": alphas,
        "lambdas": lambdas,
        "transitions": transitions,
        "working_lang": working_lang,
        "viterbi_path": path,
        "viterbi_langs": viterbi_langs,
        "log_likelihood": log_z,
    }
```

### Key Numpy Operations Summary

| Step | Math | Numpy | Shape |
|------|------|-------|-------|
| Similarity | $S_i = T_i T_{i+1}^T$ | `S = T_i @ T_next.T` | $(K, K)$ |
| Row-normalize | $A_i[k,:] = S_i[k,:] / \sum S_i[k,:]$ | `A = S / S.sum(axis=1, keepdims=True)` | $(K, K)$ |
| Prediction | $\mathbf{r} = A_i^T \boldsymbol{\alpha}_i$ | `pred = A.T @ alpha` | $(K,)$ |
| Update | $\boldsymbol{\alpha}_{i+1} \propto \mathbf{b}_{i+1} \odot \mathbf{r}$ | `alpha = b * pred; alpha /= alpha.sum()` | $(K,)$ |
| Language proj. | $\boldsymbol{\lambda}_i = \boldsymbol{\alpha}_i^T T_i$ | `lam = alpha @ T_i` | $(M,)$ |
| Working lang. | $\bar{\boldsymbol{\lambda}} = \frac{1}{N}\sum_i \boldsymbol{\lambda}_i$ | `wl = lambdas.mean(axis=0)` | $(M,)$ |
| Viterbi scores | $\text{scores}[k,k'] = \delta_i[k] \cdot A_i[k,k']$ | `scores = deltas[i][:,None] * A` | $(K, K)$ |

---

## 7. Toy Examples

### Toy Example 1: English-Dominant Layer ($N=3, K=2, M=2$)

**Setup.** Languages: $m=0$ (English), $m=1$ (French). Two tokens per position: token 0 is English-leaning, token 1 is French-leaning.

**Data:**

$$\mathbf{p}_0 = \begin{pmatrix} 0.7 \\ 0.3 \end{pmatrix}, \quad T_0 = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{pmatrix}$$

$$\mathbf{p}_1 = \begin{pmatrix} 0.6 \\ 0.4 \end{pmatrix}, \quad T_1 = \begin{pmatrix} 0.85 & 0.15 \\ 0.2 & 0.8 \end{pmatrix}$$

$$\mathbf{p}_2 = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}, \quad T_2 = \begin{pmatrix} 0.95 & 0.05 \\ 0.1 & 0.9 \end{pmatrix}$$

**Step 1: Transition matrices.**

Transition $A_0$ (positions $0 \to 1$):

$$S_0 = T_0 T_1^T = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{pmatrix} \begin{pmatrix} 0.85 & 0.2 \\ 0.15 & 0.8 \end{pmatrix} = \begin{pmatrix} 0.78 & 0.26 \\ 0.22 & 0.74 \end{pmatrix}$$

Row sums: $(1.04, \; 0.96)$. Row-normalize:

$$A_0 = \begin{pmatrix} 0.750 & 0.250 \\ 0.229 & 0.771 \end{pmatrix}$$

*Interpretation:* English token at position 0 transitions to English token at position 1 with probability 0.75 (high, since both are English-dominant). French-to-French transition is 0.771.

Transition $A_1$ (positions $1 \to 2$):

$$S_1 = T_1 T_2^T = \begin{pmatrix} 0.85 & 0.15 \\ 0.2 & 0.8 \end{pmatrix} \begin{pmatrix} 0.95 & 0.1 \\ 0.05 & 0.9 \end{pmatrix} = \begin{pmatrix} 0.815 & 0.220 \\ 0.230 & 0.740 \end{pmatrix}$$

Row sums: $(1.035, \; 0.970)$. Row-normalize:

$$A_1 = \begin{pmatrix} 0.787 & 0.213 \\ 0.237 & 0.763 \end{pmatrix}$$

**Step 2: Forward pass.**

*Initialization:*

$$\boldsymbol{\alpha}_0 = \mathbf{p}_0 = \begin{pmatrix} 0.7 \\ 0.3 \end{pmatrix}, \qquad \boldsymbol{\lambda}_0 = \boldsymbol{\alpha}_0^T T_0 = \begin{pmatrix} 0.66 \\ 0.34 \end{pmatrix}$$

*Step $0 \to 1$:*

Prediction: $A_0^T \boldsymbol{\alpha}_0 = \begin{pmatrix} 0.75 & 0.229 \\ 0.25 & 0.771 \end{pmatrix} \begin{pmatrix} 0.7 \\ 0.3 \end{pmatrix} = \begin{pmatrix} 0.5937 \\ 0.4063 \end{pmatrix}$

Update: $\boldsymbol{\alpha}_1 \propto \mathbf{p}_1 \odot \begin{pmatrix} 0.5937 \\ 0.4063 \end{pmatrix} = \begin{pmatrix} 0.6 \\ 0.4 \end{pmatrix} \odot \begin{pmatrix} 0.5937 \\ 0.4063 \end{pmatrix} = \begin{pmatrix} 0.3562 \\ 0.1625 \end{pmatrix}$

Sum $= 0.5187$. Normalize:

$$\boldsymbol{\alpha}_1 = \begin{pmatrix} 0.6867 \\ 0.3133 \end{pmatrix}, \qquad \boldsymbol{\lambda}_1 = \boldsymbol{\alpha}_1^T T_1 = \begin{pmatrix} 0.6464 \\ 0.3536 \end{pmatrix}$$

*Step $1 \to 2$:*

Prediction: $A_1^T \boldsymbol{\alpha}_1 = \begin{pmatrix} 0.787 & 0.237 \\ 0.213 & 0.763 \end{pmatrix} \begin{pmatrix} 0.6867 \\ 0.3133 \end{pmatrix} = \begin{pmatrix} 0.6150 \\ 0.3850 \end{pmatrix}$

Update: $\boldsymbol{\alpha}_2 \propto \mathbf{p}_2 \odot \begin{pmatrix} 0.6150 \\ 0.3850 \end{pmatrix} = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix} \odot \begin{pmatrix} 0.6150 \\ 0.3850 \end{pmatrix} = \begin{pmatrix} 0.3075 \\ 0.1925 \end{pmatrix}$

Sum $= 0.5000$. Normalize:

$$\boldsymbol{\alpha}_2 = \begin{pmatrix} 0.6150 \\ 0.3850 \end{pmatrix}, \qquad \boldsymbol{\lambda}_2 = \boldsymbol{\alpha}_2^T T_2 = \begin{pmatrix} 0.6228 \\ 0.3772 \end{pmatrix}$$

**Step 3: Working language.**

$$\bar{\boldsymbol{\lambda}} = \frac{1}{3}\left(\begin{pmatrix} 0.66 \\ 0.34 \end{pmatrix} + \begin{pmatrix} 0.6464 \\ 0.3536 \end{pmatrix} + \begin{pmatrix} 0.6228 \\ 0.3772 \end{pmatrix}\right) = \begin{pmatrix} 0.6431 \\ 0.3569 \end{pmatrix}$$

**Dominant working language:** English ($\hat{m} = 0$, probability 0.643).

**Viterbi path:** $(0, 0, 0)$ — all English tokens. This is the most likely token sequence, and its language trajectory is $(0.9, 0.1) \to (0.85, 0.15) \to (0.95, 0.05)$, consistently English.

**Comparison to marginal:**

| Position | Marginal $\boldsymbol{\lambda}_i^{\text{marg}}$ | Trellis $\boldsymbol{\lambda}_i$ |
|----------|------|---------|
| 0 | $(0.66, 0.34)$ | $(0.66, 0.34)$ — identical (no transition yet) |
| 1 | $(0.59, 0.41)$ | $(0.6464, 0.3536)$ — trellis is more English |
| 2 | $(0.525, 0.475)$ | $(0.6228, 0.3772)$ — trellis is more English |

The trellis preserves English dominance more strongly than the marginal because the transition model reinforces same-language continuations. The marginal at position 2 gives nearly equal weight to English and French (0.525 vs 0.475), but the trellis, remembering the English-dominant positions 0–1, assigns 0.623 to English.

---

### Toy Example 2: Language Shift — English to French ($N=3, K=3, M=3$)

**Setup.** Languages: $m=0$ (English), $m=1$ (French), $m=2$ (German). Three tokens per position with a gradual shift from English-dominant to French-dominant lens predictions.

**Data:**

$$\mathbf{p}_0 = \begin{pmatrix} 0.5 \\ 0.3 \\ 0.2 \end{pmatrix}, \quad T_0 = \begin{pmatrix} 0.8 & 0.1 & 0.1 \\ 0.1 & 0.8 & 0.1 \\ 0.1 & 0.1 & 0.8 \end{pmatrix}$$

$$\mathbf{p}_1 = \begin{pmatrix} 0.3 \\ 0.5 \\ 0.2 \end{pmatrix}, \quad T_1 = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.15 & 0.75 & 0.1 \\ 0.1 & 0.1 & 0.8 \end{pmatrix}$$

$$\mathbf{p}_2 = \begin{pmatrix} 0.2 \\ 0.6 \\ 0.2 \end{pmatrix}, \quad T_2 = \begin{pmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.85 & 0.05 \\ 0.1 & 0.15 & 0.75 \end{pmatrix}$$

*Token character:* token 0 is English-leaning, token 1 is French-leaning, token 2 is German-leaning at each position. The lens probabilities shift from favoring token 0 (English) to token 1 (French).

**Step 1: Transition matrix $A_0$ (positions $0 \to 1$).**

$$S_0 = T_0 T_1^T = \begin{pmatrix} 0.59 & 0.205 & 0.17 \\ 0.24 & 0.625 & 0.17 \\ 0.17 & 0.17 & 0.66 \end{pmatrix}$$

Row sums: $(0.965, \; 1.035, \; 1.000)$. Row-normalize:

$$A_0 = \begin{pmatrix} 0.611 & 0.212 & 0.176 \\ 0.232 & 0.604 & 0.164 \\ 0.170 & 0.170 & 0.660 \end{pmatrix}$$

*Interpretation:* The matrix is strongly diagonal — English-to-English (0.611), French-to-French (0.604), German-to-German (0.660) are the dominant transitions, reflecting language continuity.

**Transition matrix $A_1$ (positions $1 \to 2$).**

$$S_1 = T_1 T_2^T = \begin{pmatrix} 0.490 & 0.245 & 0.175 \\ 0.325 & 0.658 & 0.203 \\ 0.170 & 0.135 & 0.625 \end{pmatrix}$$

Row sums: $(0.910, \; 1.185, \; 0.930)$. Row-normalize:

$$A_1 = \begin{pmatrix} 0.538 & 0.269 & 0.192 \\ 0.274 & 0.555 & 0.171 \\ 0.183 & 0.145 & 0.672 \end{pmatrix}$$

**Step 2: Forward pass.**

*Initialization:*

$$\boldsymbol{\alpha}_0 = \mathbf{p}_0 = \begin{pmatrix} 0.5 \\ 0.3 \\ 0.2 \end{pmatrix}, \qquad \boldsymbol{\lambda}_0 = \boldsymbol{\alpha}_0^T T_0 = (0.45, \; 0.31, \; 0.24)$$

*Step $0 \to 1$:*

Prediction: $A_0^T \boldsymbol{\alpha}_0 = \begin{pmatrix} 0.611 & 0.232 & 0.170 \\ 0.212 & 0.604 & 0.170 \\ 0.176 & 0.164 & 0.660 \end{pmatrix} \begin{pmatrix} 0.5 \\ 0.3 \\ 0.2 \end{pmatrix} = \begin{pmatrix} 0.4093 \\ 0.3214 \\ 0.2694 \end{pmatrix}$

Update: $\boldsymbol{\alpha}_1 \propto \mathbf{p}_1 \odot \begin{pmatrix} 0.4093 \\ 0.3214 \\ 0.2694 \end{pmatrix} = \begin{pmatrix} 0.3 \\ 0.5 \\ 0.2 \end{pmatrix} \odot \begin{pmatrix} 0.4093 \\ 0.3214 \\ 0.2694 \end{pmatrix} = \begin{pmatrix} 0.1228 \\ 0.1607 \\ 0.0539 \end{pmatrix}$

Sum $= 0.3374$. Normalize:

$$\boldsymbol{\alpha}_1 = \begin{pmatrix} 0.3640 \\ 0.4763 \\ 0.1597 \end{pmatrix}, \qquad \boldsymbol{\lambda}_1 = \boldsymbol{\alpha}_1^T T_1 = (0.3422, \; 0.4460, \; 0.2118)$$

*Key observation:* Even though the lens at position 1 already favors French ($\mathbf{p}_1[1] = 0.5$), the trellis amplifies the French probability further. The prediction step, coming from the English-dominant position 0, gives only 0.3214 to the French token. But the emission $\mathbf{p}_1[1] = 0.5$ is strong enough that after normalization, the French state probability rises to 0.4763 (vs 0.5 marginal). Meanwhile, English drops to 0.364 (vs 0.3 marginal) because the transition from the English-dominant $\alpha_0$ boosts the English prediction.

*Step $1 \to 2$:*

Prediction: $A_1^T \boldsymbol{\alpha}_1 = \begin{pmatrix} 0.538 & 0.274 & 0.183 \\ 0.269 & 0.555 & 0.145 \\ 0.192 & 0.171 & 0.672 \end{pmatrix} \begin{pmatrix} 0.3640 \\ 0.4763 \\ 0.1597 \end{pmatrix} = \begin{pmatrix} 0.3558 \\ 0.3855 \\ 0.2587 \end{pmatrix}$

Update: $\boldsymbol{\alpha}_2 \propto \mathbf{p}_2 \odot \begin{pmatrix} 0.3558 \\ 0.3855 \\ 0.2587 \end{pmatrix} = \begin{pmatrix} 0.2 \\ 0.6 \\ 0.2 \end{pmatrix} \odot \begin{pmatrix} 0.3558 \\ 0.3855 \\ 0.2587 \end{pmatrix} = \begin{pmatrix} 0.0712 \\ 0.2313 \\ 0.0517 \end{pmatrix}$

Sum $= 0.3542$. Normalize:

$$\boldsymbol{\alpha}_2 = \begin{pmatrix} 0.2009 \\ 0.6530 \\ 0.1461 \end{pmatrix}, \qquad \boldsymbol{\lambda}_2 = \boldsymbol{\alpha}_2^T T_2 = (0.2005, \; 0.6372, \; 0.1623)$$

**Step 3: Working language.**

$$\bar{\boldsymbol{\lambda}} = \frac{1}{3}\big((0.45, 0.31, 0.24) + (0.3422, 0.4460, 0.2118) + (0.2005, 0.6372, 0.1623)\big)$$

$$= \frac{1}{3}(0.9927, \; 1.3932, \; 0.6141) = (0.3309, \; 0.4644, \; 0.2047)$$

**Dominant working language:** French ($\hat{m} = 1$, probability 0.464).

**Viterbi path:** $(1, 1, 1)$ — all French tokens. The most likely single path selects the French token at every position. Language trajectory: $(0.1, 0.8, 0.1) \to (0.15, 0.75, 0.1) \to (0.1, 0.85, 0.05)$, consistently French-dominant.

**Comparison: marginal vs trellis:**

| Position | Marginal $\boldsymbol{\lambda}_i^{\text{marg}}$ | Trellis $\boldsymbol{\lambda}_i$ | Difference |
|----------|------|---------|------------|
| 0 | $(0.450, 0.310, 0.240)$ | $(0.450, 0.310, 0.240)$ | Identical (initialization) |
| 1 | $(0.305, 0.455, 0.240)$ | $(0.342, 0.446, 0.212)$ | Trellis: more English, less German |
| 2 | $(0.200, 0.600, 0.200)$ | $(0.201, 0.637, 0.162)$ | Trellis: more French, less German |

The trellis suppresses the German probability (which has no strong lens support and no strong transition reinforcement) and amplifies the dominant French trend. The temporal smoothing effect is most visible at position 1, where the trellis gives more English than the marginal because position 0 was English-dominant and the transition preserves some of that signal.

---

### Toy Example 3: Code-Switching Detection ($N=5, K=3, M=2$)

**Setup.** Languages: English ($m=0$), French ($m=1$). The lens predictions switch from English-dominant to French-dominant around position 2, then back toward English at position 4.

**Data:**

| $i$ | $\mathbf{p}_i$ | $T_i$ (rows: tok0, tok1, tok2) |
|-----|---------|------|
| 0 | $(0.6, 0.3, 0.1)$ | $\begin{pmatrix} 0.9 & 0.1 \\ 0.15 & 0.85 \\ 0.5 & 0.5 \end{pmatrix}$ |
| 1 | $(0.5, 0.35, 0.15)$ | $\begin{pmatrix} 0.85 & 0.15 \\ 0.2 & 0.8 \\ 0.4 & 0.6 \end{pmatrix}$ |
| 2 | $(0.2, 0.6, 0.2)$ | $\begin{pmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \\ 0.5 & 0.5 \end{pmatrix}$ |
| 3 | $(0.15, 0.65, 0.2)$ | $\begin{pmatrix} 0.75 & 0.25 \\ 0.15 & 0.85 \\ 0.3 & 0.7 \end{pmatrix}$ |
| 4 | $(0.55, 0.3, 0.15)$ | $\begin{pmatrix} 0.9 & 0.1 \\ 0.2 & 0.8 \\ 0.4 & 0.6 \end{pmatrix}$ |

**Forward pass results (computed):**

| $i$ | $\boldsymbol{\alpha}_i$ | $\boldsymbol{\lambda}_i$ (Eng, Fre) | Marginal $\boldsymbol{\lambda}_i^{\text{marg}}$ |
|-----|---------|---------|---------|
| 0 | $(0.600, 0.300, 0.100)$ | $(0.635, 0.365)$ | $(0.635, 0.365)$ |
| 1 | $(0.582, 0.281, 0.137)$ | $(0.606, 0.395)$ | $(0.555, 0.445)$ |
| 2 | $(0.249, 0.533, 0.218)$ | $(0.362, 0.638)$ | $(0.320, 0.680)$ |
| 3 | $(0.123, 0.679, 0.198)$ | $(0.253, 0.747)$ | $(0.270, 0.730)$ |
| 4 | $(0.376, 0.439, 0.186)$ | $(0.500, 0.500)$ | $(0.615, 0.385)$ |

**Working language:** $\bar{\boldsymbol{\lambda}} = (0.471, 0.529)$ — French dominant overall.

**Viterbi path:** $(0, 0, 1, 1, 1)$ — English tokens at positions 0–1, French tokens at positions 2–4. This captures the **code-switch point** at position 2.

**Key observations:**

1. **Transition smoothing:** At position 1, the trellis gives $\lambda_1 = (0.606, 0.395)$, more English than the marginal $(0.555, 0.445)$, because the transition from the English-dominant position 0 preserves English probability.

2. **Switch detection:** The trellis language probability crosses from English-dominant to French-dominant between positions 1 and 2. The KL divergence $\text{KL}(\boldsymbol{\lambda}_2 \| \boldsymbol{\lambda}_1) \approx 0.174$ bits quantifies this shift.

3. **Memory effect at position 4:** The marginal at position 4 gives $(0.615, 0.385)$ (English-dominant, matching the lens), but the trellis gives $(0.500, 0.500)$ (balanced). This is because the transition from the French-dominant position 3 pulls the prediction toward French, partially counteracting the English-leaning lens probabilities. The trellis "remembers" the French context.

4. **Viterbi vs forward:** The Viterbi path shows a clean code-switch at position 2, while the forward pass shows a gradual transition. Both perspectives are informative: Viterbi identifies the switch point, while the forward pass quantifies the uncertainty around it.

---

## 8. Verification

### Consistency checks

1. **Probability axioms:** At every position, $\sum_k \alpha_i[k] = 1$ (by normalization) and $\sum_m \lambda_i[m] = 1$ (as a convex combination of probability distributions). Verified numerically.

2. **Degenerate case — uniform language distributions:** If all tokens have uniform language distributions ($T_i[k,m] = 1/M$), then $S_i[k,k'] = 1/M$ for all $k,k'$, $A_i[k,k'] = 1/K$, and the forward pass reduces to $\alpha_{i+1} \propto p_{i+1}$ (the marginal). The trellis adds no information when all tokens are language-ambiguous.

3. **Degenerate case — single language ($M=1$):** All $T_i[k,0] = 1$, so $S_i[k,k'] = 1$, $A_i[k,k'] = 1/K$, and $\lambda_i[0] = 1$ for all $i$. The working language is trivially that single language.

4. **Degenerate case — $K=1$:** Only one token per position, so $\alpha_i[0] = 1$ for all $i$, and $\lambda_i = T_i[0,:]$ (the single token's language distribution).

5. **Relationship to marginal:** When $A_i$ is uniform ($1/K$), the prediction step gives a constant, and $\alpha_{i+1} \propto p_{i+1}$. Thus the trellis reduces to the marginal approach.

### Assumptions

1. **Markov property:** The transition depends only on the current and next position's language distributions, not on earlier positions. This is a first-order Markov assumption.

2. **Language as transition signal:** We assume that language similarity between tokens is a meaningful proxy for "transition likelihood" in the model's internal computation. This is reasonable for multilingual models where language identity is a major factor in token prediction.

3. **Position-independent $K$:** We assume the same number of top-$K$ tokens at each position. If $K$ varies, the transition matrix dimensions change, requiring padding or truncation.

4. **Normalized lens probabilities:** We assume $p_i[k]$ are normalized (sum to 1 over $k$). If the raw lens probabilities don't sum to 1 (because the top-$K$ doesn't cover all probability mass), they should be renormalized first.

### Limitations

1. **No cross-layer transitions:** The trellis operates within a single layer. Cross-layer analysis (comparing trellis results across layers) requires running the algorithm separately for each layer and comparing the resulting $\bar{\boldsymbol{\lambda}}$ vectors.

2. **Transition model is heuristic:** The language-similarity-based transition is a modeling choice, not derived from the model's actual internal dynamics. Alternative transition models (e.g., based on token co-occurrence statistics or attention patterns) could be substituted.

3. **Small $K$ truncation:** Using top-$K$ predictions discards the tail of the distribution. If important tokens are outside the top-$K$, the trellis may miss relevant language information.

4. **No learned parameters:** The transition matrix is computed from data (language distributions) rather than learned. A learned transition model could potentially capture more nuanced patterns.

### Alternative approaches

1. **Attention-based transitions:** Instead of language similarity, use the model's attention weights between positions $i$ and $i+1$ to define transition probabilities.

2. **Token embedding similarity:** Use cosine similarity between token embeddings as the transition signal.

3. **CRF formulation:** Replace the generative HMM with a conditional random field (CRF), which models $P(\text{states} \mid \text{observations})$ directly and can incorporate richer features.

4. **Full Bayesian smoothing:** Use a Dirichlet prior on the transition matrix and compute posterior estimates, which would regularize the transitions for small $K$.
