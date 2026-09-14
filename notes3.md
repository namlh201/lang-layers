Detecting the internal "working language" (or *language of thought*) across an LLM's layers relies on mapping the intermediate logit lens projections (the top-$k$ tokens decoded from each layer's residual stream) into a language space.

Here is a structured, mathematically sound method to compute and analyze the working language at any position $i$ and layer $l$.

---

## 1. Mathematical Formulation

At token position $i$ and layer $l$, you have a distribution over the top-$k$ vocabulary tokens:

$$\mathcal{T}_{l,i} = \{(t_j, p(t_j \mid l, i))\}_{j=1}^k$$

Where $t_j$ is a token ID, and $p(t_j \mid l, i)$ is the probability assigned to token $t_j$ after decoding layer $l$'s state (e.g., via Logit Lens or Tuned Lens) and applying softmax across the top-$k$.

You also have a token-level language mapping $P(\mathcal{L} \mid t)$, where $\mathcal{L} = \{L_1, L_2, \dots, L_m\}$ represents your set of target languages (e.g., English, Spanish, Mandarin, etc.).

### Step-by-Step Calculation

#### A. Expected Language Probability Vector

Compute the expected probability of language $L_a$ at layer $l$ and position $i$ by computing the weighted average across the top-$k$ tokens:

$$P(L_a \mid l, i) = \frac{\sum_{j=1}^k P(L_a \mid t_j) \cdot p(t_j \mid l, i)}{\sum_{j=1}^k p(t_j \mid l, i)}$$

> **Note:** Normalizing by $\sum_{j=1}^k p(t_j \mid l, i)$ ensures the probabilities sum to $1$ over the candidate set, correcting for any probability mass outside the top-$k$.

#### B. Handling Ambiguous & Shared Tokens (Token Entropy Weighting)

Many subword tokens (e.g., punctuation, numbers, single letters, shared roots) exist in multiple languages with high uncertainty. If a token $t_j$ has uniform probability across all languages, it adds noise rather than signal.

To solve this, scale each token's contribution by its **Language Information Content** (or inverse language entropy):

$$w(t_j) = 1 - \frac{H(\mathcal{L} \mid t_j)}{\log_2 \vert{}\mathcal{L}\vert{}}$$

Where $H(\mathcal{L} \mid t_j) = -\sum_{a=1}^m P(L_a \mid t_j) \log_2 P(L_a \mid t_j)$ is the Shannon entropy of the language distribution for token $t_j$.

The information-weighted working language probability becomes:

$$P^*(L_a \mid l, i) = \frac{\sum_{j=1}^k w(t_j) \cdot P(L_a \mid t_j) \cdot p(t_j \mid l, i)}{\sum_{j=1}^k w(t_j) \cdot p(t_j \mid l, i)}$$

---

## 2. Defining Working Language Metrics

Once you have $P^*(L_a \mid l, i)$, you can extract three key metrics to quantify internal language dynamics:

| Metric | Formula / Definition | What It Tells You |
| --- | --- | --- |
| **Dominant Working Language** | $\hat{L}(l, i) = \arg\max_{L_a} P^*(L_a \mid l, i)$ | The primary language the model is processing in at layer $l$. |
| **Language Ambiguity / Cross-lingual Entropy** | $H_{\text{working}}(l, i) = -\sum_{a} P^*(L_a \mid l,i) \log_2 P^*(L_a \mid l,i)$ | **High:** Abstract conceptual/language-agnostic space.<br>

<br>**Low:** Decisive representation in a specific language. |
| **Language Shift Point** | $\Delta_{l} = \mathbf{D}_{\text{KL}}\left(P^*(\mathcal{L} \mid l, i) \parallel P^*(\mathcal{L} \mid l-1, i)\right)$ | Pinpoints exact layers where the model translates or switches internal working space. |

---

## 3. How to Interpret Layer Trends

In multilingual Transformer models (like Llama, Mistral, or Qwen), analyzing $P^*(\mathcal{L} \mid l, i)$ across layers typically reveals three distinct phases:

```
[Layer 0 → L_low]     Early Layers: Surface lexical processing (Input prompt language)
        │
[L_low → L_mid]     Middle Layers: Semantic pivot phase (Often shifts to English or abstract space)
        │
[L_mid → L_high]    Late Layers: Alignment & Decoding (Shifts to target output language)

```

1. **Lower Layers (0–25% depth):** High reliance on literal surface tokens; matches the input prompt language.
2. **Middle Layers (25–75% depth):** The **Concept Space**. For non-English inputs, LLMs frequently show a shift toward English (the dominant training corpus language) as an intermediate "pivot language" before resolving final reasoning.
3. **Upper Layers (75–100% depth):** Output generation alignment. The distribution rapidly converts into the intended output language.

---

## 4. Practical Implementation Checklist

* **Pre-filter Vocabulary:** Mask out numbers, white spaces, punctuation, and code tokens from $t_j$ evaluation, as they skew language probabilities.
* **Calibrate Language Classifiers:** Ensure your token-level language probability $P(\mathcal{L} \mid t)$ is computed using subword frequency counts from large monolingual corpora (e.g., Wikipedia/CC100 tokenized by your model's exact tokenizer).
* **Top-$k$ Cutoff:** Set $k \in [10, 50]$. Values below $10$ may miss subtle language biases in late candidates; values above $50$ introduce excess low-probability noise.
