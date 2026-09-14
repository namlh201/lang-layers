# Language Switch Detection and Layer Specificity from Trellis Language Probabilities

Companion to `trellis_formulation.md` (Definitions 1–10 there; notation reused here). Two questions are addressed:

1. **Switch detection:** Given the trellis language probabilities $\boldsymbol{\lambda}_i^{(j)}$, detect *language switching points* in the generated sequence and assign each a *confidence*.
2. **Layer specificity:** Decide whether a layer $j$ is *language-specific* or *language-neutral*, with a null model and a decision rule.

---

## 1. Setup and Notation

### Recap of trellis quantities (per layer $j$)

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $\boldsymbol{\alpha}_i$ | $(K,)$ | Forward state posterior over top-$K$ tokens at position $i$ |
| $A_i$ | $(K,K)$ | Language-similarity transition matrix (Def. 7 there) |
| $S_i$ | $(K,K)$ | Raw similarity $T_i T_{i+1}^T$ (Def. 6 there) |
| $z_{i+1}$ | scalar | Forward normalizer $= \sum_{k'} b_{i+1}[k']\,(A_i^T\boldsymbol{\alpha}_i)[k']$ |
| $\boldsymbol{\lambda}_i$ | $(M,)$ | Filtered language posterior $\boldsymbol{\alpha}_i^T T_i$ (Def. 9 there) |
| $\ell_j$ | scalar | Trellis log-likelihood $\log \sum_k b_0[k] + \sum_{i} \log z_{i+1}$ |

**Definition 1 (Boundary).** A *boundary* is an adjacent position pair; boundary $i$ links positions $i$ and $i+1$, for $i \in \{0, \ldots, N-2\}$. All switch metrics are indexed by boundaries.

**Definition 2 (Ground-truth language).** For each position $i$, the reference language is

$$g_i = \arg\max_m D^{\text{tok}}_i[m], \qquad D^{\text{tok}}_i \in \Delta^{M-1}$$

where $D^{\text{tok}}_i$ is the language distribution of the **actual generated token** at position $i$, looked up from the token-language database. If $D^{\text{tok}}_i$ is uniform (language-ambiguous token), $g_i$ is undefined and that position is excluded from alignment metrics.

**Definition 3 (Dominant language).** The dominant language at position $i$ is $\mu_i = \arg\max_m \lambda_i[m]$ (ties broken by language index; ties are treated as non-committed, see Def. 7).

**Definition 4 (Tracked languages).** The task languages $\mathcal{A} = \{a_{\text{src}}, a_{\text{mid}}, a_{\text{tgt}}\}$ (source, pivot — typically English —, target). The *task-restricted posterior* is the renormalized restriction $\tilde{\boldsymbol{\lambda}}_i = \boldsymbol{\lambda}_i[\mathcal{A}] / \sum_{a \in \mathcal{A}} \lambda_i[a] \in \Delta^{|\mathcal{A}|-1}$.

---

## 2. Language Switch Detection

### 2.1 Naive baseline and its failure modes

The naive detector flags boundary $i$ iff $\mu_{i+1} \neq \mu_i$. Failure modes:

1. **No magnitude:** a 0.51→0.49 flip counts as much as a 0.9→0.1 flip.
2. **No uncertainty handling:** when $\boldsymbol{\lambda}$ is diffuse (near-uniform), the argmax is essentially arbitrary and flips are pure noise.
3. **No persistence:** a single-position blip that immediately reverts is noise, not a switch.
4. **Ignores cross-layer consistency:** a real behavioral switch should be visible across a band of layers, not one.

### 2.2 Switch evidence: filtered total variation

**Definition 5 (Switch evidence).** The *language shift* at boundary $i$ is the total variation distance between consecutive filtered language posteriors:

$$d_i \;=\; \mathrm{TV}(\boldsymbol{\lambda}_{i+1},\, \boldsymbol{\lambda}_i) \;=\; \tfrac{1}{2}\sum_{m=0}^{M-1}\big|\lambda_{i+1}[m] - \lambda_i[m]\big| \;\in\; [0, 1]$$

TV is chosen over KL/JS because it (a) is bounded on $[0,1]$ without normalization, (b) equals the worst-case change in the probability of *any* language event: $|\lambda_{i+1}[a] - \lambda_i[a]| \le d_i$ for all $a$, and (c) is symmetric.

**Property (SNR advantage of the trellis).** The $\boldsymbol{\lambda}_i$ are *filtered* — each already incorporates the language-continuity prior through the prediction step $A_i^T \boldsymbol{\alpha}_i$. Consequently:

- *Spurious* fluctuations (noise in one position's lens predictions) are **absorbed** by the transition prior — $d_i$ shrinks relative to the marginal version.
- *Genuine* switches require emission evidence strong enough to overcome the transition prior — $d_i$ is preserved or **amplified** at true boundaries, because the trellis delays the switch (memory effect) and then releases it.

This improves the signal-to-noise ratio of $d_i$ relative to computing TV on the marginal posteriors $\mathbf{p}_i^T T_i$. (Verified numerically in §5.1.)

**Definition 6 (Task-restricted switch evidence).** For translation-pivot analysis, $d_i^{\text{task}} = \mathrm{TV}(\tilde{\boldsymbol{\lambda}}_{i+1}, \tilde{\boldsymbol{\lambda}}_i)$ on the renormalized restriction to $\mathcal{A}$. Use $d_i^{\text{task}}$ to detect movement among {source, pivot, target} (e.g. source→pivot→target cascades); use $d_i$ (full $M$) for general code-switching.

### 2.3 Confidence decomposition

A switch is *confident* when three conditions hold simultaneously: the posterior actually moved ($d$), both endpoints are language-committed ($\kappa$), and the new language persists ($\rho$).

**Definition 7 (Commitment).** The commitment of position $i$ is the normalized excess of its dominant-language probability over chance:

$$\kappa_i \;=\; \frac{\max_m \lambda_i[m] - \tfrac{1}{M}}{1 - \tfrac{1}{M}} \;\in\; [0, 1]$$

$\kappa_i = 0$ iff $\boldsymbol{\lambda}_i$ is uniform (position uncommitted / language-neutral evidence); $\kappa_i = 1$ iff all mass is on one language. (For the tracked-language analysis, replace $M$ by $|\mathcal{A}|$ and $\lambda_i$ by $\tilde{\boldsymbol{\lambda}}_i$.)

**Definition 8 (Persistence).** With a sustain window $w \ge 1$:

$$\rho_i \;=\; \frac{1}{\min(w,\, N-2-i)} \sum_{t=1}^{\min(w,\, N-2-i)} \lambda_{i+1+t}[\mu_{i+1}] \;\in\; [0, 1]$$

the mean posterior mass on the post-boundary dominant language $\mu_{i+1}$ over the $w$ positions *after* the switch position $i+1$ (truncated at the sequence end). Edge cases: if $\mu_{i+1}$ is a tie, $\rho_i := 0$; for the final boundary ($i = N-2$, empty window), $\rho_i := \lambda_{i+1}[\mu_{i+1}]$ — evidence limited to the switch position itself.

**Definition 9 (Switch confidence).**

$$c_i \;=\; d_i \;\cdot\; \min(\kappa_i,\, \kappa_{i+1}) \;\cdot\; \rho_i \;\in\; [0, 1]$$

The **min** (not mean) is deliberate: a switch requires *both* endpoints identified — one cannot switch away from, or into, an unidentified language. A diffuse endpoint zeroes the confidence regardless of the other side.

**Interpretation.** Each factor is a necessary condition quantified on $[0,1]$: worst-case posterior swing, identification confidence of the weaker endpoint, and sustained continuation probability. Their product is a heuristic joint confidence — a *lower-bound-style* score for the event "the working language changed at boundary $i$ and was identified on both sides and persisted." It is not a posterior; the exact Bayesian version is §2.4.

**Definition 10 (Detection rule).** Boundary $i$ is a *detected switch* iff $c_i \ge \theta_c$. With $N-1$ boundaries tested, control the family-wise error by choosing $\theta_c$ from the empirical distribution of the statistic being thresholded — e.g. the $95^{\text{th}}$ percentile of the consensus values $\bar{c}$ (per-layer $c$ for onset computation), or an explicitly given threshold; apply a Bonferroni-style correction when a per-boundary $p$-value is available (via the permutation calibration below).

**Practical calibration.** Boundaries with $\min(\kappa_i,\kappa_{i+1}) \approx 0$ are excluded by construction. A data-driven noise floor for $d$ can be estimated from boundaries whose endpoints fall in low-commitment regions (empirical quantile of $d$ among $\kappa$-filtered boundaries), giving an alternative threshold $\theta_d$.

### 2.4 Gold standard: smoothed switch posterior (forward–backward)

The filtered $\boldsymbol{\lambda}_i$ use evidence up to $i$ only. The fully Bayesian switch probability uses *all* evidence. Add the standard backward pass:

$$\boldsymbol{\beta}_{N-1} = \mathbf{1}, \qquad \beta_i[k] = \sum_{k'} A_i[k, k'] \, b_{i+1}[k'] \, \beta_{i+1}[k']$$

**Definition 11 (Pairwise smoothed posterior).** For boundary $i$:

$$\xi_i[k, k'] \;=\; \frac{\alpha_i[k] \; A_i[k, k'] \; b_{i+1}[k'] \; \beta_{i+1}[k']}{P(b_0, \ldots, b_{N-1})}$$

**Definition 12 (Smoothed switch probability).** Project the pairwise posterior onto language (dis)agreement via the similarity matrix $S_i$ (Def. 6 of the formulation doc), which is exactly the probability that two draws from the two token-language distributions coincide:

$$P(\text{switch at } i \mid \text{all evidence}) \;=\; 1 - \sum_{k, k'} \xi_i[k, k'] \; S_i[k, k']$$

This is the principled replacement for $c_i$: an exact posterior under the trellis generative model. It is **conservative**: because $A_i$ was *built* to favor same-language continuations, the model's prior leans against switches, so the posterior understates them. Cost: one extra backward pass per layer ($\mathcal{O}(N K^2)$, same as forward).

### 2.5 Likelihood-ratio boundary statistic

A complementary, token-level diagnostic that is *already computed* by the forward pass up to logging:

**Definition 13 (Boundary log-likelihood ratio).** Let $\mathbf{r}_{i+1} = A_i^T \boldsymbol{\alpha}_i$ (the one-step prediction; $\sum_{k'} r_{i+1}[k'] = 1$ since $A_i$ is row-stochastic) and $z_{i+1} = \langle \mathbf{b}_{i+1}, \mathbf{r}_{i+1} \rangle$ (the forward normalizer). Under the language-agnostic null $A^{\text{null}}_i[k,k'] = 1/K$, the prediction is uniform and $z^{\text{null}}_{i+1} = 1/K$. Define

$$G_i \;=\; \log \frac{z_{i+1}}{z^{\text{null}}_{i+1}} \;=\; \log\big(K \cdot z_{i+1}\big) \;\in\; \mathbb{R}$$

- $G_i > 0$: the emission at $i+1$ is *better explained* by language-coherent continuation than by an agnostic model — no switch.
- $G_i \approx 0$: language-agnostic — the transition carries no information here (diffuse evidence).
- $G_i < 0$: the emission *actively contradicts* the language context — **strong switch evidence**, sharper than any $\boldsymbol{\lambda}$-based statistic because it operates on the token-level prediction–emission misalignment rather than on the projected language probabilities.

Note $G_i$ and $d_i$ measure different things: $G_i$ detects *incoherence* (prediction vs. emission), $d_i$ detects *movement* (posterior vs. posterior). A switch typically shows $G_i < 0$ **and** $d_i$ large; noise shows $G_i \approx 0$ with moderate $d_i$.

### 2.6 Cross-layer consensus and switch onset

The same boundary $i$ yields a confidence $c_i^{(j)}$ per layer $j$. A behaviorally real switch should be detected by a coherent band of layers.

**Definition 14 (Consensus confidence).** With layer weights $w_j \ge 0$ (from §3.5; layers without language information should not vote):

$$\bar{c}_i \;=\; \frac{\sum_j w_j \, c_i^{(j)}}{\sum_j w_j}$$

**Definition 15 (Switch onset depth).** For a detected switch at boundary $i$,

$$j^{*}_i \;=\; \min\Big\{ j \;:\; \tfrac{1}{J - j + 1} \sum_{j' \ge j} \mathbb{1}\big[c_i^{(j')} \ge \theta_c\big] \;\ge\; 0.8 \Big\}$$

the earliest layer from which the detection rate stays $\ge 80\%$. $j^{*}_i$ answers *"at what depth does the model resolve this language switch?"* — the layer-wise analogue of the per-position onset.

### 2.7 Edge cases

| Case | Behavior |
|------|----------|
| Uniform fallback rows in $T_i$ (token absent from DB) | $\boldsymbol{\lambda}_i$ pulled toward uniform → $\kappa_i \to 0$ → $c_i \to 0$. False positives are suppressed *by construction*; at worst the boundary is unreportable. |
| Zero emission vector $\mathbf{b}_i = \mathbf{0}$ (all entropy weights zero) | Implementation falls back to uniform $\mathbf{b}_i = \mathbf{1}/K$; same as above. |
| Sequence start | Boundary set starts at $i=0$; no left-context issue since $\boldsymbol{\lambda}_0$ is a valid posterior. |
| $N = 1$ | No boundaries; no switches detectable. |
| $M = 1$ or all tokens same language | $d_i \equiv 0$; correctly no detections. |
| Argmax ties | Handled as non-commitment ($\rho := 0$; $\kappa$ uses the max value which is shared). |
| Many fallback positions | Effective sample shrinks; report the fraction of committed positions alongside any switch statistics. |

---

## 3. Layer Language-Specificity

**Answer to the question:** yes — the trellis language probabilities support a principled language-specific vs. language-neutral decision, but *peakedness alone is insufficient*. A layer can be peaked yet wrong (monolingual shortcut: always decoding to one language regardless of context). Specificity decomposes into three separable axes, each with its own null:

1. **Commitment** — are the $\boldsymbol{\lambda}_i^{(j)}$ peaked? (internal, no ground truth)
2. **Tracking** — does the peak *follow the true language* $g_i$? (external, needs $g$)
3. **Coherence** — do language-coherent transitions explain the emissions? (internal, temporal; reuses the forward-pass likelihood)

### 3.1 Definitions

**Definition 16 (Layer commitment).**

$$C_j \;=\; \frac{1}{N}\sum_{i=0}^{N-1} \kappa_i^{(j)} \;=\; \frac{1}{N}\sum_i \frac{\max_m \lambda_i^{(j)}[m] - \tfrac{1}{M}}{1 - \tfrac{1}{M}} \;\in\; [0, 1]$$

Mean normalized commitment (same $\kappa$ as Def. 7 — the identical quantity appears per-boundary in switch confidence and per-layer here). $C_j \approx 0$: the layer's language posterior is diffuse → *neutral on the commitment axis*.

**Definition 17 (Soft tracking score).** Against a reference sequence $g = (g_0, \ldots, g_{N-1})$ over committed positions $\mathcal{I} = \{i : g_i \text{ defined}\}$:

$$T_j \;=\; \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \lambda_i^{(j)}[g_i] \;\in\; [0, 1]$$

mean posterior mass on the true language. (Hard variant: $A_j = \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \mathbb{1}[\mu_i^{(j)} = g_i]$.)

### 3.2 Null models

**Null A (uniform).** If every $\boldsymbol{\lambda}_i^{(j)}$ were uniform, $T_j = 1/M$ regardless of $g$. This is the *weak* baseline; a layer that always predicts one language on a task dominated by that language beats it trivially.

**Null B (permutation).** The *strong* null for "no positional alignment": under $H_0$, the pairing between the layer's language trajectory and the ground truth is exchangeable. For a permutation $\sigma$ of $\mathcal{I}$:

$$T_j^{\sigma} \;=\; \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \lambda_{\sigma(i)}^{(j)}[g_i]$$

**Definition 18 (Permutation z-score and p-value).** With $P$ random permutations ($P \approx 1000$):

$$Z_j = \frac{T_j - \hat{\mu}_{\text{perm}}}{\hat{\sigma}_{\text{perm}}} \quad (\hat{\sigma}_{\text{perm}} > 0; \text{ else } Z_j := 0), \qquad p_j = \frac{1 + \#\{T_j^{\sigma} \ge T_j\}}{P + 1}$$

The permutation mean has the exact closed form $\hat{\mu}_{\text{perm}} = \sum_a \hat{p}_g(a)\, \bar{\lambda}^{(j)}[a]$ (frequency of language $a$ in $g$ times the layer's mean posterior on $a$): the null removes only the *positional* alignment, keeping the layer's marginal language profile. This is what separates "commits to the right languages overall" from "commits at the right times."

**Caveat.** The exchangeability assumption is violated when $g$ (or $\boldsymbol{\lambda}$) is autocorrelated — which it is, since language is locally coherent. Block permutations (permuting contiguous blocks) restore validity; with a single-language reference ($g$ constant), all permutations are identical and Null B degenerates — then use Null A plus the coherence axis (Def. 19 / Null C).

**Null C (temporal, for the coherence score).** See §3.3.

### 3.3 Coherence: a ground-truth-free specificity score

The forward pass already computes how well the language-coherent transition model explains the emissions. Comparing against the language-agnostic transition:

**Definition 19 (Layer coherence score).**

$$D_j \;=\; 2\big(\ell_j - \ell_j^{\text{null}}\big) \;=\; 2\sum_{i=0}^{N-2} G_i \;=\; 2\Big(\ell_j + (N-1)\log K\Big)$$

since $\ell^{\text{null}} = \sum_i \log(1/K) = -(N-1)\log K$. ($G_i$ from Def. 13 — $D_j$ is exactly its double-sum.)

- $D_j \gg 0$: language-coherent — the layer's emissions are predictable from the language context → *language-specific in the working-language sense*.
- $D_j \approx 0$: language-agnostic emissions → *neutral*.
- $D_j < 0$: emissions systematically *contradict* language continuity → internally language-incoherent (or dense switching).

This score needs **no ground truth** and reuses the forward-pass normalizers. It is a genuine likelihood ratio, but *not* Wilks-regular (the transitions are data-determined, not free parameters), so calibrate via **Null C**: permute the temporal order of the $(\mathbf{b}_i, T_i)$ pairs (breaking emission–transition alignment while preserving per-position marginals) and recompute $D_j$; the null distribution centers near $0$.

### 3.4 Decision rule and classification

**Definition 20 (Language-specificity decision).** Layer $j$ is **language-specific** iff

$$Z_j \ge z_{\text{crit}} \quad \text{and} \quad C_j \ge C_{\min}$$

(defaults: $z_{\text{crit}} = 2$, $C_{\min} = 0.3$; tune $C_{\min}$ on the observed commitment distribution). Otherwise it is **language-neutral**. Report $D_j$ (Null C calibrated) alongside as the ground-truth-free corroboration.

| | $Z_j \approx 0$ (no tracking) | $Z_j \ge z_{\text{crit}}$ (tracking) |
|---|---|---|
| **$C_j$ high** | *Monolingual shortcut*: committed to a fixed language regardless of truth (e.g. early layers always decoding to English). Check: variance of $\mu_i^{(j)}$ across positions $\approx 0$ while $g$ varies. | **Language-specific (tracking)** |
| **$C_j$ low** | **Language-neutral** | *Weakly tracking*: diffuse but informative — e.g. mid-layers where the language signal is forming but not yet committed. Often neighbors (in depth) of specific layers. |

### 3.5 Connection back to switch detection

The layer weights of Def. 14 should be $w_j = \max(0, Z_j)$ (or the indicator $\mathbb{1}[Z_j \ge z_{\text{crit}}]$): layers classified neutral do not vote on switch boundaries. This closes the loop — layer classification gates the cross-layer switch consensus, and switch boundaries with high $\bar{c}_i$ but detected *only* by low-$Z_j$ layers are demoted.

---

## 4. Computation Notes (shapes and reuse)

| Quantity | Input | Output shape | Already computed by `trellis_layer_analysis`? |
|----------|-------|-------------|----------------------------------------------|
| $d_i, \kappa_i, \rho_i, c_i$ | `lambdas` $(N, M)$ | $(N-1,)$ each | Yes — from `lambdas` |
| $G_i$ | per-step $z_{i+1}$ | $(N-1,)$ | Computed internally; the sum is `log_likelihood` — expose per-step $z$ |
| $D_j$ | `log_likelihood`, $N$, $K$ | scalar | Yes: $D_j = 2(\ell_j + (N{-}1)\log K)$ |
| $C_j$ | `lambdas` | scalar | Yes |
| $T_j, Z_j, p_j$ | `lambdas` + ground truth $g$ | scalars | No — needs $g$ (actual token → lang DB) + permutations |
| $P(\text{switch} \mid \text{all})$ | $\alpha$, $A$, $b$, $S$ | $(N-1,)$ | No — needs backward pass $\beta$ (same cost as forward) |

Ground truth $g_i$: look up the actual generated token (available in the lens JSON) in the token-language database; exclude uniform-fallback positions.

---

## 5. Worked Examples

### 5.1 Switch detection on the code-switch toy (Example 3 of `trellis_formulation.md`)

$N = 5$, $M = 2$ (English, French), trellis $\boldsymbol{\lambda}$ (from the forward pass there):

| $i$ | $\boldsymbol{\lambda}_i$ | $\kappa_i = 2\max - 1$ |
|-----|------------|------------------|
| 0 | $(0.635,\, 0.365)$ | $0.270$ |
| 1 | $(0.606,\, 0.395)$ | $0.212$ |
| 2 | $(0.362,\, 0.638)$ | $0.276$ |
| 3 | $(0.253,\, 0.747)$ | $0.494$ |
| 4 | $(0.500,\, 0.500)$ | $0.000$ |

(true switch: Eng→Fre between positions 1 and 2; position 4 reverts in the lens but the trellis memory pulls it to a tie)

**Switch evidence** $d_i = \mathrm{TV}(\boldsymbol{\lambda}_{i+1}, \boldsymbol{\lambda}_i)$:

| boundary | $d_i$ | marginal-TV counterpart |
|----------|-------|------------------------|
| 0 | $\tfrac12(0.029 + 0.030) = 0.030$ | $0.080$ |
| 1 | $\tfrac12(0.244 + 0.243) = 0.244$ | $0.235$ |
| 2 | $\tfrac12(0.109 + 0.109) = 0.109$ | $0.100$ |
| 3 | $\tfrac12(0.247 + 0.247) = 0.247$ | $0.345$ |

The trellis **suppresses** noise boundaries (0: $0.030$ vs $0.080$; 3: $0.247$ vs $0.345$) while **preserving/enhancing** the true switch (1: $0.244$ vs $0.235$) — the SNR property of §2.2. But naive $d$-thresholding still ranks boundary 3 (the memory-effect tie) *highest* — the confidence factors are needed.

**Confidence** $c_i = d_i \cdot \min(\kappa_i, \kappa_{i+1}) \cdot \rho_i$, window $w = 2$:

| boundary | $d_i$ | $\min(\kappa_i, \kappa_{i+1})$ | $\rho_i$ | $c_i$ |
|----------|-------|-------------------------------|----------|-------|
| 0 | $0.030$ | $\min(0.270, 0.212) = 0.212$ | $(0.362+0.253)/2 = 0.308$ | $0.0019$ |
| 1 | $0.244$ | $\min(0.212, 0.276) = 0.212$ | $(0.747+0.500)/2 = 0.624$ | $\mathbf{0.0322}$ |
| 2 | $0.109$ | $\min(0.276, 0.494) = 0.276$ | $0.500$ (window truncated) | $0.0150$ |
| 3 | $0.247$ | $\min(0.494, 0.000) = 0$ | — (tie) | $0$ |

**Result:** boundary 1 dominates ($c_1 \approx 17\times c_0$, $\approx 2\times c_2$, $\infty \times c_3$). The min-commitment rule exactly kills the boundary-3 false positive (position 4 is uncommitted: $\kappa_4 = 0$). This matches the Viterbi path $(0,0,1,1,1)$ — switch at position 2, i.e. boundary 1.

**Scale caveat.** Absolute $c$ values are small here because $M = 2$ with near-uniform $\boldsymbol{\lambda}$; with $M \approx 203$ and committed posteriors ($\max_m \lambda \in [0.7, 0.99]$), factors are an order of magnitude larger. Threshold $\theta_c$ must be calibrated on the real data (Def. 10), not fixed a priori.

### 5.2 Layer specificity: two-layer toy

$N = 4$, $M = 2$, ground truth $g = (\text{E}, \text{E}, \text{F}, \text{F})$ (a source→target switch at the midpoint).

**Layer A (specific, tracking):**
$\boldsymbol{\lambda}^{\text{A}} = \big((0.9, 0.1), (0.8, 0.2), (0.15, 0.85), (0.1, 0.9)\big)$

- Commitment: $\kappa = (0.8, 0.6, 0.7, 0.8)$ → $C_{\text{A}} = 0.725$
- Tracking: $T_{\text{A}} = (0.9 + 0.8 + 0.85 + 0.9)/4 = 0.8625$; hard accuracy $A_{\text{A}} = 4/4$
- Permutation null (all $\binom{4}{2} = 6$ distinct alignments): $T^{\sigma} \in \{0.8625, 0.5375, 0.5125, 0.4875, 0.4625, 0.1375\}$, mean $0.5$, $\hat{\sigma} = 0.2105$ → $Z_{\text{A}} = (0.8625 - 0.5)/0.2105 = 1.72$, exact $p = 1/6$ (observed is the maximum)

**Layer B (neutral):**
$\boldsymbol{\lambda}^{\text{B}} = \big((0.45, 0.55), (0.52, 0.48), (0.48, 0.52), (0.55, 0.45)\big)$

- Commitment: $\kappa = (0.1, 0.04, 0.04, 0.1)$ → $C_{\text{B}} = 0.07$
- Tracking: $T_{\text{B}} = (0.45 + 0.52 + 0.52 + 0.45)/4 = 0.485$; hard accuracy $A_{\text{B}} = 2/4$ (chance)
- Permutation null: $T^{\sigma} \in \{0.485, 0.465, 0.5, 0.5, 0.535, 0.515\}$, mean $0.5$, $\hat{\sigma} = 0.0220$ → $Z_{\text{B}} = -0.68$, $p \approx 0.83$

**Decision** ($z_{\text{crit}} = 2$, $C_{\min} = 0.3$): B is clearly **neutral** ($C_{\text{B}} = 0.07 \ll 0.3$, $Z_{\text{B}} < 0$). A fails the $z$ cutoff *only because $N = 4$ gives the permutation test minimal power* (minimum achievable $p$ is $1/6$). Same effect size at realistic $N = 61$: $Z$ scales $\propto \sqrt{N}$, so $Z_{\text{A}} \approx 1.72 \times \sqrt{61/4} \approx 6.7$ — decisively specific.

### 5.3 Micro-example: the likelihood-ratio statistic $G_i$

$K = 2$, $N = 2$: $\boldsymbol{\alpha}_0 = (1, 0)$ (token 0 certain), $A_0 = \begin{pmatrix} 0.9 & 0.1 \\ 0.2 & 0.8 \end{pmatrix}$.

**Coherent continuation** ($\mathbf{b}_1 = (0.9, 0.1)$, same language):
$z_1 = \langle \mathbf{b}_1, A_0^T \boldsymbol{\alpha}_0 \rangle = 0.9 \cdot 0.9 + 0.1 \cdot 0.1 = 0.82$, so $G_0 = \log(2 \cdot 0.82) = \log 1.64 = +0.495$ — language-coherent, no switch.

**Switched continuation** ($\mathbf{b}_1 = (0.1, 0.9)$, other language):
$z_1 = 0.1 \cdot 0.9 + 0.9 \cdot 0.1 = 0.18$, so $G_0 = \log(2 \cdot 0.18) = \log 0.36 = -1.022$ — the emission contradicts the language context: strong switch evidence, visible in $G_i$ even though both states remain possible.

The sign flip between the two scenarios illustrates why $G_i$ is the sharpest per-boundary discriminator, while $D_j$ aggregates it into a per-layer coherence score.

---

## 6. Assumptions, Limitations, Extensions

**Assumptions.**

1. The trellis model itself (Markov transitions from language similarity) — all metrics inherit its modeling choices; in particular the smoothed posterior (Def. 12) is only as good as the transition model.
2. Permutation nulls assume exchangeability under $H_0$; autocorrelated sequences need block permutations.
3. The product form of $c_i$ treats the three conditions as (approximately) independent necessary events.

**Limitations.**

1. **Lens blind spot:** a layer may process language in ways that do not surface in the unembedding space at that depth; the metrics can only see what the lens sees.
2. **Multiple testing:** $N-1$ boundaries × $J$ layers of statistics; per-boundary inference needs FDR or Bonferroni control.
3. **Calibration:** all thresholds ($\theta_c$, $C_{\min}$, $z_{\text{crit}}$) must be calibrated on real data — the toys show order-of-magnitude effects, not calibrated scales.
4. **Constant references:** for pure single-target translation ($g$ constant), Null B degenerates and the tracking axis reduces to the uniform baseline; the coherence axis ($D_j$) carries the discriminative weight.

**Extensions.**

1. Expose per-step $z_{i+1}$ in `trellis_layer_analysis` to get $G_i$ and $D_j$ for free.
2. Add the backward pass for the smoothed switch posterior (Def. 12) — same asymptotic cost as the forward pass.
3. Weighted variants: replace TV with Rényi/χ² divergences; replace min-commitment with geometric mean if softer confidence is desired.
4. Multi-sample aggregation: run the metrics across many samples and report the distribution of switch onset depths $j^{*}_i$ — a per-layer "language resolution profile" of the model.

