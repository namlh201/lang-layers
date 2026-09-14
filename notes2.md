To accurately construct a token-level language mapping $P(\text{Lang} \mid t)$ for a specific tokenizer (e.g., Llama-3, Qwen, Mistral), you must compute the probability that a specific subword token $t$ originates from a given language $L_a$.

Because LLM tokenizers split text into subword fragments (BPE, SentencePiece, or Unigram), you cannot simply pass tokens to a standard text language detector (like FastText). Instead, you must reverse the generative process: **estimate token frequency across massive monolingual corpora.**

---

## The Core Bayesian Formula

By Bayes' Theorem, the probability that a token $t$ represents language $L_a$ is:

$$P(L_a \mid t) = \frac{P(t \mid L_a) \cdot P(L_a)}{P(t)} = \frac{P(t \mid L_a) \cdot P(L_a)}{\sum_{m} P(t \mid L_m) \cdot P(L_m)}$$

Assuming an uninformative prior across target languages ($P(L_1) = P(L_2) = \dots = P(L_m)$), this simplifies to the relative frequency of the token in monolingual text:

$$P(L_a \mid t) = \frac{f(t, L_a)}{\sum_{m=1}^M f(t, L_m)}$$

Where $f(t, L_a)$ is the **normalized relative frequency** (tokens per million) of token $t$ in a clean corpus of language $L_a$.

---

## Step-by-Step Construction Pipeline

### Step 1: Gather Monolingual Corpora

Download balanced, clean monolingual corpora for each language $L_a \in \{L_1, \dots, L_m\}$ you want to track.

* **Recommended Datasets:** Wikipedia dumps, OSCAR, or CC100.
* **Corpus Size:** Aim for roughly equal size per language (e.g., 50MB–200MB of raw text per language).

### Step 2: Tokenize and Count Frequency $f(t, L_a)$

Tokenize each corpus using the **exact target tokenizer** and count how many times each token ID appears. Convert raw counts into relative frequencies to account for minor size variations in your dataset:

$$\text{freq}(t, L_a) = \frac{\text{Count}(t \text{ in } L_a)}{\text{Total Tokens in } L_a}$$

```python
from collections import Counter
from transformers import AutoTokenizer

def build_language_frequencies(tokenizer_name, corpora_by_lang):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    
    # Store normalized frequencies per language
    lang_freqs = {} 

    for lang, text_samples in corpora_by_lang.items():
        counter = Counter()
        total_tokens = 0
        
        for text in text_samples:
            # Encode raw text with the exact tokenizer
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            counter.update(token_ids)
            total_tokens += len(token_ids)
            
        # Store relative frequencies with Add-1 / Laplace Smoothing
        lang_freqs[lang] = {
            t_id: (counter[t_id] + 1) / (total_tokens + vocab_size)
            for t_id in range(vocab_size)
        }
        
    return lang_freqs, vocab_size

```

### Step 3: Compute $P(\text{Lang} \mid \text{token})$ Matrix

Construct the conditional probability dictionary mapping every token ID $t$ to its language distribution vector:

```python
import numpy as np

def construct_token_language_dict(lang_freqs, vocab_size, languages):
    # Map token_id -> array of P(Lang_a | token_id)
    p_lang_given_token = {}
    
    for t_id in range(vocab_size):
        # Gather frequencies across all languages for token t_id
        freq_vector = np.array([lang_freqs[lang][t_id] for lang in languages])
        
        # Normalize across languages to get P(Lang | token)
        prob_vector = freq_vector / np.sum(freq_vector)
        
        p_lang_given_token[t_id] = {
            lang: prob_vector[i] for i, lang in enumerate(languages)
        }
        
    return p_lang_given_token

```

---

## Edge Cases & Advanced Refinements

A naive frequency mapping will fail on shared tokens, punctuation, and subwords. Apply the following refinements to avoid noisy data:

### 1. Punctuation, Digits, and Byte Tokens

Tokens like `.` `,` `123`, `\n`, or whitespace are universal across languages. Their raw frequency might heavily skew toward one corpus depending on dataset formatting.

* **Fix:** Hardcode uniform probability ($P(L_a \mid t) = \frac{1}{M}$) for pure digits, spaces, and punctuation tokens.

```python
import string

def is_language_agnostic(token_str):
    # Strip leading/trailing space markers common in BPE/SentencePiece (e.g., 'Ġ', ' ')
    clean_str = token_str.replace("Ġ", "").replace(" ", "").strip()
    if not clean_str:
        return True # Whitespace
    if all(char in string.punctuation or char.isdigit() for char in clean_str):
        return True # Numbers or symbols
    return False

```

### 2. Byte-Fallback Tokens (e.g., `<0x41>`)

Tokenizers with byte fallback map unseen UTF-8 bytes to individual byte tokens.

* Non-ASCII languages (e.g., Chinese, Arabic, Cyrillic) rely on sequences of multi-byte tokens.
* **Fix:** If a token is a single byte (e.g., `<0xD0>`), evaluate its frequency in raw byte sequences per language rather than as decoded Unicode characters.

### 3. Shared Subword Roots (Homographs)

Subwords like `in`, `de`, or `con` exist in English, Spanish, French, and Latin.

* **Fix:** Ensure you apply the **Language Entropy Weighting** ($w(t_j)$) from step 1 in your logit-lens detection pipeline. $w(t_j)$ will naturally penalize these high-entropy tokens, automatically preventing them from degrading your working language detection.

---
