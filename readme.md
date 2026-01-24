# Makemore: From Scratch (Notes & Intuitions)

**Goal:** Re-implementing Andrej Karpathy's `makemore` series to understand the "atoms" of Large Language Models.
**Philosophy:** This repo is less about the code (which is standard) and more about the *intuitions* on whyV we use certain concepts and methods

---

## Table of Contents
1. [Part 1: Bigram Language Model](#part-1-bigram-language-model)
2. [Part 2: MLP](#part-2-mlp)

---

## Part 1: Bigram Language Model
*Predicting the next character given only the immediate previous one.*

### The Core Fascinating Moments

#### 1. Two approaches, same outcome
The most fascinating realization was that **Counting** and **Neural Networks** are mathematically identical in this specific case.
* **Approach A (Counting):** We explicitly count how often 'b' follows 'a' in a table.
* **Approach B (Neural Net):** We use a single linear layer (`x @ W`).
* **The Connection:** The Counting approach calculates the global solution instantly (Analytical). The Neural Net approaches that same solution step-by-step (Iterative). We use the Neural Net approach because the "Counting" approach becomes impossible when you have 10,000 words of context (the table would be too big). The Neural Net scales; the table doesn't.

#### 2. The "Loss" Function (Negative Log Likelihood)
Why NLL?
* **Goal:** Maximize the probability of the *correct* next character.
* **Problem:** Probabilities are multiplied ($p_1 \times p_2 \dots$), leading to tiny numbers.
* **Fix:** Take the `log`. Multiplication becomes addition.
* **Inversion:** Since `log(p)` is negative (for $p < 1$), we flip the sign to minimize "Loss."
* **Intuition:** NLL is just a fancy way of saying "Make the probability of the correct answer as close to 1 as possible."

#### 3. Softmax is the "Thermostat"
Softmax turns raw "logits" (scores) into probabilities (sum to 1).

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$

* **Insight:** It creates a **competition**. If one logit rises, the others *must* fall. This "suppression" dynamic is critical for how models make decisions.

#### 4. Regularization = Smoothing
* **In Counting:** We add `+1` to counts so we never have zero probability (infinite loss).
* **In Neural Nets:** We use "Weight Decay" (pulling $W$ towards 0).
* **The Link:** If weights are 0, logits are identical, and probabilities are uniform (maximum smoothing). Regularization is just forcing the model to be "humble" and spread its bets.

#### 5. The "Mechanics" (Python Implementation)
* **Matrix Multiplication as Indexing:** x @ W acts as a lookup table when x is one-hot. It selects the "voting row" for the current character.

* **Broadcasting:** Using keepdim=True allows us to divide a [5, 27] matrix by a [5, 1] vector. Python "stretches" the sum across the row to normalize every probability.

* **Zip:** Used to slide the window over the text: zip(chars, chars[1:]).

### Code Snippet: The Training Loop
```python
# The pattern that repeats in every single Transformer

# 1. Forward Pass
logits = xenc @ W                              # [5, 27] (Batch of 5, 27 log-counts)
counts = logits.exp()                          # [5, 27] (Exponentiate to get positive counts)
probs = counts / counts.sum(1, keepdims=True)  # [5, 27] (Normalize rows to sum to 1)

# 2. Loss Calculation
# Pick the probability assigned to the CORRECT character for each row
correct_probs = probs[torch.arange(num), ys]   # [5] vector of probabilities
loss = -correct_probs.log().mean()             # Scalar float (Average NLL)
reg_loss = loss + 0.1 * (W**2).mean()          # Add regularisation to penalise extreme weights and for a more uniform distribution

# 3. Backward Pass
W.grad = None                                  # Zero out old gradients
loss.backward()                                # Calculate new gradients
W.data += -lr * W.grad                         # Update weights (Step down the hill)
```

## Part 2: MLP

## The Intuition

Instead of just counting which letter likely follows another, this model actually "learns" features about the characters. It looks at a context window (e.g., the previous 3 characters) to predict the 4th.

### 1. The Setup (B, T, C)

The hardest part of this project was understanding the tensor shapes. I realized the data isn't just a flat list; it lives in three dimensions:

* **Batch (B):** The stack of examples we process at once (32 in my case).
* **Time (T):** How far back we look (the 3 previous characters).
* **Channels (C):** The information density of each character (the embedding).

### 2. The Flattening

To make the data fit into the neural network, I had to flatten it.

* **The naive way:** Splitting the tensor apart and sticking it back together (using `unbind` and `cat`). This is wasteful because it creates new memory.
* **The efficient way:** Using `.view()`. This was a big realization, we can just treat the memory block as a different shape without actually moving any data. It’s instantaneous.

### 3. The Classroom Analogy (Mini-batches)

I also explored why we train on small batches (32 items) instead of the whole dataset (30,000 items).

#### This analogy helped me understand it the most:

Think of it like a teacher grading exams:

* **Full Batch:** You grade all 30,000 students before changing your lesson plan. It's accurate, but you only improve once a semester.
* **Mini-batch:** You grade 32 random students and immediately adjust the lesson plan. It's noisy and approximate (the "drunk" walk), but you can improve thousands of times faster. Speed wins.

## Results

After training for roughly 100,000 steps, I managed to get the loss down to about **2.31**. The model is now generating names that sound surprisingly plausible compared to the random gibberish I started with.