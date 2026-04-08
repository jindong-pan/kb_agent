The Zettelkasten method was invented by German sociologist Niklas Luhmann in the 1950s–80s. He wrote over 70 books and 400 papers using it, and credited the system entirely. "Zettelkasten" literally means **slip-box** — a physical box of index cards, one idea per card.

The digital version maps perfectly onto a knowledge base.

---

**The core philosophy**

Most note-taking captures information. Zettelkasten captures **understanding**.

The difference: instead of pasting an article into your notes, you read it, digest it, then write the idea *in your own words* as a standalone atomic note. That act of rewriting forces comprehension. And because the note must stand alone, you can never hide behind vague summaries.

---

**The three note types**

**1. Fleeting notes** — raw capture, temporary

Quick captures during the day. Voice memos, scraps, highlights. Not meant to survive long-term. You process these within a day or two into permanent notes.

```
[fleeting]
interesting idea from paper: sparse attention only attends 
to local + global tokens, not all pairs → O(n) not O(n²)
```

**2. Literature notes** — what a source says

One note per source. Written in your own words. What did this book/paper/video actually say? Brief, no direct quotes.

```
## Literature: "Attention is All You Need" (Vaswani 2017)

The paper proposes replacing RNNs entirely with attention mechanisms.
The key insight is that attention can be parallelized during training
unlike sequential RNN steps. They introduce multi-head attention to
let the model attend to different representation subspaces simultaneously.
Performance beat state-of-the-art on translation tasks at lower compute cost.

Source: https://arxiv.org/abs/1706.03762
```

**3. Permanent notes** ← the heart of the system

One idea. Your own words. Written as if for a reader with no context. These are what go into your knowledge base.

```
## Sparse attention reduces transformer complexity from O(n²) to O(n)

Standard self-attention computes relationships between every pair of tokens,
making it O(n²) in sequence length. Sparse attention restricts each token
to attend only to a local window of nearby tokens plus a few global tokens.
This drops complexity to O(n), making very long sequences feasible.

The tradeoff: some long-range relationships may be missed if they fall
outside the local window. Works well when locality is a reasonable assumption
(e.g. text, not random-access data structures).

Related: [[attention mechanism]], [[transformer scalability]], [[Longformer]]
Source: literature/attention-is-all-you-need.md
```

---

**The atomic principle — in detail**

"Atomic" means the note is **indivisible** — it contains exactly one idea that can be understood without reading anything else.

A simple test: cover the title and read only the body. Does it make complete sense? If you need to look something up to understand it, it's not atomic enough.

**Not atomic** — two ideas crammed together:
```
## Transformers and RNNs

Transformers use attention and can be parallelized. RNNs process 
tokens sequentially which is slow. LSTMs solve the vanishing 
gradient problem that plain RNNs have. Transformers have largely 
replaced RNNs for NLP tasks.
```

**Atomic** — split into three separate notes:
```
## Transformers are parallelizable, RNNs are not

Transformers process all tokens simultaneously using attention,
making training parallelizable across GPUs. RNNs process one token
at a time sequentially, so each step depends on the previous —
parallelization is impossible. This is the primary reason transformers
train faster at scale despite similar parameter counts.

Related: [[transformer architecture]], [[RNN limitations]]
```

```
## LSTMs solve the vanishing gradient problem in plain RNNs

In plain RNNs, gradients shrink exponentially as they backpropagate
through many timesteps, making it impossible to learn long-range
dependencies. LSTMs introduce a cell state with additive updates
and gating mechanisms that allow gradients to flow unchanged over
long sequences.

Related: [[RNN limitations]], [[backpropagation through time]]
```

```
## Transformers have largely replaced RNNs for NLP

Since 2018, transformer-based models (BERT, GPT, T5) have set
state-of-the-art results on nearly every NLP benchmark, displacing
LSTM and GRU architectures. The exception is edge/embedded devices
where transformers are too large and RNNs remain practical.

Related: [[transformer architecture]], [[LSTM]]
```

This is why atomicity helps retrieval so much — each note produces a sharp, focused vector. When you ask "why are transformers faster to train?", the first note is retrieved with high confidence, not a diluted chunk mixing four different ideas.

---

**The linking system**

Every permanent note links to related notes using `[[double brackets]]`. This creates a web of knowledge rather than a hierarchy of folders.

```
## Attention mechanism

Each token computes a query vector, and attends to other tokens
by measuring dot-product similarity with their key vectors.
The resulting weights (after softmax) determine how much of each
token's value vector to aggregate into the output representation.

Related: [[self-attention vs cross-attention]], [[multi-head attention]], 
          [[attention complexity O(n²)]], [[transformer architecture]]
Source: literature/attention-is-all-you-need.md
Created: 2024-03-15
```

For your vector database the `[[links]]` don't do anything mechanical — but they're valuable for two reasons: they force you to think about where this idea fits, and they make the note denser with relevant terminology which improves embedding quality.

---

**A practical folder structure**

```
knowledge/
├── fleeting/
│   └── 2024-03-15.md        ← daily scratch pad, cleared weekly
├── literature/
│   ├── attention-paper.md
│   └── deep-learning-book.md
└── permanent/               ← this is what you ingest
    ├── attention-mechanism.md
    ├── sparse-attention.md
    ├── lstm-vanishing-gradient.md
    └── transformer-vs-rnn-speed.md
```

Only ingest `permanent/` into your vector database. Fleeting and literature notes are too noisy — they contain mixed ideas and other people's words, which weakens retrieval.

---

**The writing habit**

The system only works if you process fleeting notes regularly. A common rhythm:

- **During the day** — capture fleeting notes freely, no polish
- **End of day (10–15 min)** — turn the best fleeting notes into permanent notes, archive or delete the rest
- **Weekly** — review new permanent notes, add links to older ones

The goal isn't to capture everything — it's to capture what you've actually understood well enough to explain in your own words. That constraint is what makes the notes useful for retrieval later.
