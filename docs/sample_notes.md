# My Notes on Machine Learning

## Gradient Descent
Gradient descent is an optimization algorithm used to minimize a loss function.
It works by iteratively moving in the direction of steepest descent as defined
by the negative of the gradient. The learning rate controls how big each step is.
A small learning rate means slow but stable convergence; too large and it may overshoot.

## Transformers
The Transformer architecture was introduced in "Attention is All You Need" (2017).
It relies entirely on attention mechanisms instead of RNNs. The key idea is
self-attention: each token attends to every other token in the sequence.
This allows parallelization during training, unlike sequential RNNs.

## Embeddings
Embeddings convert discrete objects (words, sentences) into dense vectors.
Similar items end up close together in vector space. Word2Vec, GloVe, and
sentence-transformers are common embedding approaches. They are the backbone
of semantic search and RAG (retrieval-augmented generation) systems.

## RAG (Retrieval-Augmented Generation)
RAG combines a retrieval system with a language model. Given a question:
1. Embed the question into a vector
2. Search a vector database for the most similar document chunks
3. Pass those chunks as context to the LLM
4. The LLM answers using only the retrieved context
This grounds the LLM in your actual documents and reduces hallucination.
