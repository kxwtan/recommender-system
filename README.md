# Recommender System

Based on [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031) paper for recommending movie and Pinterest posts according to user interactions.

The model is built on PyTorch and combines the more simple Matrix Factorization and the more complex Multi-Layer perceptron to learn both linear and non-linear relationships between user interactions. It was made clear by the paper that this results in better HR@10 (Hit Rate @ 10) and NDCG@10 (Normalized Discounted Cumulative Gain @ 10) scores.

The model was able to achieve a HR@10 of almost 0.7 and a NDCG@10 of around 0.4 which aligns with the scores found in the research paper.