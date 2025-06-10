# Improving Fairness in Recommender Systems with Multi-subgroup Debiasing and DNCF

## 🔍 Overview

This project implements **DNCF (Debiased Neural Collaborative Filtering)**, a fairness-enhanced recommender system. DNCF supports both **binary** and **multi-subgroup** debiasing and regularization, improving equity across demographic groups while maintaining predictive performance.


## 📈 Motivation

Recommender systems often encode societal biases due to skewed historical data. This project aims to:
- Mitigate unfair treatment across **gender and age groups**.
- Extend existing models (MF, NCF, NFCF) to **multi-subgroup fairness**.
- Evaluate trade-offs between fairness and performance.


## ⚙️ Models Implemented

- **MF (Matrix Factorization)**
- **NCF (Neural Collaborative Filtering)**
- **NFCF (Neural Fair Collaborative Filtering)** – baseline gender debiasing
- **DNCF (Proposed Model)** – supports:
  - Multi-subgroup debiasing
  - Composite fairness regularization


## 🧪 Dataset

- **MovieLens 1M**
  - 1 million ratings from 6,000 users and 4,000 movies
  - Subgroup labels:
    - **Gender**: male/female
    - **Binary Age**: young adults vs. middle-aged
    - **Multi Age**: 6 groups (18–56+)


## 🧩 Methodology

### Architecture

- Based on NCF with embedding size of 128 (MF) or 256 (NCF)
- Two-stage training:
  1. Movie recommendation pre-training
  2. Career recommendation fine-tuning with debiased embeddings

### Debiasing

- **Binary subgroups**: Remove directional bias vector between group means
- **Multi-subgroups**: Remove projection on average deviation from subgroup centroids

### Fairness Metrics

- **ϵ-mean**: Differential fairness across subgroups (lower is fairer)
- **U_abs**: Measures output difference between advantaged and disadvantaged groups


## 📊 Results Summary

| Model | HR@10 | NDCG@10 | ϵ-mean (Gender) | ϵ-mean (Binary Age) | ϵ-mean (Multi Age) |
|-------|--------|----------|-------------------|------------------------|------------------------|
| MF     | 0.546 | 0.837   | 0.119             | 0.028                  | 1.843                  |
| NCF    | 0.518 | 0.805   | 0.107             | 0.151                  | 0.877                  |
| DNCF   | 0.519 | 0.806   | **0.091**         | **0.106**              | **0.825**              |

- DNCF improves fairness **without harming performance**
- **Multi-subgroup regularization** further enhances fairness at small performance cost


## 🖼️ Visualizations

- **PCA** plots show reduced embedding bias after debiasing
- **Career recommendations** show more consistent and fair outcomes across subgroups


## 🧠 Future Directions

- Dynamic embedding updates for evolving users
- Overlapping subgroup fairness (e.g., young & female)
- Real-time fairness-aware recommendation

