# 🚀 Enterprise-Grade Hybrid RAG with Automated Evaluation Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An end-to-end production architecture implementing **Hybrid Retrieval (Dense + BM25)**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Re-ranking**, and automated benchmark metrics (**RAGAS Framework**).

Designed to eliminate hallucination rates and optimize Context Recall for domain-specific knowledge bases.

---

## 🔬 Architecture Overview

```mermaid
graph TD
    UserQuery[User Query] --> SparseBranch[BM25 Lexical Retrieval]
    UserQuery --> DenseBranch[Dense Embeddings Vector Search]
    
    SparseBranch --> RRF[Reciprocal Rank Fusion - RRF]
    DenseBranch --> RRF
    
    RRF --> TopK[Top-N Candidate Chunks]
    TopK --> Reranker[BAAI/bge-reranker-large Cross-Encoder]
    Reranker --> LLM[LLM Context Injection]
    LLM --> Response[Final Answer + Citations]