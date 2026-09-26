import math
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

class HybridRAGRetriever:
    """
    Engine de Búsqueda Híbrida de Grado Empresarial.
    Combina BM25 (Búsqueda Léxica) y Dense Vector Embeddings (Búsqueda Semántica)
    utilizando Reciprocal Rank Fusion (RRF) y Cross-Encoder Re-ranking.
    """
    def __init__(
        self,
        documents: List[Dict[str, Any]],
        dense_retriever_func,
        cross_encoder_model: str = "BAAI/bge-reranker-large",
        rrf_k: int = 60
    ):
        self.documents = documents
        self.dense_retriever_func = dense_retriever_func
        self.rrf_k = rrf_k
        
        # Inicialización de BM25 (Lexical)
        corpus_tokens = [doc["content"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus_tokens)
        
        # Inicialización del Reranker Cross-Encoder
        self.reranker = CrossEncoder(cross_encoder_model)

    def _lexical_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx].copy()
            doc["score"] = scores[idx]
            doc["rank"] = rank + 1
            results.append(doc)
        return results

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Calcula el puntaje RRF: RRF_Score(d) = \sum (w / (k + rank(d)))
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Procesar Dense
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 - alpha) * (1.0 / (self.rrf_k + rank))

        # Procesar Sparse
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + alpha * (1.0 / (self.rrf_k + rank))

        # Ordenar por puntaje RRF
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def retrieve(self, query: str, top_k: int = 10, rerank_top_k: int = 3, alpha: float = 0.5) -> List[Dict[str, Any]]:
        # 1. Recuperación en paralelo (Candidate Retrieval)
        dense_candidates = self.dense_retriever_func(query, top_k=top_k * 2)
        sparse_candidates = self._lexical_search(query, top_k=top_k * 2)

        # 2. Fusión mediante RRF
        fused_candidates = self._reciprocal_rank_fusion(dense_candidates, sparse_candidates, alpha=alpha)
        top_fused = fused_candidates[:top_k]

        # 3. Segunda Etapa: Re-ranking con Cross-Encoder
        pairs = [[query, doc["content"]] for doc in top_fused]
        rerank_scores = self.reranker.predict(pairs)

        for idx, score in enumerate(rerank_scores):
            top_fused[idx]["rerank_score"] = float(score)

        reranked_docs = sorted(top_fused, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs[:rerank_top_k]