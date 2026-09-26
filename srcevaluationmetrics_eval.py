import time
import pandas as pd
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
from datasets import Dataset

class RAGEvaluator:
    """
    Suite de evaluación automatizada para pipelines RAG.
    Mide precisión de contexto, fidelidad de la respuesta y latencia end-to-end.
    """
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline

    def run_benchmark(self, test_dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        queries, answers, contexts, ground_truths, latencies = [], [], [], [], []

        for item in test_dataset:
            start_time = time.time()
            
            # Ejecución del pipeline
            retrieved_docs, generated_answer = self.rag_pipeline.run(item["query"])
            
            latency = time.time() - start_time
            
            queries.append(item["query"])
            answers.append(generated_answer)
            contexts.append([doc["content"] for doc in retrieved_docs])
            ground_truths.append(item["ground_truth"])
            latencies.append(latency)

        # Crear dataset en formato HuggingFace para RAGAS
        hf_dataset = Dataset.from_dict({
            "question": queries,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        # Evaluar con RAGAS
        metrics_results = evaluate(
            dataset=hf_dataset,
            metrics=[faithfulness, answer_relevance, context_precision, context_recall]
        )

        df_results = metrics_results.to_pandas()
        df_results["latency_seconds"] = latencies
        return df_results