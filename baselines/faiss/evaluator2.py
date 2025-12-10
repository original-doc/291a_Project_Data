#!/usr/bin/env python3
"""
Evaluator2 for FAISS baseline.

Matches retrieved results to ground truth passages using the same
embedding-based cosine similarity approach as RAG/evaluation/evaluator.py,
computes recall/precision/NDCG@K and MRR, and saves aggregated metrics in
the same JSON schema as RAG/evaluation_results.json.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Allow importing the RAG embedder
SCRIPT_DIR = Path(__file__).parent
RAG_ROOT = (SCRIPT_DIR.parent.parent / "RAG").resolve()
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

try:
    from embeddings.code_embedder import UniXcoderEmbedder  # type: ignore
except Exception as exc:  # pragma: no cover
    logger.warning("Falling back to no-op embedder due to import error: %s", exc)
    UniXcoderEmbedder = None  # type: ignore


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class EvaluationQuery:
    query_id: str
    query_text: str
    relevant_ids: List[int]
    relevant_texts: List[str]
    relevance_scores: Optional[List[float]] = None


@dataclass
class RetrievalMatch:
    retriever_id: str
    text: str
    score: float
    matched_gt_id: Optional[int]


@dataclass
class EvaluationResult:
    query_id: str
    matches: List[RetrievalMatch]
    relevant_ids: Set[int]
    metrics: Dict[str, float]
    latency_ms: float


# -----------------------------
# Evaluator
# -----------------------------
class Evaluator2:
    def __init__(self, k_values: List[int] = [1, 3, 5, 10], sim_threshold: float = 0.3):
        self.k_values = k_values
        self.sim_threshold = sim_threshold
        self.embedder = self._init_embedder()

    def _init_embedder(self):
        """Initialize UniXcoder embedder (same as RAG)."""
        if UniXcoderEmbedder is None:
            return None
        try:
            return UniXcoderEmbedder(device="auto")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize UniXcoderEmbedder: %s", exc)
            return None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        if self.embedder is None:
            return np.empty((len(texts), 0))
        return self.embedder.embed(texts)

    @staticmethod
    def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.clip(a_norm @ b_norm.T, -1.0, 1.0)

    # Metric helpers (mirrors RAG evaluator)
    def recall_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        if not relevant:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        return len(retrieved_at_k & relevant) / len(relevant)

    def precision_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        if k == 0:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        return len(retrieved_at_k & relevant) / k

    def mrr(self, retrieved: List[int], relevant: Set[int]) -> float:
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def dcg_at_k(self, retrieved: List[int], relevance_map: Dict[int, float], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance_map.get(doc_id, 0.0)
            dcg += (2**rel - 1) / np.log2(i + 2)
        return dcg

    def ndcg_at_k(
        self, retrieved: List[int], relevant: Set[int], relevance_scores: Optional[List[float]], k: int
    ) -> float:
        if not relevant:
            return 0.0

        relevant_list = sorted(list(relevant))
        if relevance_scores is None:
            relevance_map = {doc_id: 1.0 for doc_id in relevant_list}
        else:
            relevance_map = {}
            for i, doc_id in enumerate(relevant_list):
                relevance_map[doc_id] = relevance_scores[i] if i < len(relevance_scores) else 1.0

        dcg = self.dcg_at_k(retrieved, relevance_map, k)

        ideal_ranking = sorted(relevance_map.items(), key=lambda x: x[1], reverse=True)[:k]
        ideal_ids = [doc_id for doc_id, _ in ideal_ranking]
        idcg = self.dcg_at_k(ideal_ids, relevance_map, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate_single(
        self, query: EvaluationQuery, matches: List[RetrievalMatch], latency_ms: float = 0.0
    ) -> EvaluationResult:
        relevant_ids = set(query.relevant_ids)
        matched_gt_ids = [m.matched_gt_id for m in matches if m.matched_gt_id is not None]

        metrics: Dict[str, float] = {}
        for k in self.k_values:
            metrics[f"recall@{k}"] = self.recall_at_k(matched_gt_ids, relevant_ids, k)
            metrics[f"precision@{k}"] = self.precision_at_k(matched_gt_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(
                matched_gt_ids, relevant_ids, query.relevance_scores, k
            )
        metrics["mrr"] = self.mrr(matched_gt_ids, relevant_ids)

        return EvaluationResult(
            query_id=query.query_id,
            matches=matches,
            relevant_ids=relevant_ids,
            metrics=metrics,
            latency_ms=latency_ms,
        )

    def _match_retrieved_to_groundtruth(
        self, retrieved_texts: List[str], gt_texts: List[str]
    ) -> List[Optional[int]]:
        if not retrieved_texts or not gt_texts:
            return [None] * len(retrieved_texts)

        retrieved_embeddings = self._embed_texts(retrieved_texts)
        gt_embeddings = self._embed_texts(gt_texts)
        sim_matrix = self._cosine_sim_matrix(retrieved_embeddings, gt_embeddings)
        best_gt_idx = sim_matrix.argmax(axis=1) if sim_matrix.size else np.zeros(len(retrieved_texts), dtype=int)
        best_similarity = sim_matrix.max(axis=1) if sim_matrix.size else np.zeros(len(retrieved_texts))

        matches: List[Optional[int]] = []
        for similarity, gt_idx in zip(best_similarity, best_gt_idx):
            if similarity >= self.sim_threshold:
                matches.append(int(gt_idx))
            else:
                matches.append(None)
        return matches

    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        if not results:
            return {}

        all_metrics: Dict[str, List[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                all_metrics.setdefault(metric, []).append(value)

        aggregated = {
            "num_queries": len(results),
            "mean_latency_ms": float(np.mean([r.latency_ms for r in results])),
            "metrics": {},
        }

        for metric, values in all_metrics.items():
            aggregated["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return aggregated

    def _build_query(self, result_entry: Dict[str, Any], ground_truth: Dict[str, Any]) -> EvaluationQuery:
        relevant_docs = ground_truth.get("relevant_docs", [])
        relevant_texts = [doc.get("text", "") for doc in relevant_docs]
        relevant_ids = list(range(len(relevant_docs)))
        relevance_scores = [doc.get("score", 1.0) for doc in relevant_docs] if relevant_docs else None

        query_id = str(result_entry.get("query_id", ground_truth.get("id", result_entry.get("query", ""))))

        return EvaluationQuery(
            query_id=query_id,
            query_text=result_entry.get("query", ""),
            relevant_ids=relevant_ids,
            relevant_texts=relevant_texts,
            relevance_scores=relevance_scores,
        )

    def evaluate_from_files(
        self,
        results_path: Path,
        ground_truth_path: Path,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Loading retrieval results: {results_path}")
        retrieval_results = json.loads(Path(results_path).read_text())

        logger.info(f"Loading ground truth: {ground_truth_path}")
        ground_truth = json.loads(Path(ground_truth_path).read_text())
        gt_by_query = {item.get("query"): item for item in ground_truth if "query" in item}

        evaluated: List[EvaluationResult] = []
        skipped = 0

        for result in retrieval_results:
            query_text = result.get("query")
            if not query_text or query_text not in gt_by_query:
                skipped += 1
                continue

            gt_entry = gt_by_query[query_text]
            query = self._build_query(result, gt_entry)

            retrieved_docs = result.get("retrieved_docs", [])
            scores = result.get("scores", [])

            retrieved_texts: List[str] = []
            for doc in retrieved_docs:
                display = doc.get("display", "")
                combined = f"{display}".strip()
                retrieved_texts.append(combined if combined else display)

            matched_gt_ids = self._match_retrieved_to_groundtruth(retrieved_texts, query.relevant_texts)

            matches: List[RetrievalMatch] = []
            for i, doc in enumerate(retrieved_docs):
                score = float(scores[i]) if i < len(scores) else 0.0
                matches.append(
                    RetrievalMatch(
                        retriever_id=str(doc.get("index", i)),
                        text=retrieved_texts[i] if i < len(retrieved_texts) else "",
                        score=score,
                        matched_gt_id=matched_gt_ids[i] if i < len(matched_gt_ids) else None,
                    )
                )

            latency = float(result.get("latency", 0.0)) * 1000.0  # convert seconds to ms
            evaluated.append(self.evaluate_single(query=query, matches=matches, latency_ms=latency))

        logger.info(f"Evaluated {len(evaluated)} queries (skipped {skipped})")
        aggregated = self._aggregate_results(evaluated)

        if output_path:
            self.save_results(aggregated, output_path)

        return aggregated

    @staticmethod
    def save_results(results: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {path}")


def main():
    script_dir = Path(__file__).parent
    results_path = script_dir / "faiss_result" / "retrieval_results.json"
    ground_truth_path = script_dir / ".." / ".." / "requests" / "new_requests.json"
    output_path = script_dir / "faiss_result" / "evaluation_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Retrieval results not found: {results_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")

    evaluator = Evaluator2()
    evaluator.evaluate_from_files(results_path, ground_truth_path, output_path)


if __name__ == "__main__":
    main()

