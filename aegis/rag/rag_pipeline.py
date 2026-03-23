"""
3-stage RAG: retrieve (FAISS) → augment → generate (HF/local or OpenAI).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from rag.generator import RiskExplanationGenerator
from rag.retriever import FaissNarrativeRetriever

logger = logging.getLogger("aegis")


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str,
        generator_model: str,
        top_k: int,
        max_context_chars: int,
        max_new_tokens: int,
        latency_budget_seconds: float,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.retriever = FaissNarrativeRetriever(embedding_model)
        self.generator = RiskExplanationGenerator(
            model_name=generator_model,
            max_new_tokens=max_new_tokens,
            openai_model=openai_model,
        )
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.latency_budget_seconds = latency_budget_seconds

    def build_corpus(self, narratives: List[Dict[str, Any]]) -> None:
        self.retriever.build_index(narratives)

    def explain(self, transaction_summary: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Returns (explanation, latency_seconds, metadata).
        Logs warning if latency exceeds budget (default 13s).
        """
        t0 = time.perf_counter()
        hits = self.retriever.search(transaction_summary, top_k=self.top_k)
        context = "\n---\n".join([h[0] for h in hits])[: self.max_context_chars]
        meta = {"retrieved": [h[0][:200] for h in hits], "scores": [h[1] for h in hits]}
        explanation = self.generator.generate(transaction_summary, context)
        elapsed = time.perf_counter() - t0
        logger.info("RAG latency: %.3fs (budget %.1fs)", elapsed, self.latency_budget_seconds)
        if elapsed > self.latency_budget_seconds:
            logger.warning(
                "RAG latency %.3fs exceeded budget %.1fs; try fewer max_new_tokens or a warm HF cache.",
                elapsed,
                self.latency_budget_seconds,
            )
        meta["latency_seconds"] = elapsed
        return explanation, elapsed, meta
