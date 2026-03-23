"""
FAISS-backed dense retriever over historical risk narratives.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger("aegis")


class FaissNarrativeRetriever:
    """Embed narrative corpus with Sentence-Transformers; search with inner-product (cosine after L2 norm)."""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model_name
        self._model = None
        self._index: Optional[faiss.Index] = None
        self._texts: List[str] = []
        self._meta: List[Dict[str, Any]] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def build_index(self, narratives: List[Dict[str, Any]], text_key: str = "text") -> None:
        self._texts = [str(n[text_key]) for n in narratives]
        self._meta = [dict(n) for n in narratives]
        model = self._get_model()
        emb = model.encode(self._texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(emb)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self._index = index
        logger.info("FAISS index built with %d vectors of dim %d", emb.shape[0], dim)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Index not built — call build_index first.")
        model = self._get_model()
        q = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, min(top_k, len(self._texts)))
        out: List[Tuple[str, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            out.append((self._texts[i], float(s)))
        return out
