"""
Lightweight explanation generator — HuggingFace causal LM or OpenAI when configured.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("aegis")


class RiskExplanationGenerator:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 128,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.openai_model = openai_model
        self._pipe = None

    def _hf_pipeline(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
            )
        return self._pipe

    def generate(self, query: str, context: str, use_openai: bool = True) -> str:
        if use_openai and os.environ.get("OPENAI_API_KEY"):
            return self._openai_generate(query, context)
        return self._hf_generate(query, context)

    def _openai_generate(self, query: str, context: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed; falling back to HF generator.")
            return self._hf_generate(query, context)

        client = OpenAI()
        prompt = (
            "You are a fraud analyst. Given retrieved similar cases and a transaction summary, "
            "explain briefly why the transaction may be risky.\n\n"
            f"Retrieved cases:\n{context}\n\nTransaction summary:\n{query}\n\n"
            "Answer in 2-3 sentences."
        )
        resp = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(self.max_new_tokens, 300),
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    def _hf_generate(self, query: str, context: str) -> str:
        pipe = self._hf_pipeline()
        prompt = (
            "You are a fraud analyst. Given retrieved similar cases and a transaction summary, "
            "explain briefly why the transaction may be risky.\n\n"
            f"Retrieved cases:\n{context}\n\nTransaction summary:\n{query}\n\n"
            "Explanation:"
        )
        tok = pipe.tokenizer
        out = pipe(
            prompt,
            max_new_tokens=min(self.max_new_tokens, 96),
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        text = out[0]["generated_text"]
        if "Explanation:" in text:
            return text.split("Explanation:")[-1].strip()
        return text[len(prompt) :].strip()
