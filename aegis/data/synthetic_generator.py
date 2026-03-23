"""
Synthetic transaction dataset aligned with transformer sequences, graph nodes, and RAG narratives.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataGenerator:
    """Generates tabular transactions plus textual risk narratives for RAG."""

    num_users: int = 120
    num_devices: int = 80
    num_merchants: int = 60
    locations: List[str] = field(
        default_factory=lambda: [
            "US-NY",
            "US-CA",
            "US-TX",
            "UK-LON",
            "DE-BER",
            "SG",
            "IN-MUM",
        ]
    )
    transactions_per_user_min: int = 8
    transactions_per_user_max: int = 24
    fraud_ratio_hint: float = 0.18
    seed: int = 42

    def __post_init__(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _make_ids(self) -> Tuple[List[str], List[str], List[str]]:
        users = [f"user_{i:04d}" for i in range(self.num_users)]
        devices = [f"dev_{i:04d}" for i in range(self.num_devices)]
        merchants = [f"merch_{i:04d}" for i in range(self.num_merchants)]
        return users, devices, merchants

    def _normal_amount(self) -> float:
        return float(np.clip(np.random.lognormal(mean=3.2, sigma=0.45), 1.0, 500.0))

    def _fraud_amount(self) -> float:
        return float(np.clip(np.random.lognormal(mean=5.5, sigma=0.6), 80.0, 15000.0))

    def generate_transactions(self) -> pd.DataFrame:
        users, devices, merchants = self._make_ids()
        rows: List[Dict[str, Any]] = []
        base_time = datetime(2024, 1, 1, 8, 0, 0)

        for u in users:
            n_tx = random.randint(self.transactions_per_user_min, self.transactions_per_user_max)
            # Stable "home" device and merchant for normal behavior
            home_dev = random.choice(devices)
            usual_merch = random.choice(merchants)
            home_loc = random.choice(self.locations[:4])

            for _ in range(n_tx):
                is_fraud = random.random() < self.fraud_ratio_hint
                if is_fraud:
                    amount = self._fraud_amount()
                    device = random.choice(devices)  # unseen / risky device mix
                    merchant = random.choice(merchants)
                    loc = random.choice(self.locations)
                    hour_jitter = random.randint(0, 23)
                    ts = base_time + timedelta(
                        days=random.randint(0, 90),
                        hours=hour_jitter,
                        minutes=random.randint(0, 59),
                    )
                    risk_label = 1
                else:
                    amount = self._normal_amount()
                    device = home_dev if random.random() < 0.85 else random.choice(devices)
                    merchant = usual_merch if random.random() < 0.75 else random.choice(merchants)
                    loc = home_loc if random.random() < 0.8 else random.choice(self.locations)
                    ts = base_time + timedelta(
                        days=random.randint(0, 90),
                        hours=random.randint(8, 21),
                        minutes=random.randint(0, 59),
                    )
                    risk_label = 0

                rows.append(
                    {
                        "transaction_id": str(uuid.uuid4()),
                        "user_id": u,
                        "merchant_id": merchant,
                        "device_id": device,
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "location": loc,
                        "risk_label": risk_label,
                    }
                )

        df = pd.DataFrame(rows)
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        return df

    def generate_risk_narratives(self, n: int = 48) -> List[Dict[str, Any]]:
        """Template narratives describing fraud patterns — used as RAG corpus."""
        templates = [
            (
                "Fraud case: unusually high amount combined with first-time merchant "
                "and device never seen for this customer in the prior 90 days."
            ),
            (
                "Risk narrative: transaction at odd hours with amount spike versus "
                "the user's rolling median."
            ),
            (
                "Past alert: velocity anomaly — multiple high-value purchases across "
                "new merchants within a short window."
            ),
            (
                "Investigation summary: device fingerprint mismatch with historical "
                "sessions; location hop inconsistent with travel velocity."
            ),
            (
                "Similar fraud: small baseline spend user suddenly charged a large "
                "amount at an electronics merchant using a new device."
            ),
            (
                "Behavioral signal: merchant category shift from groceries to luxury "
                "with immediate high-ticket purchase."
            ),
        ]
        out: List[Dict[str, Any]] = []
        for i in range(n):
            base = random.choice(templates)
            nid = f"nar_{i:04d}"
            label = 1 if random.random() < 0.55 else 0
            text = base + f" Case reference {nid}."
            out.append({"narrative_id": nid, "text": text, "risk_label": label})
        return out

    def save_artifacts(
        self,
        df: pd.DataFrame,
        narratives: List[Dict[str, Any]],
        csv_path: str,
        json_path: str,
    ) -> None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(narratives, f, indent=2)


def generate_sample_dataset(
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str, str]:
    """Convenience: build generator from config dict and save sample files."""
    cfg = cfg or {}
    dcfg = cfg.get("data", {})
    gen = SyntheticDataGenerator(
        num_users=int(dcfg.get("num_users", 120)),
        num_devices=int(dcfg.get("num_devices", 80)),
        num_merchants=int(dcfg.get("num_merchants", 60)),
        locations=list(dcfg.get("locations", SyntheticDataGenerator().locations)),
        transactions_per_user_min=int(dcfg.get("transactions_per_user_min", 8)),
        transactions_per_user_max=int(dcfg.get("transactions_per_user_max", 24)),
        fraud_ratio_hint=float(dcfg.get("fraud_ratio_hint", 0.18)),
        seed=int(cfg.get("seed", 42)),
    )
    df = gen.generate_transactions()
    narratives = gen.generate_risk_narratives(n=48)
    csv_path = dcfg.get("sample_csv", "outputs/sample_transactions.csv")
    json_path = dcfg.get("narratives_json", "outputs/risk_narratives.json")
    gen.save_artifacts(df, narratives, csv_path, json_path)
    return df, narratives, csv_path, json_path
