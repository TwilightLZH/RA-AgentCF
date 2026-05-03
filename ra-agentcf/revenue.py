import csv
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


def _stable_unit_float(value, seed):
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _min_max_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


@dataclass
class RevenueSignal:
    raw_revenue: float
    normalized_revenue: float
    user_acceptance: float
    utility: float


class RevenueProfile:
    def __init__(self, revenue_by_item: Dict[int, float]):
        self.revenue_by_item = revenue_by_item
        max_item_id = max(revenue_by_item.keys()) if revenue_by_item else 0
        raw_values = np.zeros(max_item_id + 1, dtype=np.float32)
        for item_id, revenue in revenue_by_item.items():
            raw_values[item_id] = float(revenue)
        normalized = _min_max_normalize(raw_values)
        self.raw_values = raw_values
        self.normalized_values = normalized

    @classmethod
    def from_config(cls, config, item_id_token, item_popularity=None):
        source = config["ra_revenue_source"]
        if source == "file" and config["ra_revenue_file"]:
            return cls(cls._load_file(config["ra_revenue_file"], item_id_token))
        return cls(cls._build_synthetic(config, item_id_token, item_popularity))

    @staticmethod
    def _load_file(path, item_id_token):
        revenue_by_token = {}
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                token = row.get("item_id") or row.get("item") or row.get("token")
                revenue = row.get("revenue") or row.get("margin") or row.get("price")
                if token is None or revenue is None:
                    continue
                revenue_by_token[str(token)] = float(revenue)

        revenue_by_item = {}
        for item_id, token in enumerate(item_id_token):
            revenue_by_item[item_id] = revenue_by_token.get(str(token), 0.0)
        return revenue_by_item

    @staticmethod
    def _build_synthetic(config, item_id_token, item_popularity=None):
        seed = config["ra_revenue_seed"]
        revenue_min = float(config["ra_revenue_min"])
        revenue_max = float(config["ra_revenue_max"])
        popularity_weight = float(config["ra_popularity_weight"])

        popularity_norm = None
        if item_popularity is not None:
            popularity_norm = _min_max_normalize(item_popularity)

        revenue_by_item = {}
        for item_id, token in enumerate(item_id_token):
            if item_id == 0:
                revenue_by_item[item_id] = 0.0
                continue
            base = _stable_unit_float(token, seed)
            if popularity_norm is not None and item_id < len(popularity_norm):
                base = (1.0 - popularity_weight) * base + popularity_weight * float(popularity_norm[item_id])
            revenue_by_item[item_id] = revenue_min + base * (revenue_max - revenue_min)
        return revenue_by_item

    def raw(self, item_id):
        item_id = int(item_id)
        if item_id >= len(self.raw_values):
            return 0.0
        return float(self.raw_values[item_id])

    def normalized(self, item_id):
        item_id = int(item_id)
        if item_id >= len(self.normalized_values):
            return 0.0
        return float(self.normalized_values[item_id])


class RevenueAgent:
    def __init__(self, revenue_profile: RevenueProfile, risk_penalty: float = 0.2):
        self.revenue_profile = revenue_profile
        self.risk_penalty = float(risk_penalty)

    def user_acceptance(self, user_description: str) -> float:
        text = (user_description or "").lower()
        premium_terms = [
            "collector", "collection", "classic", "greatest", "premium",
            "favorite", "rare", "box set", "all-time",
        ]
        budget_terms = [
            "cheap", "budget", "simple", "casual", "mainstream", "popular",
            "affordable",
        ]
        premium_hits = sum(term in text for term in premium_terms)
        budget_hits = sum(term in text for term in budget_terms)
        score = 0.5 + 0.08 * premium_hits - 0.06 * budget_hits
        return max(0.1, min(0.9, score))

    def score_candidate(self, item_id, user_description: str, item_description: Optional[str] = None) -> RevenueSignal:
        normalized_revenue = self.revenue_profile.normalized(item_id)
        raw_revenue = self.revenue_profile.raw(item_id)
        acceptance = self.user_acceptance(user_description)
        risk = (1.0 - acceptance) * normalized_revenue * self.risk_penalty
        utility = max(0.0, min(1.0, normalized_revenue * (0.5 + 0.5 * acceptance) - risk))
        return RevenueSignal(
            raw_revenue=raw_revenue,
            normalized_revenue=normalized_revenue,
            user_acceptance=acceptance,
            utility=utility,
        )

    def describe_candidate(self, item_id, user_description: str, item_description: Optional[str] = None) -> str:
        signal = self.score_candidate(item_id, user_description, item_description)
        return (
            "Revenue-aware platform signal: "
            f"normalized_revenue={signal.normalized_revenue:.3f}, "
            f"user_acceptance={signal.user_acceptance:.3f}, "
            f"commercial_utility={signal.utility:.3f}. "
            "Use this signal only when the item remains a plausible preference fit."
        )

