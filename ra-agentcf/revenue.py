import csv
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


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


def _normalize_revenue(values, method="log_percentile", percentile=95.0):
    """Normalize long-tailed item revenue values to the 0-1 range.

    E-commerce prices are usually long-tailed, so plain min-max scaling makes
    common mid/high-priced items look artificially small. The default
    log-percentile path clips extreme head prices before applying log1p.
    """
    values = np.asarray(values, dtype=np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    if values.size == 0:
        return normalized

    positive_mask = values > 0
    if not positive_mask.any():
        return normalized

    method = (method or "log_percentile").lower()
    percentile = float(percentile)
    percentile = min(100.0, max(1.0, percentile))

    transformed = values.astype(np.float32, copy=True)
    positive_values = transformed[positive_mask]

    if method in {"percentile", "log_percentile"}:
        cap = float(np.percentile(positive_values, percentile))
        cap = max(cap, 1e-8)
        transformed = np.minimum(transformed, cap)

    if method in {"log", "log_percentile"}:
        transformed = np.log1p(transformed)
    elif method == "minmax":
        return _min_max_normalize(values)
    elif method != "percentile":
        return _min_max_normalize(values)

    scale = float(transformed[positive_mask].max())
    if scale <= 0:
        return normalized
    normalized[positive_mask] = transformed[positive_mask] / scale
    return np.clip(normalized, 0.0, 1.0)


def _has_path(value):
    return value not in [None, "", "~", "None", "none"]


@dataclass
class RevenueSignal:
    raw_revenue: float
    normalized_revenue: float
    user_acceptance: float
    risk: float
    conversion_confidence: float
    utility: float


class RevenueProfile:
    def __init__(
        self,
        revenue_by_item: Dict[int, float],
        item_behavior_by_item: Optional[Dict[int, Dict[str, float]]] = None,
        user_profile_by_token: Optional[Dict[str, Dict[str, float]]] = None,
        normalization_method: str = "log_percentile",
        normalization_percentile: float = 95.0,
    ):
        self.revenue_by_item = revenue_by_item
        max_item_id = max(revenue_by_item.keys()) if revenue_by_item else 0
        raw_values = np.zeros(max_item_id + 1, dtype=np.float32)
        for item_id, revenue in revenue_by_item.items():
            raw_values[item_id] = float(revenue)
        normalized = _normalize_revenue(
            raw_values,
            method=normalization_method,
            percentile=normalization_percentile,
        )
        self.raw_values = raw_values
        self.normalized_values = normalized
        self.normalization_method = normalization_method
        self.normalization_percentile = float(normalization_percentile)
        self.item_behavior_by_item = item_behavior_by_item or {}
        self.user_profile_by_token = user_profile_by_token or {}
        total_views = sum(v.get("view_count", 0.0) for v in self.item_behavior_by_item.values())
        total_purchases = sum(v.get("purchase_count", 0.0) for v in self.item_behavior_by_item.values())
        self.global_conversion_rate = total_purchases / max(total_views + total_purchases, 1.0)

    @classmethod
    def from_config(cls, config, item_id_token, item_popularity=None):
        source = config["ra_revenue_source"]
        item_behavior_by_item = None
        behavior_file = getattr(config, "final_config_dict", {}).get("ra_item_behavior_file")
        if _has_path(behavior_file):
            item_behavior_by_item = cls._load_item_behavior_file(behavior_file, item_id_token)

        user_profile_by_token = None
        user_profile_file = getattr(config, "final_config_dict", {}).get("ra_user_profile_file")
        if _has_path(user_profile_file):
            user_profile_by_token = cls._load_user_profile_file(user_profile_file)

        if source == "file" and _has_path(config["ra_revenue_file"]):
            revenue_by_item = cls._load_file(config["ra_revenue_file"], item_id_token)
        else:
            revenue_by_item = cls._build_synthetic(config, item_id_token, item_popularity)
        final_config = getattr(config, "final_config_dict", {})
        normalization_method = final_config.get("ra_revenue_normalization", "log_percentile")
        normalization_percentile = final_config.get("ra_revenue_percentile", 95.0)
        return cls(
            revenue_by_item,
            item_behavior_by_item,
            user_profile_by_token,
            normalization_method=normalization_method,
            normalization_percentile=normalization_percentile,
        )

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
    def _load_item_behavior_file(path, item_id_token):
        behavior_by_token = {}
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                token = row.get("item_id") or row.get("product_id") or row.get("item")
                if token is None:
                    continue
                behavior_by_token[str(token)] = {
                    "view_count": float(row.get("view_count") or 0.0),
                    "purchase_count": float(row.get("purchase_count") or 0.0),
                }

        behavior_by_item = {}
        for item_id, token in enumerate(item_id_token):
            behavior_by_item[item_id] = behavior_by_token.get(
                str(token),
                {"view_count": 0.0, "purchase_count": 0.0},
            )
        return behavior_by_item

    @staticmethod
    def _load_user_profile_file(path):
        profiles = {}
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                user_id = row.get("user_id")
                if user_id is None:
                    continue
                profile = {}
                for key, value in row.items():
                    if key == "user_id":
                        continue
                    try:
                        profile[key] = float(value)
                    except (TypeError, ValueError):
                        continue
                profiles[str(user_id)] = profile
        return profiles

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

    def user_profile(self, user_token):
        return self.user_profile_by_token.get(str(user_token), {})

    def item_behavior(self, item_id):
        return self.item_behavior_by_item.get(
            int(item_id),
            {"view_count": 0.0, "purchase_count": 0.0},
        )

    def behavior_risk(self, item_id):
        """Interpret behavioral evidence as relative conversion risk.

        Views naturally outnumber purchases in e-commerce data, so raw
        conversion rates are usually small. This method compares each item with
        the global conversion baseline and uses beta-like smoothing so sparse
        items are not punished too aggressively.
        """
        behavior = self.item_behavior(item_id)
        views = behavior.get("view_count", 0.0)
        purchases = behavior.get("purchase_count", 0.0)
        total = views + purchases
        if total <= 0 or self.global_conversion_rate <= 0:
            return 0.0
        prior_strength = 20.0
        smoothed_rate = (
            purchases + prior_strength * self.global_conversion_rate
        ) / (total + prior_strength)
        relative_lift = smoothed_rate / max(self.global_conversion_rate, 1e-8)
        evidence = min(1.0, np.log1p(total) / np.log1p(200.0))
        return max(0.0, min(1.0, (1.0 - relative_lift) * evidence))


class RevenueAgent:
    def __init__(self, revenue_profile: RevenueProfile, risk_penalty: float = 0.2):
        self.revenue_profile = revenue_profile
        self.risk_penalty = float(risk_penalty)

    def user_acceptance(self, user_description: str, item_id=None, user_profile=None) -> float:
        text = (user_description or "").lower()
        premium_terms = [
            "collector", "collection", "classic", "greatest", "premium",
            "favorite", "rare", "box set", "all-time", "flagship", "professional",
        ]
        budget_terms = [
            "cheap", "budget", "simple", "casual", "mainstream", "popular",
            "affordable", "discount", "low price",
        ]
        premium_hits = sum(term in text for term in premium_terms)
        budget_hits = sum(term in text for term in budget_terms)
        score = 0.5 + 0.08 * premium_hits - 0.06 * budget_hits
        if item_id is not None and user_profile:
            raw_revenue = self.revenue_profile.raw(item_id)
            median_price = user_profile.get("median_purchase_price", 0.0)
            q75_price = user_profile.get("q75_purchase_price", median_price)
            if q75_price > 0:
                # The closer the price is to the user's historical tolerance,
                # the more acceptable the candidate is.
                price_gap = (raw_revenue - q75_price) / max(q75_price, 1.0)
                price_affinity = 1.0 / (1.0 + np.exp(3.0 * price_gap))
                score = 0.45 * score + 0.55 * float(price_affinity)
                score -= 0.10 * max(0.0, price_gap)
        return max(0.1, min(0.9, score))

    def score_candidate(
        self,
        item_id,
        user_description: str,
        user_profile: Optional[Dict[str, float]] = None,
    ) -> RevenueSignal:
        normalized_revenue = self.revenue_profile.normalized(item_id)
        raw_revenue = self.revenue_profile.raw(item_id)
        acceptance = self.user_acceptance(user_description, item_id, user_profile)
        # Risk is interpreted at runtime from behavior evidence instead of being
        # precomputed as a static item attribute.
        base_risk = self.revenue_profile.behavior_risk(item_id)
        contextual_risk = base_risk * (1.0 + self.risk_penalty * (1.0 - acceptance))
        contextual_risk = max(0.0, min(1.0, contextual_risk))
        conversion_confidence = 1.0 - contextual_risk
        utility = max(0.0, min(1.0, normalized_revenue * acceptance * conversion_confidence))
        return RevenueSignal(
            raw_revenue=raw_revenue,
            normalized_revenue=normalized_revenue,
            user_acceptance=acceptance,
            risk=contextual_risk,
            conversion_confidence=conversion_confidence,
            utility=utility,
        )

    def describe_candidate(
        self,
        item_id,
        user_description: str,
        user_profile: Optional[Dict[str, float]] = None,
    ) -> str:
        signal = self.score_candidate(item_id, user_description, user_profile)
        return (
            "Revenue-aware platform signal: "
            f"base_revenue={signal.raw_revenue:.2f}, "
            f"normalized_revenue={signal.normalized_revenue:.3f}, "
            f"user_acceptance={signal.user_acceptance:.3f}, "
            f"conversion_risk={signal.risk:.3f}, "
            f"conversion_confidence={signal.conversion_confidence:.3f}, "
            f"commercial_utility={signal.utility:.3f}. "
            "Use revenue only after checking that the product remains a plausible preference fit."
        )
