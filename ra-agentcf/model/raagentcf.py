import os
import sys
import importlib.util
import contextlib

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RA_ROOT = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(RA_ROOT)
AGENTCF_ROOT = os.path.join(REPO_ROOT, "agentcf")
if AGENTCF_ROOT not in sys.path:
    sys.path.insert(0, AGENTCF_ROOT)
if RA_ROOT not in sys.path:
    sys.path.insert(0, RA_ROOT)

agentcf_model_path = os.path.join(AGENTCF_ROOT, "model", "agentcf.py")
agentcf_spec = importlib.util.spec_from_file_location("agentcf_original_model", agentcf_model_path)
agentcf_original_model = importlib.util.module_from_spec(agentcf_spec)
agentcf_spec.loader.exec_module(agentcf_original_model)
AgentCF = agentcf_original_model.AgentCF

from revenue import RevenueAgent, RevenueProfile


class RAAgentCF(AgentCF):
    """Revenue-aware AgentCF.

    This class keeps AgentCF's user/item/recommender agents intact, then adds a
    revenue-aware agent around evaluation:

    - prompt-side: injects revenue utility context into candidate descriptions;
    - score-side: combines LLM preference scores with contextual revenue utility.
    """

    def __init__(self, config, dataset):
        self.debug = self._config_bool_from_config(config, "debug", False)
        with self._maybe_suppress_output(self.debug):
            super().__init__(config, dataset)
        if self.max_his_len is None:
            self.max_his_len = self._config_int("MAX_ITEM_LIST_LENGTH", 20)
        self.ra_enabled = self._config_bool("ra_enabled", True)
        self.ra_alpha = float(config["ra_alpha"])
        self.ra_inject_revenue_context = self._config_bool("ra_inject_revenue_context", True)
        item_popularity = self._item_popularity(dataset)
        self.revenue_profile = RevenueProfile.from_config(config, self.item_id_token, item_popularity)
        self.revenue_agent = RevenueAgent(
            self.revenue_profile,
            risk_penalty=float(config["ra_revenue_risk_penalty"]),
        )
        self.last_ra_batch_report = []

    @staticmethod
    @contextlib.contextmanager
    def _maybe_suppress_output(debug):
        if debug:
            yield
            return
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield

    @staticmethod
    def _config_bool_from_config(config, key, default=False):
        value = getattr(config, "final_config_dict", {}).get(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() in {"1", "true", "yes", "y"}

    def _config_bool(self, key, default=False):
        return self._config_bool_from_config(self.config, key, default)

    def _config_int(self, key, default):
        value = getattr(self.config, "final_config_dict", {}).get(key, default)
        if value in [None, "", "~"]:
            return int(default)
        return int(value)

    def _item_popularity(self, dataset):
        popularity = torch.zeros(self.n_items, dtype=torch.float32)
        try:
            item_ids = dataset.inter_feat[self.ITEM_ID].numpy()
            for item_id in item_ids:
                popularity[int(item_id)] += 1
        except Exception:
            return None
        return popularity.numpy()

    def get_batch_inputs(self, interaction, idxs, i, user_embedding):
        (
            user_his_text,
            candidate_text,
            candidate_text_order,
            candidate_idx,
            candidate_text_order_description,
        ) = super().get_batch_inputs(interaction, idxs, i, user_embedding)

        if self.ra_enabled and self.ra_inject_revenue_context:
            user_id = int(interaction[self.USER_ID][i])
            user_description = self.user_agents[user_id].memory_1[-1]
            enriched_descriptions = []
            for j, item in enumerate(idxs[i]):
                item_id = int(item)
                item_description = list(self.item_agents[item_id].memory_embedding.keys())[-1]
                revenue_note = self.revenue_agent.describe_candidate(
                    item_id,
                    user_description,
                    item_description,
                )
                enriched_descriptions.append(
                    f"{candidate_text_order_description[j]}\n{revenue_note}"
                )
            candidate_text_order_description = enriched_descriptions

        return (
            user_his_text,
            candidate_text,
            candidate_text_order,
            candidate_idx,
            candidate_text_order_description,
        )

    def full_sort_predict(self, interaction, idxs):
        scores = super().full_sort_predict(interaction, idxs)
        if not self.ra_enabled or self.ra_alpha <= 0:
            self.last_ra_batch_report = []
            return scores
        return self._apply_revenue_aware_rerank(scores, interaction, idxs)

    def _apply_revenue_aware_rerank(self, scores, interaction, idxs):
        batch_user = interaction[self.USER_ID]
        self.last_ra_batch_report = []

        for row_idx in range(idxs.shape[0]):
            user_id = int(batch_user[row_idx])
            user_description = self.user_agents[user_id].memory_1[-1]
            item_ids = [int(item) for item in idxs[row_idx].detach().cpu().tolist()]
            preference_scores = torch.tensor(
                [float(scores[row_idx, item_id]) for item_id in item_ids],
                device=scores.device,
                dtype=torch.float32,
            )
            preference_norm = self._normalize_preference_scores(preference_scores)

            utility_values = []
            raw_revenues = []
            for item_id in item_ids:
                item_description = list(self.item_agents[item_id].memory_embedding.keys())[-1]
                signal = self.revenue_agent.score_candidate(
                    item_id,
                    user_description,
                    item_description,
                )
                utility_values.append(signal.utility)
                raw_revenues.append(signal.raw_revenue)

            utility_tensor = torch.tensor(utility_values, device=scores.device, dtype=torch.float32)
            combined = (1.0 - self.ra_alpha) * preference_norm + self.ra_alpha * utility_tensor

            for local_idx, item_id in enumerate(item_ids):
                scores[row_idx, item_id] = combined[local_idx] * float(self.config["recall_budget"])

            self.last_ra_batch_report.append(
                {
                    "user_id": user_id,
                    "candidate_item_ids": item_ids,
                    "preference_scores": [float(v) for v in preference_scores.detach().cpu().tolist()],
                    "revenue_values": raw_revenues,
                    "utility_values": utility_values,
                    "combined_scores": [float(v) for v in combined.detach().cpu().tolist()],
                }
            )

        return scores

    def _normalize_preference_scores(self, preference_scores):
        valid_mask = preference_scores > -5000
        if not bool(valid_mask.any()):
            return torch.zeros_like(preference_scores)
        valid_values = preference_scores[valid_mask]
        min_value = valid_values.min()
        max_value = valid_values.max()
        normalized = torch.zeros_like(preference_scores)
        if float(max_value - min_value) <= 1e-8:
            normalized[valid_mask] = 1.0
        else:
            normalized[valid_mask] = (valid_values - min_value) / (max_value - min_value)
        return normalized
