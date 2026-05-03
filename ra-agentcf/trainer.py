import importlib.util
import os
from collections import OrderedDict, defaultdict

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
AGENTCF_TRAINER = os.path.join(REPO_ROOT, "agentcf", "trainer.py")

spec = importlib.util.spec_from_file_location("agentcf_base_trainer", AGENTCF_TRAINER)
agentcf_base_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentcf_base_trainer)
LanguageLossTrainer = agentcf_base_trainer.LanguageLossTrainer


class RAAgentCFTrainer(LanguageLossTrainer):
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        self._ra_metric_sums = defaultdict(float)
        self._ra_metric_count = 0
        result = super().evaluate(
            eval_data,
            load_best_model=load_best_model,
            model_file=model_file,
            show_progress=show_progress,
        )
        if self._ra_metric_count > 0:
            result = OrderedDict(result)
            for key in sorted(self._ra_metric_sums.keys()):
                result[key] = round(self._ra_metric_sums[key] / self._ra_metric_count, 4)
        return result

    def _full_sort_batch_eval(self, batched_data, sampled_items):
        interaction, scores, positive_u, positive_i = super()._full_sort_batch_eval(
            batched_data,
            sampled_items,
        )
        self._collect_revenue_metrics(interaction, scores)
        return interaction, scores, positive_u, positive_i

    def _collect_revenue_metrics(self, interaction, scores):
        if not hasattr(self.model, "revenue_profile"):
            return

        topk = self.config["topk"]
        if isinstance(topk, int):
            topk = [topk]
        max_k = max(topk)
        _, top_items = torch.topk(scores, k=max_k, dim=1)

        for row_idx in range(top_items.shape[0]):
            user_id = int(interaction[self.model.USER_ID][row_idx])
            user_description = self.model.user_agents[user_id].memory_1[-1]
            for k in topk:
                item_ids = top_items[row_idx, :k].detach().cpu().tolist()
                revenues = [self.model.revenue_profile.normalized(item_id) for item_id in item_ids]
                utilities = [
                    self.model.revenue_agent.score_candidate(item_id, user_description).utility
                    for item_id in item_ids
                ]
                self._ra_metric_sums[f"revenue@{k}"] += sum(revenues) / len(revenues)
                self._ra_metric_sums[f"utility@{k}"] += sum(utilities) / len(utilities)
            self._ra_metric_count += 1

