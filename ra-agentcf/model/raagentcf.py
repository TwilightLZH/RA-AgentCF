import os
import sys
import importlib.util
import contextlib
import csv
import re

import torch
from agentverse.parser import OutputParser, OutputParserError

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


class RARecommenderParser(OutputParser):
    """Parse recommender choices with tolerant e-commerce wording."""

    def parse(self, text):
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)

        choice_match = re.search(
            r"(?:Choice|Selected product|Selected item|Recommendation)\s*:\s*(.*)",
            cleaned_output,
            flags=re.IGNORECASE,
        )
        if choice_match:
            choice_start = choice_match.end(1) - len(choice_match.group(1))
            tail = cleaned_output[choice_start:].strip()
        else:
            tail = cleaned_output

        explanation_match = re.search(
            r"\b(?:Explanation|Reason|Rationale|Why)\s*:",
            tail,
            flags=re.IGNORECASE,
        )
        if explanation_match:
            choice = tail[:explanation_match.start()].strip()
            reason = tail[explanation_match.end():].strip()
        else:
            lines = [line.strip() for line in tail.split("\n") if line.strip()]
            if not lines:
                raise OutputParserError(text)
            choice = lines[0]
            reason = " ".join(lines[1:]).strip() or cleaned_output

        choice = re.sub(r"^[\-\*\d\.\)\s]+", "", choice).strip().strip("'\"")
        if not choice:
            raise OutputParserError(text)
        return choice, reason

    def parse_evaluation(self, text):
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        match = re.search(r"Rank\s*:\s*", cleaned_output, flags=re.IGNORECASE)
        if match:
            cleaned_output = cleaned_output[match.end():].strip()
        return [line.strip() for line in cleaned_output.split("\n") if line.strip()]

    def parse_backward(self, text):
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        match = re.search(r"Updated Strategy\s*:\s*", cleaned_output, flags=re.IGNORECASE)
        if match:
            return cleaned_output[match.end():].strip()
        return cleaned_output

    def parse_summary(self, text):
        cleaned_output = text.strip()
        return re.sub(r"\n+", "\n", cleaned_output).strip()


class RAItemAgentParser(OutputParser):
    """Parse item-memory updates for e-commerce prompts.

    The original AgentCF parser only accepts "first/second CD". RA-AgentCF uses
    product wording, so we keep this parser local to RA-AgentCF and accept both
    formats for compatibility.
    """

    def parse(self, text):
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        pattern = re.compile(
            r"The updated description of the first\s+(?:CD|product|item)\s*(?:is)?\s*:\s*"
            r"(.*?)\s*"
            r"The updated description of the second\s+(?:CD|product|item)\s*(?:is)?\s*:\s*"
            r"(.*)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(cleaned_output)
        if not match:
            raise OutputParserError(text)

        first_description = match.group(1).strip()
        second_description = match.group(2).strip()
        if not first_description or not second_description:
            raise OutputParserError(text)
        return first_description, second_description


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
        self._install_ra_parsers()
        if self.max_his_len is None:
            self.max_his_len = self._config_int("MAX_ITEM_LIST_LENGTH", 20)
        self.ra_enabled = self._config_bool("ra_enabled", True)
        self.ra_alpha = float(config["ra_alpha"])
        self.ra_fusion_mode = getattr(config, "final_config_dict", {}).get("ra_fusion_mode", "multiplicative")
        self.ra_inject_revenue_context = self._config_bool("ra_inject_revenue_context", True)
        item_popularity = self._item_popularity(dataset)
        self.revenue_profile = RevenueProfile.from_config(config, self.item_id_token, item_popularity)
        self.revenue_agent = RevenueAgent(
            self.revenue_profile,
            risk_penalty=float(config["ra_revenue_risk_penalty"]),
        )
        self.last_ra_batch_report = []

    def _install_ra_parsers(self):
        self.rec_agent.output_parser = RARecommenderParser()
        parser = RAItemAgentParser()
        for item_agent in self.item_agents.values():
            item_agent.output_parser = parser

    def logging_during_updation(
        self,
        batch_user,
        system_explanations,
        user_backward_prompts,
        pos_item_descriptions_forward,
        neg_item_descriptions_forward,
        user_update_descriptions,
        item_update_memories,
    ):
        for i in range(len(batch_user)):
            user_id = int(batch_user[i])
            path = os.path.join(
                self.config["record_path"],
                self.dataset_name,
                "record",
                f"user_record_{self.record_idx}",
            )
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"user.{user_id}"), "a", encoding="utf-8") as file:
                file.write("~" * 20 + "Updation during reflection" + "~" * 20 + "\n")
                file.write(
                    "There are two candidate products.\n"
                    f"The purchased product has the following information: {pos_item_descriptions_forward[i]}.\n"
                    f"The non-purchased candidate has the following information: {neg_item_descriptions_forward[i]}\n\n"
                )
                file.write(
                    "The recommender system made an unsuitable recommendation.\n"
                    f"Its reasons are as follows: {system_explanations[i]}\n\n"
                )
                file.write(
                    f"The user's previous self-description is as follows: "
                    f"{self.user_agents[user_id].memory_1[-1]}\n\n"
                )
                file.write(
                    f"The prompts to update the user's descriptions are as follows: "
                    f"{user_backward_prompts[i]}\n\n"
                )
                file.write(
                    f"The user updates their self-description as follows: "
                    f"{user_update_descriptions[i]}\n\n"
                )
                if self.config["update_neg_item"]:
                    file.write(
                        "The two candidate products update their descriptions.\n"
                        f"The purchased product has the following updated information: {item_update_memories[i][1]}\n"
                        f"The non-purchased candidate has the following updated information: {item_update_memories[i][0]}\n\n"
                    )
                else:
                    file.write(
                        f"The purchased product has the following updated information: "
                        f"{item_update_memories[i][1]}\n\n"
                    )

    def logging_after_updation(self, batch_user, batch_pos_item, batch_neg_item):
        print("~" * 20 + f"logging in record_{self.record_idx}" + "~" * 20)
        batch_size = batch_user.size(0)
        user_path = os.path.join(
            self.config["record_path"],
            self.dataset_name,
            "record",
            f"user_record_{self.record_idx}",
        )
        item_path = os.path.join(
            self.config["record_path"],
            self.dataset_name,
            "record",
            f"item_record_{self.record_idx}",
        )
        os.makedirs(user_path, exist_ok=True)
        os.makedirs(item_path, exist_ok=True)

        for i, user in enumerate(batch_user):
            user_id = int(user)
            pos_item_id = int(batch_pos_item[i])
            neg_item_id = int(batch_neg_item[i])
            with open(os.path.join(user_path, f"user.{user_id}"), "a", encoding="utf-8") as file:
                file.write("~" * 20 + "New interaction" + "~" * 20 + "\n")
                file.write(
                    "There are two candidate products.\n"
                    f"The purchased product has the following information: "
                    f"{list(self.item_agents[pos_item_id].memory_embedding.keys())[-1]}.\n"
                    f"The non-purchased candidate has the following information: "
                    f"{list(self.item_agents[neg_item_id].memory_embedding.keys())[-1]}\n\n"
                )
                file.write(
                    f"The user's previous self-description is as follows: "
                    f"{self.user_agents[user_id].memory_1[-1]}\n\n"
                )
                file.write(
                    f"The user updates their self-description as follows: "
                    f"{self.user_agents[user_id].update_memory[-1]}\n\n"
                )
                if self.config["update_neg_item"]:
                    file.write(
                        "The two candidate products update their descriptions.\n"
                        f"The purchased product has the following updated information: "
                        f"{self.item_agents[pos_item_id].update_memory[-1]}\n"
                        f"The non-purchased candidate has the following updated information: "
                        f"{self.item_agents[neg_item_id].update_memory[-1]}\n\n"
                    )
                else:
                    file.write(
                        f"The purchased product has the following updated information: "
                        f"{self.item_agents[pos_item_id].update_memory[-1]}\n\n"
                    )

        for i in range(batch_size):
            pos_item_id = int(batch_pos_item[i])
            neg_item_id = int(batch_neg_item[i])
            user_id = int(batch_user[i])
            with open(os.path.join(item_path, f"item.{pos_item_id}"), "a", encoding="utf-8") as file:
                file.write("~" * 20 + "New interaction" + "~" * 20 + "\n")
                file.write(
                    f"This product: {self.item_agents[pos_item_id].role_description['item_title']} "
                    f"and the other candidate: {self.item_agents[neg_item_id].role_description['item_title']} "
                    "were recommended to a user.\n\n"
                )
                file.write(
                    f"This product has the following description: "
                    f"{list(self.item_agents[pos_item_id].memory_embedding.keys())[-1]}\n\n"
                )
                file.write(
                    f"The other candidate has the following description: "
                    f"{list(self.item_agents[neg_item_id].memory_embedding.keys())[-1]}\n\n"
                )
                file.write(
                    f"The user's previous self-description is as follows: "
                    f"{self.user_agents[user_id].memory_1[-1]}\n\n"
                )
                file.write(
                    f"The user updates their self-description as follows: "
                    f"{self.user_agents[user_id].update_memory[-1]}\n\n"
                )
                file.write(
                    f"This product updates its description as follows: "
                    f"{self.item_agents[pos_item_id].update_memory[-1]}\n\n"
                )
                if self.config["update_neg_item"]:
                    file.write(
                        f"The other candidate updates its description as follows: "
                        f"{self.item_agents[neg_item_id].update_memory[-1]}\n\n"
                    )

            if self.config["update_neg_item"]:
                with open(os.path.join(item_path, f"item.{neg_item_id}"), "a", encoding="utf-8") as file:
                    file.write("~" * 20 + "New interaction" + "~" * 20 + "\n")
                    file.write(
                        f"This product: {self.item_agents[neg_item_id].role_description['item_title']} "
                        f"and the purchased product: {self.item_agents[pos_item_id].role_description['item_title']} "
                        "were recommended to a user.\n\n"
                    )
                    file.write(
                        f"The purchased product has the following description: "
                        f"{list(self.item_agents[pos_item_id].memory_embedding.keys())[-1]}\n\n"
                    )
                    file.write(
                        f"The user's previous self-description is as follows: "
                        f"{self.user_agents[user_id].memory_1[-1]}\n\n"
                    )
                    file.write(
                        f"The user updates their self-description as follows: "
                        f"{self.user_agents[user_id].update_memory[-1]}\n\n"
                    )
                    file.write(
                        f"This product updates its description as follows: "
                        f"{self.item_agents[neg_item_id].update_memory[-1]}\n\n"
                    )

    def _dataset_item_file(self):
        dataset_item_file = os.path.join(self.data_path, f"{self.dataset_name}.item")
        if os.path.exists(dataset_item_file):
            return dataset_item_file
        return None

    def _read_dataset_items(self):
        item_file = self._dataset_item_file()
        if item_file is None:
            return None
        rows = {}
        with open(item_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                token = row.get("item_id:token") or row.get("item_id")
                if token is None:
                    continue
                rows[str(token)] = row
        return rows

    def _read_user_profiles(self):
        profile_file = getattr(self.config, "final_config_dict", {}).get("ra_user_profile_file")
        if profile_file in [None, "", "~", "None", "none"] or not os.path.exists(profile_file):
            return None
        rows = {}
        with open(profile_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                user_id = row.get("user_id")
                if user_id is not None:
                    rows[str(user_id)] = row
        return rows

    def load_text(self):
        rows = self._read_dataset_items()
        if rows is None:
            return super().load_text()

        item_text = ["[PAD]"]
        for token in self.item_id_token:
            if token == "[PAD]":
                continue
            row = rows.get(str(token), {})
            title = row.get("title:token_seq") or row.get("title") or f"Product {token}"
            item_text.append(title)
        return item_text

    def load_user_context(self):
        rows = self._read_user_profiles()
        if rows is None:
            return super().load_user_context()

        user_context = {
            0: {
                "agent_type": "useragent",
                "role_description": {},
                "memory_1": ["[PAD]"],
                "update_memory": ["[PAD]"],
                "role_description_string_1": "[PAD]",
                "role_description_string_3": "[PAD]",
                "role_task": "[PAD]",
                "prompt_template": self.config["user_prompt_template"],
                "user_prompt_system_role": self.config["user_prompt_system_role"],
                "llm": self._build_text_llm_config(self.config["llm_temperature"]),
                "llm_chat": self._build_chat_llm_config(self.config["llm_temperature"]),
                "agent_mode": "user",
                "output_parser_type": "useragent",
                "historical_interactions": [],
                "user_prompt_template_true": self.config["user_prompt_template_true"],
            }
        }
        for token in self.user_id_token:
            if token == "[PAD]" or token not in self.user_token_id:
                continue
            row = rows.get(str(token), {})
            user_id = self.user_token_id[token]
            median_price = row.get("median_purchase_price", "unknown")
            q75_price = row.get("q75_purchase_price", "unknown")
            mean_price = row.get("mean_purchase_price", "unknown")
            view_count = row.get("view_count", "0")
            purchase_count = row.get("purchase_count", "0")
            # 这里只写入可观察事实，不把浏览/购买比例直接命名为价格敏感度。
            # 后续风险与转化信心由 RevenueAgent 根据这些事实和候选商品动态解释。
            self_description = (
                "I am an online shopping user. "
                f"My historical purchase price median is {median_price}, "
                f"the upper-quartile purchase price is {q75_price}, "
                f"and the mean purchase price is {mean_price}. "
                f"My behavior history contains {view_count} views and {purchase_count} purchases."
            )
            user_context[user_id] = {
                "agent_type": "useragent",
                "role_description": {},
                "role_description_string_3": self_description,
                "role_description_string_1": self_description,
                "user_prompt_system_role": self.config["user_prompt_system_role"],
                "memory_1": [self_description],
                "update_memory": [self_description],
                "prompt_template": self.config["user_prompt_template"],
                "llm": self._build_text_llm_config(self.config["llm_temperature"]),
                "llm_chat": self._build_chat_llm_config(self.config["llm_temperature"]),
                "agent_mode": "user",
                "output_parser_type": "useragent",
                "historical_interactions": [],
                "user_prompt_template_true": self.config["user_prompt_template_true"],
            }
        return user_context

    def load_item_context(self):
        rows = self._read_dataset_items()
        if rows is None:
            return super().load_item_context()

        item_context = {
            0: {
                "agent_type": "itemagent",
                "role_description": {
                    "item_title": "[PAD]",
                    "item_release_year": "[PAD]",
                    "item_class": "[PAD]",
                },
                "memory": ["[PAD]"],
                "memory_embedding": {},
                "update_memory": ["[PAD]"],
                "item_prompt_template_true": self.config["item_prompt_template_true"],
                "role_description_string": "[PAD]",
                "role_task": "[PAD]",
                "prompt_template": self.config["user_prompt_template"],
                "llm": self._build_text_llm_config(self.config["llm_temperature"]),
                "llm_chat": self._build_chat_llm_config(self.config["llm_temperature"]),
                "agent_mode": "user",
                "output_parser_type": "itemagent",
            }
        }
        init_item_descriptions = []
        for token in self.item_id_token:
            if token == "[PAD]" or token not in self.item_token_id:
                continue
            row = rows.get(str(token))
            if row is None:
                continue
            item_id = self.item_token_id[token]
            title = row.get("title:token_seq") or row.get("title") or f"Product {token}"
            category = row.get("category:token_seq") or row.get("category") or "unknown category"
            brand = row.get("brand:token") or row.get("brand") or "unknown brand"
            price = row.get("price:float") or row.get("price") or "unknown"
            view_count = row.get("view_count:float") or row.get("view_count") or "0"
            purchase_count = row.get("purchase_count:float") or row.get("purchase_count") or "0"
            # 这里把商品侧可观察事实写入 ItemAgent 的初始记忆。
            # 不在预处理阶段判断 risk，让 agent 在交互和排序时解释这些行为证据。
            role_description_string = (
                f"The product is '{title}'. Brand: {brand}. Category: {category}. "
                f"Estimated revenue/price: {price}. "
                f"Historical behavior evidence: {view_count} views and {purchase_count} purchases."
            )
            item_context[item_id] = {
                "agent_type": "itemagent",
                "update_memory": [role_description_string],
                "role_description": {"item_title": title, "item_class": category},
                "role_description_string": role_description_string,
                "prompt_template": self.config["item_prompt_template"],
                "item_prompt_template_true": self.config["item_prompt_template_true"],
                "llm": self._build_text_llm_config(self.config["llm_temperature"]),
                "llm_chat": self._build_chat_llm_config(self.config["llm_temperature"]),
                "agent_mode": "item",
                "output_parser_type": "itemagent",
            }
            init_item_descriptions.append(role_description_string)

        if self.config["evaluation"] == "rag":
            init_item_description_embeddings = self.generate_embedding(init_item_descriptions)
            for i, item in enumerate(item_context.keys()):
                if item == 0:
                    continue
                item_context[item]["memory_embedding"] = {
                    init_item_descriptions[i - 1]: init_item_description_embeddings[i - 1]
                }
        else:
            for i, item in enumerate(item_context.keys()):
                if item == 0:
                    continue
                item_context[item]["memory_embedding"] = {init_item_descriptions[i - 1]: None}

        return item_context

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
            user_token = self.user_id_token[user_id]
            user_description = self.user_agents[user_id].memory_1[-1]
            user_profile = self.revenue_profile.user_profile(user_token)
            enriched_descriptions = []
            for j, item in enumerate(idxs[i]):
                item_id = int(item)
                item_description = list(self.item_agents[item_id].memory_embedding.keys())[-1]
                revenue_note = self.revenue_agent.describe_candidate(
                    item_id,
                    user_description,
                    item_description,
                    user_profile,
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
            user_token = self.user_id_token[user_id]
            user_description = self.user_agents[user_id].memory_1[-1]
            user_profile = self.revenue_profile.user_profile(user_token)
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
                    user_profile,
                )
                utility_values.append(signal.utility)
                raw_revenues.append(signal.raw_revenue)

            utility_tensor = torch.tensor(utility_values, device=scores.device, dtype=torch.float32)
            if self.ra_fusion_mode == "linear":
                combined = (1.0 - self.ra_alpha) * preference_norm + self.ra_alpha * utility_tensor
            else:
                # 乘性融合避免“纯高价但不相关”的商品被收益项单独推上去。
                commercial_factor = (1.0 - self.ra_alpha) + self.ra_alpha * (0.5 + utility_tensor)
                combined = preference_norm * commercial_factor

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
