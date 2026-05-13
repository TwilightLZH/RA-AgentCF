from model.raagentcf import RAAgentCF


class AgentCF(RAAgentCF):
    """AgentCF baseline adapted to the e-commerce data format.

    It keeps the e-commerce item/user loading and robust parsers from
    RAAgentCF, but disables revenue-aware prompt injection and score fusion.
    Revenue metrics can still be computed after ranking for fair comparison.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.ra_enabled = False
        self.ra_alpha = 0.0
        self.ra_inject_revenue_context = False

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
            user_id = self.user_token_id[token]
            self_description = (
                "I am an online shopping user. My preferences should be inferred from "
                "the product categories, brands, functions, styles, and use cases that "
                "I choose or reject during interactions."
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
            readable_category = self._readable_category(category)
            role_description_string = (
                f"The product is '{title}'. Brand: {brand}. "
                f"Category: {readable_category}."
            )
            item_context[item_id] = {
                "agent_type": "itemagent",
                "update_memory": [role_description_string],
                "role_description": {
                    "item_title": title,
                    "item_class": readable_category,
                },
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
    def _readable_category(category):
        words = str(category or "unknown category").replace(".", " ").replace("_", " ").split()
        if not words:
            return "unknown category"
        return " ".join(words[-2:] if len(words) >= 2 else words)
