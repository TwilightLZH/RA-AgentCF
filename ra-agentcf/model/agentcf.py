from model.raagentcf import RAAgentCF


class AgentCF(RAAgentCF):
    """AgentCF baseline adapted to the e-commerce data format.

    It keeps the e-commerce item/user loading and robust parsers from
    RAAgentCF, but disables revenue-aware prompt injection and score fusion.
    Revenue metrics can still be computed after ranking for fair comparison.
    """

    def __init__(self, config, dataset):
        self._install_preference_only_prompts(config)
        super().__init__(config, dataset)
        self.ra_enabled = False
        self.ra_alpha = 0.0
        self.ra_inject_revenue_context = False

    @staticmethod
    def _install_preference_only_prompts(config):
        """Keep the baseline focused on preference, not price or platform revenue."""
        prompts = {
            "user_prompt_system_role": (
                "You are an online shopping user.\n"
                "Here is your previous self-introduction, describing only your product "
                "interests, preferred categories, brands and dislikes:\n"
                "'$user_description'."
            ),
            "system_prompt_template": (
                "You are an online shopping user. Here is your self-introduction, expressing "
                "only your product preferences and dislikes: '$user_description'.\n\n"
                "Now you are choosing one product from two candidate products. Their features "
                "are listed below:\n$list_of_item_description.\n\n"
                "Please select the product that best matches your interests, category needs, "
                "brand preferences, and functional preferences. Ignore price, revenue, platform "
                "profit, purchase probability, and conversion statistics.\n\n"
                "Output exactly two lines and no extra text:\n"
                "Choice: <exact selected product title>\n"
                "Explanation: <one concise preference-based reason for choosing it and rejecting the other product>"
            ),
            "system_prompt_template_evaluation_basic": (
                "I am an online shopping user. Here is my self-introduction, including only my "
                "product preferences and dislikes:\n\n'$user_description'.\n\n"
                "Now I am looking for products from $candidate_num candidates. The product "
                "features are listed below:\n$example_list_of_item_description.\n\n"
                "Please rank these products only by preference fit: category interest, brand "
                "interest, product function, and disliked attributes. Do not use price, revenue, "
                "platform profit, purchase probability, or conversion evidence. Copy product "
                "titles exactly from the candidate list.\n\n"
                "Output only this format:\nRank:\n1. <exact product title>\n2. <exact product title>\n..."
            ),
            "system_prompt_template_evaluation_sequential": (
                "I am an online shopping user. Here is my self-introduction: '$user_description'. "
                "My recent browsing and purchasing history is:\n$historical_interactions.\n\n"
                "Now I am looking for products from $candidate_num candidates. The product "
                "features are listed below:\n$example_list_of_item_description.\n\n"
                "Please rank these products only by my historical interests and preference fit. "
                "Do not use price, revenue, platform profit, purchase probability, or conversion "
                "evidence. Copy product titles exactly from the candidate list.\n\n"
                "Output only this format:\nRank:\n1. <exact product title>\n2. <exact product title>\n..."
            ),
            "system_prompt_template_evaluation_retrieval": (
                "I am an online shopping user. My previous self-introduction is: "
                "'$user_past_description'. My updated self-introduction is: '$user_description'.\n\n"
                "Now I am looking for products from $candidate_num candidates. The product "
                "features are listed below:\n$example_list_of_item_description.\n\n"
                "Please rank these products only by current and past preference fit. Do not use "
                "price, revenue, platform profit, purchase probability, or conversion evidence. "
                "Copy product titles exactly from the candidate list.\n\n"
                "Output only this format:\nRank:\n1. <exact product title>\n2. <exact product title>\n..."
            ),
            "system_prompt_template_backward": (
                "You are an online shopping recommender system. In the past, you recommended "
                "one product from two candidates to a user according to a personalized "
                "preference strategy. The user's self-introduction is: '$user_description'.\n"
                "The first product is described as follows: '$item_description_1'.\n"
                "The second product is described as follows: '$item_description_2'.\n\n"
                "The user eventually selected $pos_movie and gave this reason: "
                "'$user_reasons'. Please update the recommendation strategy using only "
                "preference-related evidence such as category, brand, function, style, and "
                "use case.\n\n"
                "Output format: 'Updated Strategy: {The updated preference-only recommendation strategy}'."
            ),
            "user_prompt_template": (
                "Recently, you compared two candidate products:\n$list_of_item_description.\n\n"
                "You initially selected '$neg_item_title' and rejected the other product, with "
                "this explanation:\n'$system_reason'.\n\n"
                "However, the actual preference feedback shows that '$pos_item_title' better "
                "matched your interests while '$neg_item_title' did not. Please update your "
                "self-introduction using only product-preference evidence: categories, brands, "
                "functions, styles, use cases, and dislikes. Do not mention price, budget, "
                "revenue, purchase probability, conversion, or platform profit.\n\n"
                "Output format: 'My updated self-introduction: [updated self-introduction]'. "
                "Keep it under 180 words."
            ),
            "user_prompt_template_true": (
                "Recently, you compared two candidate products:\n$list_of_item_description.\n\n"
                "You selected '$pos_item_title' and rejected '$neg_item_title', with this "
                "explanation:\n'$system_reason'.\n\n"
                "The actual preference feedback confirms this choice. Please update your "
                "self-introduction using only product-preference evidence: categories, brands, "
                "functions, styles, use cases, and dislikes. Do not mention price, budget, "
                "revenue, purchase probability, conversion, or platform profit.\n\n"
                "Output format: 'My updated self-introduction: [updated self-introduction]'. "
                "Keep it under 180 words."
            ),
            "item_prompt_template": (
                "Here is the self-introduction of a user: '$user_description'.\n"
                "The user compared two products:\n$list_of_item_description.\n\n"
                "The system recommended '$neg_item_title', but actual feedback shows "
                "'$pos_item_title' was the better preference match. Please update the two "
                "product descriptions using only preference-related evidence: product category, "
                "brand, function, style, and use case. Do not mention price, revenue, purchase "
                "probability, conversion, popularity counts, or platform profit.\n\n"
                "Output format: 'The updated description of the first product is: [description]. "
                "\n The updated description of the second product is: [description].'. "
                "Each description must be under 50 words."
            ),
            "item_prompt_template_true": (
                "Here is the self-introduction of a user: '$user_description'.\n"
                "The user compared two products:\n$list_of_item_description.\n\n"
                "The user selected '$pos_item_title' and rejected '$neg_item_title'. Please "
                "update the two product descriptions using only preference-related evidence: "
                "product category, brand, function, style, and use case. Do not mention price, "
                "revenue, purchase probability, conversion, popularity counts, or platform profit.\n\n"
                "Output format: 'The updated description of the first product is: [description]. "
                "\n The updated description of the second product is: [description].'. "
                "Each description must be under 50 words."
            ),
        }
        for key, value in prompts.items():
            config[key] = value

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
