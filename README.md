# RA-AgentCF

RA-AgentCF is a revenue-aware extension of AgentCF for e-commerce recommendation. It keeps the original AgentCF multi-agent workflow, where UserAgent stores user preference memory, ItemAgent stores item memory, and RecAgent ranks candidate items. On top of this workflow, RA-AgentCF adds revenue-aware signals such as item price/revenue, user price tolerance, item view/purchase behavior, conversion risk, and commercial utility.

The original AgentCF source code remains in `agentcf/`. The RA-AgentCF implementation lives in `ra-agentcf/`.

## Project Layout

```text
agentcf/                         Original AgentCF source code
ra-agentcf/
  run.py                         RA-AgentCF entry point
  preprocess_ecommerce.py        E-commerce CSV preprocessing script
  revenue.py                     RevenueProfile and RevenueAgent
  trainer.py                     AgentCF trainer wrapper with revenue metrics
  model/
    raagentcf.py                 Revenue-aware AgentCF
    agentcf.py                   Preference-only AgentCF baseline for e-commerce data
  props/
    RAAgentCF.yaml               RA-AgentCF global config and prompts
    ECommerceRA.yaml             Full e-commerce dataset config
    ECommerceRASmall.yaml        API-affordable small dataset config
  dataset/
    ECommerceRASmall/            Generated small experimental dataset
```

## Method Summary

RA-AgentCF compares two model variants on the same e-commerce candidate-ranking task:

- `AgentCF`: preference-only baseline. It sees product title, brand, category, and preference-related memory, but does not use price, revenue, conversion risk, or revenue-aware reranking.
- `RAAgentCF`: revenue-aware model. It keeps AgentCF's memory-update mechanism and additionally injects revenue-aware context into candidate descriptions and applies configurable revenue-aware reranking.

The current commercial utility is defined as:

```text
utility = normalized_revenue * user_acceptance * conversion_confidence
```

where:

- `normalized_revenue` is item revenue normalized with a configurable method, currently `log_percentile`.
- `user_acceptance` estimates whether the user can plausibly accept the item price and positioning.
- `conversion_confidence = 1 - conversion_risk`, derived from item view/purchase behavior relative to the global conversion baseline.

The default reranking mode is `constrained_utility`: optimize utility inside a preference gate. Items with low AgentCF preference scores are penalized, so the model does not blindly recommend commercially attractive but irrelevant products.

## Environment

Run commands from the repository root:

```bash
cd /path/to/RA-AgentCF
```

The code uses OpenAI-compatible LLM APIs through the config in `agentcf/props/AgentCF.yaml`. Configure your API key and base URL there.


## Preprocess E-commerce Data

The preprocessing script converts an e-commerce behavior CSV into RecBole/AgentCF files.

```bash
.venv/bin/python ra-agentcf/preprocess_ecommerce.py \
  --input ra-agentcf/dataset/2020-Apr-final.csv \
  --dataset ECommerceRASmall \
  --min-purchases 3 \
  --candidate-size 100 \
  --max-users 100 \
  --max-train-examples-per-user 5 \
  --max-history-items 50
```

Generated files:

```text
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.train.inter
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.valid.inter
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.test.inter
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.item
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.random
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.revenue.csv
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.item_behavior.csv
ra-agentcf/dataset/ECommerceRASmall/ECommerceRASmall.user_profile.csv
```

Preprocessing parameters:

| Parameter | Meaning |
| --- | --- |
| `--input` | Source e-commerce CSV path. |
| `--dataset` | Output dataset name. Files are written to `ra-agentcf/dataset/{dataset}/`. |
| `--min-purchases` | Minimum generated purchase examples required per user. At least 3 is needed for train/valid/test split. |
| `--candidate-size` | Number of sampled non-positive candidates written to `{dataset}.random`. |
| `--max-users` | Optional cap on selected users. Users with more purchase examples are preferred. |
| `--max-train-examples-per-user` | Optional cap on train examples per user. Newer train examples are kept. |
| `--max-history-items` | Optional cap on item history length per example. |
| `--seed` | Random seed for candidate sampling. |

## Smoke Test

Use this to check that the evaluation path works while minimizing API usage:

```bash
.venv/bin/python ra-agentcf/run.py -m RAAgentCF -d ECommerceRASmall \
  --debug=False \
  --test_only=True \
  --loaded=False \
  --saved=False \
  --eval_batch_size=1 \
  --max_his_len=20 \
  --MAX_ITEM_LIST_LENGTH=20 \
  --debug_eval_batches=1 \
  --recall_budget=2 \
  --chat_api_batch=1 \
  --api_batch=1
```

## Full Training and Evaluation

Run RA-AgentCF:

```bash
.venv/bin/python ra-agentcf/run.py -m RAAgentCF -d ECommerceRASmall \
  --debug=False \
  --test_only=False \
  --loaded=False \
  --saved=True \
  --epochs=1 \
  --train_batch_size=20 \
  --eval_batch_size=100 \
  --max_his_len=20 \
  --MAX_ITEM_LIST_LENGTH=50 \
  --recall_budget=10 \
  --chat_api_batch=10 \
  --api_batch=20 \
  --show_progress=True
```

Run the preference-only AgentCF baseline on the same dataset:

```bash
.venv/bin/python ra-agentcf/run.py -m AgentCF -d ECommerceRASmall \
  --debug=False \
  --test_only=False \
  --loaded=False \
  --saved=True \
  --epochs=1 \
  --train_batch_size=20 \
  --eval_batch_size=100 \
  --max_his_len=20 \
  --MAX_ITEM_LIST_LENGTH=50 \
  --recall_budget=10 \
  --chat_api_batch=10 \
  --api_batch=20 \
  --show_progress=True
```

## Runtime Parameters

These parameters are passed through `ra-agentcf/run.py` to RecBole and AgentCF configs.

| Parameter | Meaning |
| --- | --- |
| `-m`, `--model` | Model name. Use `RAAgentCF` for revenue-aware reranking or `AgentCF` for the preference-only baseline. |
| `-d`, `--dataset` | Dataset name. Use `ECommerceRASmall` for the small experiment set. |
| `--debug` | If `True`, prints full config and initialization logs. If `False`, hides verbose config dumps while keeping progress bars and final metrics. |
| `--test_only` | If `True`, skip training/memory update and run evaluation only. |
| `--loaded` | If `True`, load saved UserAgent/ItemAgent memory from `saved_idx`. |
| `--saved` | If `True`, save UserAgent/ItemAgent memory after the run. |
| `--saved_idx` | Saved memory index to load when `--loaded=True`. |
| `--epochs` | Number of AgentCF memory-update epochs. |
| `--train_batch_size` | Training batch size. Smaller values reduce API burst size but increase iteration count. |
| `--eval_batch_size` | Evaluation batch size. |
| `--max_his_len` | Maximum number of historical items included in prompts. |
| `--MAX_ITEM_LIST_LENGTH` | Maximum item sequence length used by RecBole. |
| `--recall_budget` | Number of ranked candidate items parsed/scored during evaluation. Usually set to candidate top-K size. |
| `--chat_api_batch` | Batch size for chat-completion API calls. |
| `--api_batch` | Batch size for non-chat LLM/embedding API calls. |
| `--show_progress` | If `True`, show training and evaluation progress bars. |
| `--debug_eval_batches` | Optional evaluation shortcut. For example, `--debug_eval_batches=1` stops evaluation after one batch. |

## RA-AgentCF Config Parameters

Main revenue-aware settings are in `ra-agentcf/props/RAAgentCF.yaml`.

| Config | Meaning |
| --- | --- |
| `ra_enabled` | Master switch for revenue-aware prompt injection and reranking. |
| `ra_alpha` | Trade-off between AgentCF preference score and commercial signal. Larger values make reranking more commercial-oriented. |
| `ra_fusion_mode` | Reranking mode. Supported values include `multiplicative`, `linear_utility`, `linear_revenue`, `constrained_revenue`, and `constrained_utility`. |
| `ra_preference_threshold` | Preference gate threshold for constrained modes. Candidates below this normalized preference score are penalized. |
| `ra_low_preference_penalty` | Multiplicative penalty applied to candidates below the preference threshold. |
| `ra_inject_revenue_context` | If `True`, append revenue-aware platform signals to candidate descriptions during evaluation. |
| `ra_revenue_source` | `file` for dataset revenue CSV, or `synthetic` for stable generated revenue. |
| `ra_revenue_file` | Path to `{dataset}.revenue.csv`. Automatically set by `run.py` for RA datasets. |
| `ra_item_behavior_file` | Path to `{dataset}.item_behavior.csv`. Automatically set by `run.py` for RA datasets. |
| `ra_user_profile_file` | Path to `{dataset}.user_profile.csv`. Automatically set by `run.py` for RA datasets. |
| `ra_revenue_risk_penalty` | Controls how much low user acceptance amplifies conversion risk. |
| `ra_revenue_normalization` | Revenue normalization method: `minmax`, `percentile`, `log`, or `log_percentile`. |
| `ra_revenue_percentile` | Percentile cap used by percentile-based revenue normalization. |

Recommended utility-oriented setting:

```yaml
ra_fusion_mode: constrained_utility
ra_alpha: 0.7
ra_preference_threshold: 0.35
ra_low_preference_penalty: 0.8
ra_revenue_normalization: log_percentile
ra_revenue_percentile: 98
```

## Config Loading Order

`ra-agentcf/run.py` loads configs in this order:

```text
agentcf/props/overall.yaml
agentcf/props/AgentCF.yaml
ra-agentcf/props/RAAgentCF.yaml
ra-agentcf/props/{dataset}.yaml
```

Command-line overrides have the highest priority.

## Metrics

The final output includes standard ranking metrics and revenue-aware metrics:

| Metric | Meaning |
| --- | --- |
| `Recall@K` | Whether the ground-truth purchased item appears in Top-K. |
| `NDCG@K` | Ranking quality that rewards placing the ground-truth item closer to the top. |
| `revenue@K` | Average normalized revenue of Top-K recommended items. |
| `hit_revenue@K` | Raw revenue of the ground-truth item if it appears in Top-K, otherwise 0. |
| `revenue_recall@K` | Recall restricted to positive-revenue targets. In the current dataset it is usually equivalent to Recall@K. |
| `utility@K` | Average commercial utility of Top-K recommended items. |

For this project, `utility@K` is the main revenue-aware objective, while `Recall@K` and `NDCG@K` are used to observe the user-satisfaction trade-off.

## Saved Memory and Records

Agent memory and interaction logs are written under:

```text
ra-agentcf/dataset/{dataset}/record/
ra-agentcf/dataset/{dataset}/saved/
```

With `--loaded=False`, a run starts from initial memory even if previous record folders exist. The record index only increases to avoid overwriting old logs.

With `--loaded=True --saved_idx=N`, the run loads saved memory from index `N`.

## Original AgentCF README

The original upstream AgentCF README is preserved at:

```text
README-AgentCF-original.md
```
