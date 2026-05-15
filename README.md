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