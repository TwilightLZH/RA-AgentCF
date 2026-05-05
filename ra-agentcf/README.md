# RA-AgentCF

This folder contains the first revenue-aware extension of AgentCF. The original
`agentcf` folder is left untouched; RA-specific data conversion, prompts,
revenue/behavior signals, and metrics live here.

## Main Idea

RA-AgentCF keeps the original UserAgent/ItemAgent structure, but augments their
inputs with e-commerce revenue and behavior evidence:

1. `purchase` is treated as strong positive feedback and the recommendation
   target.
2. `view` enters the user history as weak-interest context, but is not treated
   as a hard negative.
3. Item revenue is estimated from purchase price.
4. Item view/purchase counts and user price statistics are treated as observed
   evidence, not as fixed risk labels.
5. Ranking uses preference first, then derives runtime conversion risk from
   agent context and relative behavior evidence.

The default fusion mode is multiplicative, so a high-price product cannot win
purely because it is expensive; it still needs a plausible preference score.

## E-commerce Dataset Preprocessing

Run from the repository root:

```bash
.venv/bin/python ra-agentcf/preprocess_ecommerce.py \
  --input ra-agentcf/dataset/2020-Apr-final.csv \
  --dataset ECommerceRA \
  --min-purchases 3 \
  --candidate-size 100
```

This generates:

```text
ra-agentcf/dataset/ECommerceRA/ECommerceRA.train.inter
ra-agentcf/dataset/ECommerceRA/ECommerceRA.valid.inter
ra-agentcf/dataset/ECommerceRA/ECommerceRA.test.inter
ra-agentcf/dataset/ECommerceRA/ECommerceRA.item
ra-agentcf/dataset/ECommerceRA/ECommerceRA.random
ra-agentcf/dataset/ECommerceRA/ECommerceRA.revenue.csv
ra-agentcf/dataset/ECommerceRA/ECommerceRA.item_behavior.csv
ra-agentcf/dataset/ECommerceRA/ECommerceRA.user_profile.csv
```

## Smoke Test

Run a tiny API-saving smoke test:

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

`--debug=False` only hides full config dumps and initialization-time internal
prints. Evaluation progress bars and the final metrics are still shown. Use
`--debug=True` if you want to see the full configuration and all initialization
details while diagnosing a run.

The config chain is:

```text
../agentcf/props/overall.yaml
../agentcf/props/AgentCF.yaml
props/RAAgentCF.yaml
props/{dataset}.yaml, if it exists; otherwise ../agentcf/props/{dataset}.yaml
```

## Full Test

```bash
.venv/bin/python ra-agentcf/run.py -m RAAgentCF -d ECommerceRASmall \
  --debug=False \
  --test_only=True \
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