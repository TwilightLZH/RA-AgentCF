# RA-AgentCF

This folder contains the first revenue-aware extension of AgentCF.

The implementation intentionally reuses the original `agentcf` codebase for
datasets, AgentVerse agents, prompts, and saved memories. New RA-specific logic
lives here so the original AgentCF baseline stays comparable.

## Main Idea

RA-AgentCF adds a lightweight `RevenueAgent` around AgentCF:

1. It builds an item-level revenue profile.
2. It injects revenue-aware platform utility signals into the LLM reranking
   prompt.
3. It combines the LLM preference ranking with contextual revenue utility.
4. It reports both recommendation metrics and revenue-aware metrics.

The simple weighted score is still configurable through `ra_alpha`, but the
revenue signal is also exposed to the agent reasoning process through prompt
context.

## Smoke Test

Run from this directory:

```bash
cd ra-agentcf

python run.py -m RAAgentCF -d CDs-100-user-dense \
  --debug=False \
  --test_only=True \
  --loaded=True \
  --saved=False \
  --saved_idx=1000 \
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
../agentcf/props/{dataset}.yaml
props/RAAgentCF.yaml
```
