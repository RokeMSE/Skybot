                        ┌──────────────┐
                        │  API Gateway │  (auth, rate limit, logging)
                        └──────┬───────┘
                               │
              ┌────────────────┼──────────────────┐
              │                                   │
    ┌─────────▼──────────┐           ┌────────────▼────────────┐
    │  Conversation Mgr  │           │  Alert Fast Path         │
    │  (session state,   │           │  (rule-based, no LLM,    │
    │   history store)   │           │   direct → SAS/SC UI)    │
    └─────────┬──────────┘           └─────────────────────────┘
              │
    ┌─────────▼──────────────────────────────────┐
    │         ReAct Orchestrator                  │
    │                                             │
    │  loop:                                      │
    │    1. Reason (what do I know so far?)       │
    │    2. Act (pick tool/agent)                 │
    │    3. Observe (read result into scratchpad) │
    │    4. Repeat until confident or maxsteps    │
    └──┬──────────┬─────────────┬────────────────┘
       │          │             │
  ┌────▼────┐ ┌───▼────┐  ┌────▼──────┐
  │ Issue   │ │  SOP/  │  │  MCP      │
  │ Invest. │ │  Doc   │  │  Tools    │
  │ Agent   │ │  Agent │  │ (data DBs)│
  └────┬────┘ └───┬────┘  └───────────┘
       └──────────┘
             │  (write to shared scratchpad)
             ▼
    ┌────────────────────┐
    │  HITL Gate         │  ← low confidence or high impact → human review
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  Reporting Agent   │  (structure, finalize, send)
    └────────────────────┘
