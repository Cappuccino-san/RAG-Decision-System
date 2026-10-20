# Design Notes — RAG-Powered Decision System

## Why structured output is the product

The consumer of this system is not a person reading prose — it is a workflow
(a ticket router, an approval flow, a dashboard) that needs to branch on the
answer. So the output is a validated schema:

- `decision` — an enum a program can switch on
- `citations` — the evidence a human auditor can check
- `risk_flags` — named rules, so a decision can be explained and counted
- `confidence` — a number a threshold policy can act on

## Confidence is computed, not requested

Asking an LLM "how confident are you?" returns a plausible number, not a
measurement. Here, confidence is a documented function of the retrieval
evidence (see `scoring.py`):

```
confidence = 0.60 * top_score + 0.25 * mean(top 3 scores) + 0.15 * source coverage
```

Weak or off-topic retrieval produces a low score, and below the threshold
the system returns `needs_human_review` instead of guessing. The threshold
(0.30 by default) is calibrated for the offline embedder; real semantic
embeddings produce higher similarities, so re-run the eval and re-tune when
switching modes — that calibration step is part of the design, not a bug.

## Decision order (and why)

1. **Reject flags first.** A request matching a written prohibition
   (passwords in a spreadsheet, vendor sharing without a DPA) is rejected
   even if retrieval is weak — refusing is the safe failure mode, and the
   flag itself is the explanation.
2. **Weak evidence → human.** If nothing prohibited matched but the
   policies retrieved don't really cover the request, a human decides.
   Automating on weak evidence is how RAG systems hallucinate authority.
3. **Condition flags → flag.** Allowed only with a condition the request
   doesn't state (contractor access with no mention of approval).
   Suppressed when the request states the condition — flags people learn
   to ignore are worse than no flags.
4. **Otherwise → approve**, with the supporting citations attached.

## The LLM decider's guardrails

In `llm` mode a LangChain model proposes the decision from the retrieved
excerpts — but two rules are enforced in code around it:

- a reject-level flag **overrides** an LLM approval;
- low confidence **downgrades** any LLM decision to `needs_human_review`,
  and the returned confidence is our computed score, never the model's.

The model advises inside a fence; it does not hold the keys.

## Chunking tradeoff

Section-aware splitting (by heading, ~600 chars, ~100 overlap) was chosen
over fixed-size splitting because citations need a section name a human
recognizes, and policy rules usually live inside one section. The cost is
uneven chunk sizes; the benefit shows up in citation quality. The eval's
retrieval hit rate is how you'd test a different strategy.

## Why an in-memory vector store by default?

A corpus of dozens of policies is thousands of vectors at most — brute-force
cosine is exact, instant, and dependency-free at that scale. The store hides
behind `add / search / save / load`, so Chroma (bundled as an optional
extra) or pgvector drops in when the corpus outgrows memory, without
touching retrieval or decision code. Choosing the boring option at the
right scale, with a clean exit, is the engineering point.

## Known limitations of the rules layer

Keyword rules are a demonstration pattern: they can false-positive on
phrasing (e.g. "email" near "password") and they cannot understand intent.
In production this layer would be a maintained ruleset owned with the
compliance team, with the eval set grown from real reviewed requests —
the LLM decider exists for the long tail the rules don't cover, inside
the guardrails above.

## Interview cheat-sheet: likely questions

- *"How do you know your RAG is any good?"* — The eval scores retrieval hit
  rate and decision accuracy separately, per embedding mode.
- *"Why not just let the LLM decide?"* — Auditability: named rules,
  replayable decisions, and prohibitions a model can't talk past.
- *"What breaks first at scale?"* — Brute-force search; that's why the
  vector store is an interface, not an implementation detail.
- *"How would you add a new policy?"* — Drop the Markdown in `data/policies`,
  reindex, add eval cases for it. No code changes.
