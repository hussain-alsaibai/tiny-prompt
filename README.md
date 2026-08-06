# tiny-prompt — Prompt Templating for AI Agents

> **Variable interpolation, chain-of-thought, few-shot, multi-model routing. Zero deps. One file.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](tiny_prompt.py)

`tiny-prompt` gives AI agents a structured way to build, compose, and render
prompt templates. Variables, conditionals, loops, chain-of-thought chains,
few-shot examples, role-based composition, and model-cost-aware routing —
all in pure Python.

```bash
pip install tiny-prompt
```

## Quick Start

### Variable Interpolation

```python
from tiny_prompt import Prompt

p = Prompt("Explain {topic} to a {audience} in {tone} tone.")
rendered = p.render(topic="quantum computing", audience="5th grader", tone="fun")
print(rendered)
# Explain quantum computing to a 5th grader in fun tone.
```

### Role-Based Composition

```python
from tiny_prompt import Prompt, Role

p = (
    Prompt.system("You are a {role} assistant.")
    + Prompt.user("Help me with {task}.")
    + Prompt.assistant("I'll help you with {task}.")
    + Prompt.user("What about {followup}?")
)

messages = p.render(role="SQL", task="query optimization", followup="indexes").to_messages()
# [{"role": "system", "content": "..."},
#  {"role": "user",   "content": "..."}, ...]
```

### Conditionals & Loops

```python
from tiny_prompt import Prompt

p = Prompt("""\
{{#if premium}}
Upgrade note: {upgrade_msg}
{{/if}}
{{#each features as feature}}
- {feature}
{{/each}}
""")

print(p.render(
    premium=True,
    upgrade_msg="You have access to 1000 requests/day",
    features=["Fast caching", "Priority support", "Custom domains"]
))
```

### Chain of Thought

```python
from tiny_prompt import Chain

chain = Chain(preamble="Solve this step by step:")
chain.step("Think",    "Given {problem}, identify the key constraint.")
chain.step("Research", "Find relevant patterns from {examples}.")
chain.step("Plan",     "List 3 approaches ranked by {criteria}.")
chain.step("Execute",  "Implement approach {choice}.")
chain.step("Verify",   "Does this solve {problem}?")

print(chain.render(
    problem="build a distributed cache",
    examples="Redis, Memcached, DynamoDB",
    criteria="consistency, latency, simplicity",
    choice="2"
))
```

### Few-Shot Examples

```python
from tiny_prompt import few_shot, Role

examples = [
    ("Translate to French: Hello", "Bonjour"),
    ("Translate to French: Goodbye", "Au revoir"),
]

prompt = few_shot(examples, "Translate to French: Thank you")
print(prompt)
```

### Multi-Model Prompt Builder

```python
from tiny_prompt import PromptBuilder, ModelProfile

pb = PromptBuilder()
pb.register_model("gpt-4", ModelProfile(
    "gpt-4", max_tokens=8192,
    supports_functions=True,
    cost_per_1k_input=0.03, cost_per_1k_output=0.06
))
pb.register_model("claude-3", ModelProfile(
    "claude-3", max_tokens=100_000,
    supports_vision=True,
    cost_per_1k_input=0.015, cost_per_1k_output=0.075
))

prompt = pb.build("Summarize {document} in {n} bullets",
                  model="gpt-4", document="...", n=5)

best = pb.best_model(requires_vision=False, requires_functions=True)
print(f"Cheapest function-calling model: {best}")
```

## Feature Reference

| Feature | Syntax | Example |
|---------|--------|---------|
| Variables | `{name}` | `{topic}` |
| Default values | `{name:default}` | `{tone:helpful}` |
| Conditionals | `{{#if var}}...{{/if}}` | `{{#if premium}}...{{/if}}` |
| Loops | `{{#each items as item}}...{{/each}}` | `{{#each items as item}}{item}{{/each}}` |
| Roles | `Prompt.system/user/assistant()` | `Prompt.system("You are...")` |
| Composition | `+` operator | `system + user + assistant` |
| Chain of Thought | `Chain().step()` | `chain.step("Think", "...")` |
| Few-shot | `few_shot(examples, task)` | `few_shot([("Q","A")], "Q2")` |
| OpenAI format | `.to_messages()` | `[{"role": ..., "content": ...}]` |

## Files

- `tiny_prompt.py` — The library (~400 LOC)
- `tests/test_tiny_prompt.py` — Tests
- `LICENSE` — MIT
- `package.json` — Metadata

## Ecosystem

Part of the **[tiny-\* ecosystem](https://github.com/hussain-alsaibai)** —
zero-dependency, single-file Python libraries for AI agents and modern
backend tooling. See also: `tiny-audit`, `tiny-diff`, `tiny-chain`,
`tiny-eval`, `tiny-workflow`.

---

*Built by [OpenClaw](https://github.com/hussain-alsaibai) 🦞*
