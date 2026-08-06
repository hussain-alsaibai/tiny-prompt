"""
tiny-prompt — Zero-dependency prompt templating and chaining for AI agents.

Prompt composition, variable interpolation, chain-of-thought templates,
few-shot example management, multi-model prompt routing, and role-based
system prompt management — all in one file.

Usage:
    from tiny_prompt import Prompt, Role, Chain, few_shot, render

    # Simple template
    prompt = Prompt("Explain {topic} to a {audience} in {tone} tone.")
    print(prompt.render(topic="quantum computing", audience="5th grader", tone="fun"))

    # Role-based
    p = (
        Prompt.system("You are a {role} assistant.")
        + Prompt.user("Help me with {task}.")
        + Prompt.assistant("I'll help you with {task}.")
        + Prompt.user("What about {followup}?")
    )

    # Chain of thought
    chain = Chain()
    chain.step("Think", "Given {problem}, what is the key constraint?")
    chain.step("Plan", "List 3 approaches to solve this.")
    chain.step("Execute", "Implement approach {choice}.")
    chain.step("Verify", "Does this solve {problem}?")

    print(chain.render(problem="build a cache", choice="2"))
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

__version__ = "0.1.0"
__all__ = ["Prompt", "Role", "Chain", "few_shot", "render", "PromptBuilder"]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class Message:
    """A single role-marked message."""
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}

    def render(self, **vars) -> Message:
        return Message(self.role, self.content.format(**vars))


@dataclass
class Prompt:
    """
    A composable prompt template.

    Supports:
    - Variable interpolation: {name}, {count}, etc.
    - Conditional blocks: {{#if var}}...{{/if}}
    - Loop blocks: {{#each items}}...{{/each}}
    - Default values: {name:Alice}
    - Type hints: {name:string, default:Alice}
    """
    _content: str = ""
    _messages: list[Message] = field(default_factory=list)
    _role: Role | None = None

    def __init__(self, content: str = "", role: Role | None = None):
        self._content = content
        self._messages = []
        self._role = role

    @classmethod
    def system(cls, content: str) -> Prompt:
        p = cls(content, Role.SYSTEM)
        p._messages = [Message(Role.SYSTEM, content)]
        return p

    @classmethod
    def user(cls, content: str) -> Prompt:
        p = cls(content, Role.USER)
        p._messages = [Message(Role.USER, content)]
        return p

    @classmethod
    def assistant(cls, content: str) -> Prompt:
        p = cls(content, Role.ASSISTANT)
        p._messages = [Message(Role.ASSISTANT, content)]
        return p

    def __str__(self) -> str:
        if self._messages:
            return "\n".join(f"[{m.role.value}] {m.content}" for m in self._messages)
        return self._content

    def __add__(self, other: Prompt | str) -> Prompt:
        if isinstance(other, str):
            other = Prompt(other, Role.USER)
        merged = Prompt(role=None)
        merged._messages = list(self._messages) + list(other._messages)
        return merged

    def __radd__(self, other: Prompt | str) -> Prompt:
        if isinstance(other, str):
            other = Prompt(other, Role.USER)
        merged = Prompt(role=None)
        merged._messages = list(other._messages) + list(self._messages)
        return merged

    # ── Variable / Template Rendering ───────────────────────────────────

    def render(self, **vars) -> Prompt:
        """Return a new Prompt with variables substituted."""
        rendered = Prompt(role=self._role)

        if self._content:
            content = self._content
            content = self._render_conditionals(content, vars)
            content = self._render_each(content, vars)
            content = self._interpolate(content, vars)
            rendered._content = content
        else:
            rendered._messages = []
            for m in self._messages:
                content = m.content
                content = self._render_conditionals(content, vars)
                content = self._render_each(content, vars)
                content = self._interpolate(content, vars)
                rendered._messages.append(Message(m.role, content))

        return rendered

    def _interpolate(self, text: str, vars: dict) -> str:
        """Replace {var} and {var:default} placeholders."""
        def replacer(match):
            full = match.group(0)          # {name} or {name:default}
            inner = match.group(1)          # name or name:default
            if ":" in inner:
                name, default = inner.split(":", 1)
                val = vars.get(name.strip(), default.strip())
            else:
                val = vars.get(inner.strip(), full)
            return str(val)

        return re.sub(r"\{([^}]+)\}", replacer, text)

    def _render_conditionals(self, text: str, vars: dict) -> str:
        """Process {{#if var}}...{{/if}} blocks."""
        pattern = re.compile(r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)

        def replacer(match):
            var = match.group(1)
            content = match.group(2)
            if vars.get(var):
                return content
            return ""

        return pattern.sub(replacer, text)

    def _render_each(self, text: str, vars: dict) -> str:
        """Process {{#each items as item}}...{{/each}} blocks."""
        pattern = re.compile(
            r"\{\{#each\s+(\w+)\s+as\s+(\w+)\}\}(.*?)\{\{/each\}\}",
            re.DOTALL
        )

        def replacer(match):
            items_key = match.group(1)
            item_name = match.group(2)
            template = match.group(3)
            items = vars.get(items_key, [])
            rendered = []
            for item in items:
                item_vars = dict(vars, **{item_name: item})
                rendered.append(self._interpolate(template, item_vars))
            return "\n".join(rendered)

        return pattern.sub(replacer, text)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_messages(self) -> list[dict]:
        """Return messages in OpenAI chat format."""
        if self._messages:
            return [m.to_dict() for m in self._messages]
        return [{"role": "user", "content": self._content}]

    def to_string(self, join_with: str = "\n") -> str:
        """Return plain text (for non-chat models)."""
        if self._messages:
            return join_with.join(m.content for m in self._messages)
        return self._content


# ── Chain of Thought ─────────────────────────────────────────────────────────

@dataclass
class Step:
    name: str
    instruction: str
    vars: dict = field(default_factory=dict)


class Chain:
    """
    A chain-of-thought (CoT) prompt builder.

    Example:
        chain = Chain()
        chain.step("Think", "Consider {problem}")
        chain.step("Plan", "List approaches to {problem}")
        chain.step("Execute", "Code: {code}")
        print(chain.render(problem="LRU cache", code="..."))
    """

    def __init__(self, preamble: str = "Solve this step by step:"):
        self.preamble = preamble
        self.steps: list[Step] = []

    def step(self, name: str, instruction: str) -> Chain:
        """Add a step. Returns self for chaining."""
        self.steps.append(Step(name=name, instruction=instruction))
        return self

    def render(self, **vars) -> str:
        """Render the full chain with variables."""
        lines = [self.preamble, ""]
        for i, step in enumerate(self.steps, 1):
            content = re.sub(r"\{([^}]+)\}", lambda m: str(vars.get(m.group(1), m.group(0))), step.instruction)
            lines.append(f"Step {i} — {step.name}:")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    def to_messages(self, **vars) -> list[dict]:
        """Render as OpenAI chat messages."""
        content = self.render(**vars)
        return [
            {"role": "system", "content": "You are a reasoning assistant. Follow the chain of thought."},
            {"role": "user", "content": content},
        ]


# ── Few-Shot Helpers ─────────────────────────────────────────────────────────

def few_shot(examples: list[tuple[str | Prompt, str | Prompt]],
             task: str,
             task_role: Role = Role.USER,
             include_labels: bool = True) -> Prompt:
    """
    Build a few-shot prompt from (input, output) examples.

    Args:
        examples: List of (input, output) tuples (Prompt or string)
        task: The actual query to run
        task_role: Role for the task query
        include_labels: Wrap examples with labels like "Example 1:"

    Returns:
        A Prompt with examples + task
    """
    sections: list[str] = []

    if include_labels:
        for i, (inp, out) in enumerate(examples, 1):
            inp_str = str(inp)
            out_str = str(out)
            sections.append(f"Example {i}:\nInput:\n{inp_str}\nOutput:\n{out_str}\n")
    else:
        for inp, out in examples:
            sections.append(f"Input: {inp}\nOutput: {out}\n")

    task_section = f"Task:\n{task}\n" if include_labels else f"{task}\n"

    return Prompt.user("".join(sections) + task_section)


# ── Rendering Helper ─────────────────────────────────────────────────────────

def render(template: str, **vars) -> str:
    """Quick one-liner template render (no Prompt object needed)."""
    p = Prompt(template)
    return p.render(**vars).to_string()


# ── Multi-Model Router ──────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Describes a model's capabilities and context limits."""
    name: str
    max_tokens: int
    supports_vision: bool = False
    supports_functions: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    preferred_roles: set[Role] = field(default_factory=lambda: {Role.USER, Role.ASSISTANT})


class PromptBuilder:
    """
    High-level prompt builder with model routing and cost awareness.

    Example:
        pb = PromptBuilder()
        pb.register_model("gpt-4", ModelProfile("gpt-4", max_tokens=8192,
                      supports_functions=True, cost_per_1k_input=0.03))
        pb.register_model("claude-3", ModelProfile("claude-3", max_tokens=100_000,
                      supports_vision=True))

        prompt = pb.build("Explain {topic}", model="gpt-4", topic="SQL")
        cost = pb.estimate_cost(prompt, "gpt-4")
    """

    def __init__(self):
        self.models: dict[str, ModelProfile] = {}
        self.default_model = "gpt-4"

    def register_model(self, name: str, profile: ModelProfile) -> None:
        self.models[name] = profile
        if not self.default_model:
            self.default_model = name

    def build(self, template: str, model: str | None = None, **vars) -> Prompt:
        """Build and render a prompt for the given model."""
        model = model or self.default_model
        profile = self.models.get(model)
        prompt = Prompt(template).render(**vars)
        return prompt

    def estimate_cost(self, prompt: Prompt, model: str) -> float:
        """Rough cost estimate in USD."""
        profile = self.models.get(model)
        if not profile:
            return 0.0
        text = prompt.to_string()
        input_tokens = len(text) // 4  # rough approximation
        return (input_tokens / 1000) * profile.cost_per_1k_input

    def best_model(self, requires_vision: bool = False,
                   requires_functions: bool = False) -> str | None:
        """Select cheapest suitable model."""
        candidates = [
            (name, p)
            for name, p in self.models.items()
            if (not requires_vision or p.supports_vision)
            and (not requires_functions or p.supports_functions)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1].cost_per_1k_input + x[1].cost_per_1k_output)[0]
