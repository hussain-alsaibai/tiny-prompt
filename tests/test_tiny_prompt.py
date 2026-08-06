"""Tests for tiny-prompt."""
import pytest
from tiny_prompt import Prompt, Role, Chain, few_shot, render, PromptBuilder, ModelProfile

def test_prompt_basic_render():
    p = Prompt("Hello {name}, you have {n} messages.")
    rendered = p.render(name="Alice", n=5)
    assert "Alice" in str(rendered)
    assert "5" in str(rendered)

def test_prompt_default_value():
    p = Prompt("Greeting: {name:Guest}")
    rendered_no_var = p.render()
    assert "Guest" in str(rendered_no_var)
    rendered_with_var = p.render(name="Bob")
    assert "Bob" in str(rendered_with_var)

def test_prompt_system_role():
    p = Prompt.system("You are {role}.")
    assert p._messages[0].role == Role.SYSTEM
    rendered = p.render(role="assistant")
    assert "assistant" in str(rendered)

def test_prompt_user_role():
    p = Prompt.user("Help with {task}.")
    assert p._messages[0].role == Role.USER

def test_prompt_assistant_role():
    p = Prompt.assistant("The answer is {answer}.")
    assert p._messages[0].role == Role.ASSISTANT

def test_prompt_composition():
    p = Prompt.system("You are a {role}.") + Prompt.user("Tell me about {topic}.")
    assert len(p._messages) == 2
    assert p._messages[0].role == Role.SYSTEM
    assert p._messages[1].role == Role.USER

def test_prompt_to_messages():
    p = Prompt.system("Sys") + Prompt.user("Q")
    msgs = p.to_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "Sys"
    assert msgs[1]["role"] == "user"

def test_prompt_to_string():
    p = Prompt.system("Sys") + Prompt.user("Q")
    s = p.to_string()
    assert "Sys" in s
    assert "Q" in s

def test_chain_basic():
    chain = Chain()
    chain.step("Think", "Consider {problem}")
    chain.step("Plan", "List solutions for {problem}")
    result = chain.render(problem="caching")
    assert "Think" in result
    assert "caching" in result
    assert "Plan" in result

def test_chain_returns_self():
    chain = Chain().step("A", "Do {x}").step("B", "Then {y}")
    assert len(chain.steps) == 2

def test_chain_to_messages():
    chain = Chain().step("Step1", "Do {task}")
    msgs = chain.to_messages(task="X")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Step1" in msgs[1]["content"]

def test_few_shot():
    examples = [
        ("Input: 1+1", "Output: 2"),
        ("Input: 2+2", "Output: 4"),
    ]
    prompt = few_shot(examples, "Input: 3+3")
    s = str(prompt)
    assert "1+1" in s
    assert "3+3" in s

def test_few_shot_without_labels():
    examples = [
        ("A", "B"),
        ("C", "D"),
    ]
    prompt = few_shot(examples, "E", include_labels=False)
    s = str(prompt)
    assert "Example" not in s
    assert "A" in s

def test_render_helper():
    result = render("Hello {who}", who="World")
    assert result == "Hello World"

def test_prompt_conditional_true():
    p = Prompt("{{#if premium}}Upgrade now{{/if}}End")
    rendered = p.render(premium=True)
    assert "Upgrade now" in str(rendered)
    assert "End" in str(rendered)

def test_prompt_conditional_false():
    p = Prompt("{{#if premium}}Upgrade now{{/if}}End")
    rendered = p.render(premium=False)
    assert "Upgrade now" not in str(rendered)
    assert "End" in str(rendered)

def test_prompt_each_loop():
    p = Prompt("{{#each items as item}}- {item}\n{{/each}}Done")
    rendered = p.render(items=["apple", "banana"])
    s = str(rendered)
    assert "apple" in s
    assert "banana" in s
    assert "Done" in s

def test_promptbuilder_register():
    pb = PromptBuilder()
    pb.register_model("gpt-4", ModelProfile("gpt-4", max_tokens=8192))
    assert "gpt-4" in pb.models

def test_promptbuilder_best_model():
    pb = PromptBuilder()
    pb.register_model("claude", ModelProfile("claude", max_tokens=100_000,
                      cost_per_1k_input=0.015))
    pb.register_model("gpt", ModelProfile("gpt", max_tokens=8192,
                      cost_per_1k_input=0.03))
    best = pb.best_model()
    assert best == "claude"

def test_promptbuilder_estimate_cost():
    pb = PromptBuilder()
    pb.register_model("gpt-4", ModelProfile("gpt-4", max_tokens=8192,
                      cost_per_1k_input=0.03))
    p = Prompt("Hello world").render()
    cost = pb.estimate_cost(p, "gpt-4")
    assert cost > 0

def test_promptbuilder_build():
    pb = PromptBuilder()
    pb.register_model("gpt-4", ModelProfile("gpt-4", max_tokens=8192))
    p = pb.build("Say {word}", word="hi")
    assert "hi" in str(p)
