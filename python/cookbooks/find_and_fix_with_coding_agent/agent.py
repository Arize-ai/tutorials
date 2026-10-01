"""A tool-calling support agent that runs on OpenAI or Anthropic.

The provider is picked from whichever API key is set. If both are set, OpenAI
is used.
"""

import json
import os

from dotenv import load_dotenv

from tools import ENABLED_TOOLS, TOOL_SPECS, execute_tool

load_dotenv()

SYSTEM_PROMPT = """You are the customer support assistant for Trailhead Outfitters, an online outdoor gear store.
Help customers with their orders, product availability, and store policies.
Keep answers short and friendly."""

MAX_TURNS = 6
OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"


def get_provider() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment or .env file.")


def run_tool(name: str, arguments: dict) -> str:
    try:
        return execute_tool(name, arguments)
    except Exception as e:
        return f"Error: {e}"


def answer_with_openai(question: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_SPECS[name]["description"],
                "parameters": TOOL_SPECS[name]["parameters"],
            },
        }
        for name in ENABLED_TOOLS
    ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=tools)
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content
        messages.append(message)
        for call in message.tool_calls:
            result = run_tool(call.function.name, json.loads(call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Sorry, I couldn't resolve that. Please contact support@trailhead.example."


def answer_with_anthropic(question: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    tools = [
        {
            "name": name,
            "description": TOOL_SPECS[name]["description"],
            "input_schema": TOOL_SPECS[name]["parameters"],
        }
        for name in ENABLED_TOOLS
    ]
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=1024, system=SYSTEM_PROMPT, messages=messages, tools=tools
        )
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            return "".join(block.text for block in response.content if block.type == "text")
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": use.id, "content": run_tool(use.name, use.input)}
                    for use in tool_uses
                ],
            }
        )
    return "Sorry, I couldn't resolve that. Please contact support@trailhead.example."


def answer(question: str) -> str:
    if get_provider() == "anthropic":
        return answer_with_anthropic(question)
    return answer_with_openai(question)
