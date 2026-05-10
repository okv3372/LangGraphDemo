"""
Parallel demo:
1. Three agents run side by side.
2. Each agent requests its own tool.
3. The results are merged in one final node.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langgraph_tools import (
    get_audience_questions,
    get_demo_examples,
    get_topic_outline,
)

load_dotenv()

TOPIC = "LangGraph demo for business teams"
MESSAGES = Annotated[list, add_messages]

class ParallelState(TypedDict, total=False):
    topic: str
    outline_messages: MESSAGES
    examples_messages: MESSAGES
    questions_messages: MESSAGES
    final_output: str


def build_model() -> AzureChatOpenAI:
    required_env = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
    )


def last_tool_output(messages: list) -> str:
    for message in reversed(messages):
        if message.__class__.__name__ == "ToolMessage":
            return str(message.content)
    return ""


def outline_agent(state: ParallelState) -> ParallelState:
    print("1️⃣ 🟦 Outline agent")
    model = build_model().bind_tools([get_topic_outline], tool_choice="required")
    response = model.invoke(
        f"""
        Topic: {state['topic']}

        Call the outline tool.
        Do not write the final answer.
        """.strip()
    )
    return {"outline_messages": [response]}


def examples_agent(state: ParallelState) -> ParallelState:
    print("1️⃣ 🟩 Examples agent")
    model = build_model().bind_tools([get_demo_examples], tool_choice="required")
    response = model.invoke(
        f"""
        Topic: {state['topic']}

        Call the examples tool.
        Do not write the final answer.
        """.strip()
    )
    return {"examples_messages": [response]}


def questions_agent(state: ParallelState) -> ParallelState:
    print("1️⃣ 🟨 Questions agent")
    model = build_model().bind_tools([get_audience_questions], tool_choice="required")
    response = model.invoke(
        f"""
        Topic: {state['topic']}

        Call the audience questions tool.
        Do not write the final answer.
        """.strip()
    )
    return {"questions_messages": [response]}


outline_tools = ToolNode([get_topic_outline], messages_key="outline_messages")
examples_tools = ToolNode([get_demo_examples], messages_key="examples_messages")
questions_tools = ToolNode([get_audience_questions], messages_key="questions_messages")


def combine(state: ParallelState) -> ParallelState:
    print("2️⃣ Combine")

    outline = last_tool_output(state["outline_messages"])
    examples = last_tool_output(state["examples_messages"])
    questions = last_tool_output(state["questions_messages"])

    print("Outline:", outline)
    print("Examples:", examples)
    print("Questions:", questions)

    model = build_model()
    response = model.invoke(
        f"""
        Topic: {state['topic']}
        Outline: {outline}
        Examples: {examples}
        Questions: {questions}

        Write:
        1. A short title
        2. Three bullets
        3. One sentence explaining why parallel work is useful here
        """.strip()
    )

    return {"final_output": str(response.content)}


def build_graph():
    graph = StateGraph(ParallelState)

    graph.add_node("outline_agent", outline_agent)
    graph.add_node("outline_tools", outline_tools)
    graph.add_node("examples_agent", examples_agent)
    graph.add_node("examples_tools", examples_tools)
    graph.add_node("questions_agent", questions_agent)
    graph.add_node("questions_tools", questions_tools)
    graph.add_node("combine", combine)

    graph.add_edge(START, "outline_agent")
    graph.add_edge(START, "examples_agent")
    graph.add_edge(START, "questions_agent")

    graph.add_edge("outline_agent", "outline_tools")
    graph.add_edge("examples_agent", "examples_tools")
    graph.add_edge("questions_agent", "questions_tools")

    graph.add_edge("outline_tools", "combine")
    graph.add_edge("examples_tools", "combine")
    graph.add_edge("questions_tools", "combine")
    graph.add_edge("combine", END)

    return graph.compile()


if __name__ == "__main__":
    print("🚀 Parallel demo")
    print("Path: 1️⃣ three agents at once -> 2️⃣ their tools run -> 3️⃣ combine")

    app = build_graph()
    result = app.invoke({"topic": TOPIC})

    print("\n=== FINAL OUTPUT ===")
    print(result["final_output"])