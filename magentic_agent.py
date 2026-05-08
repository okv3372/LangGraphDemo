import os
from typing import TypedDict

from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from langgraph_tools import (
    format_slide_bullets,
    get_demo_examples,
    get_topic_outline,
)


class MagenticState(TypedDict, total=False):
    topic: str
    coordinator_note: str
    research_brief: str
    example_brief: str
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


def coordinator(state: MagenticState) -> MagenticState:
    model = build_model()
    prompt = (
        "You are a coordinator for a tiny multi-agent demo.\n"
        f"Topic: {state['topic']}\n"
        "Write one short note telling two specialists what to focus on."
    )
    response = model.invoke(prompt)
    return {"coordinator_note": response.content}


def research_specialist(state: MagenticState) -> MagenticState:
    model = build_model()
    outline = get_topic_outline.invoke({"topic": state["topic"]})
    prompt = f"""
You are the research specialist.

Coordinator note: {state['coordinator_note']}
Topic: {state['topic']}
Outline: {outline}

Write two short bullets with the most important points.
""".strip()
    response = model.invoke(prompt)
    return {"research_brief": response.content}


def examples_specialist(state: MagenticState) -> MagenticState:
    model = build_model()
    examples = get_demo_examples.invoke({"topic": state["topic"]})
    prompt = f"""
You are the demo examples specialist.

Coordinator note: {state['coordinator_note']}
Topic: {state['topic']}
Examples: {examples}

Write two short bullets with the best demo examples.
""".strip()
    response = model.invoke(prompt)
    return {"example_brief": response.content}


def coordinator_finalize(state: MagenticState) -> MagenticState:
    model = build_model()
    bullet_draft = format_slide_bullets.invoke(
        {"items": [state["research_brief"], state["example_brief"]]}
    )
    prompt = f"""
You are the coordinator writing the final answer.

Topic: {state['topic']}
Coordinator note: {state['coordinator_note']}
Specialist summaries:
{bullet_draft}

Write:
1. A short title
2. Three bullets
3. One sentence explaining the value of coordinator + specialist agents
""".strip()
    response = model.invoke(prompt)
    return {"final_output": response.content}


def build_graph():
    graph = StateGraph(MagenticState)
    graph.add_node("coordinator", coordinator)
    graph.add_node("research_specialist", research_specialist)
    graph.add_node("examples_specialist", examples_specialist)
    graph.add_node("coordinator_finalize", coordinator_finalize)
    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "research_specialist")
    graph.add_edge("coordinator", "examples_specialist")
    graph.add_edge("research_specialist", "coordinator_finalize")
    graph.add_edge("examples_specialist", "coordinator_finalize")
    graph.add_edge("coordinator_finalize", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"topic": "LangGraph demo for business teams"})
    print(result["final_output"])
