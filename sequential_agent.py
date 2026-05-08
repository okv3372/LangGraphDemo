import os
from typing import TypedDict

from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from langgraph_tools import format_slide_bullets, get_demo_examples, get_topic_outline


class SequentialState(TypedDict, total=False):
    topic: str
    meeting_goal: str
    outline: list[str]
    examples: list[str]
    tool_output: str
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


def planner(state: SequentialState) -> SequentialState:
    model = build_model()
    prompt = (
        "Write one short meeting goal for this presentation topic.\n"
        f"Topic: {state['topic']}"
    )
    response = model.invoke(prompt)
    return {"meeting_goal": response.content}


def tool_step(state: SequentialState) -> SequentialState:
    outline = get_topic_outline.invoke({"topic": state["topic"]})
    examples = get_demo_examples.invoke({"topic": state["topic"]})
    tool_output = format_slide_bullets.invoke(
        {"items": [state["meeting_goal"], *outline[:2], examples[0]]}
    )
    return {
        "outline": outline,
        "examples": examples,
        "tool_output": tool_output,
    }


def writer(state: SequentialState) -> SequentialState:
    model = build_model()
    prompt = f"""
You are writing a very short meeting brief.

Topic: {state['topic']}
Goal: {state['meeting_goal']}
Tool output:
{state['tool_output']}

Write:
1. A short title
2. Three slide bullets
3. One sentence on why this demo matters
""".strip()
    response = model.invoke(prompt)
    return {"final_output": response.content}


def build_graph():
    graph = StateGraph(SequentialState)
    graph.add_node("planner", planner)
    graph.add_node("tool_step", tool_step)
    graph.add_node("writer", writer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tool_step")
    graph.add_edge("tool_step", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"topic": "LangGraph demo for business teams"})
    print(result["final_output"])
