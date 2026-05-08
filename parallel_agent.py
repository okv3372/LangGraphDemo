import os
from typing import TypedDict

from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from langgraph_tools import (
    format_slide_bullets,
    get_audience_questions,
    get_demo_examples,
    get_topic_outline,
)


class ParallelState(TypedDict, total=False):
    topic: str
    outline: list[str]
    examples: list[str]
    questions: list[str]
    bullet_draft: str
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


def outline_branch(state: ParallelState) -> ParallelState:
    return {"outline": get_topic_outline.invoke({"topic": state["topic"]})}


def examples_branch(state: ParallelState) -> ParallelState:
    return {"examples": get_demo_examples.invoke({"topic": state["topic"]})}


def questions_branch(state: ParallelState) -> ParallelState:
    questions = get_audience_questions.invoke({"topic": state["topic"]})
    bullet_draft = format_slide_bullets.invoke({"items": questions})
    return {"questions": questions, "bullet_draft": bullet_draft}


def combine(state: ParallelState) -> ParallelState:
    model = build_model()
    prompt = f"""
You are combining parallel analysis into a short slide-ready answer.

Topic: {state['topic']}
Outline: {state['outline']}
Examples: {state['examples']}
Audience questions:
{state['bullet_draft']}

Write:
1. A short title
2. Three bullets
3. One sentence explaining why parallel branches are useful here
""".strip()
    response = model.invoke(prompt)
    return {"final_output": response.content}


def build_graph():
    graph = StateGraph(ParallelState)
    graph.add_node("outline_branch", outline_branch)
    graph.add_node("examples_branch", examples_branch)
    graph.add_node("questions_branch", questions_branch)
    graph.add_node("combine", combine)

    # Multiple edges from START make the three branches run independently.
    graph.add_edge(START, "outline_branch")
    graph.add_edge(START, "examples_branch")
    graph.add_edge(START, "questions_branch")
    graph.add_edge("outline_branch", "combine")
    graph.add_edge("examples_branch", "combine")
    graph.add_edge("questions_branch", "combine")
    graph.add_edge("combine", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"topic": "LangGraph demo for business teams"})
    print(result["final_output"])
