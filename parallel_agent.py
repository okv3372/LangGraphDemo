import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from langgraph_tools import (
    format_slide_bullets,
    get_audience_questions,
    get_demo_examples,
    get_topic_outline,
)

load_dotenv()


class ParallelState(TypedDict, total=False):
    topic: str
    outline: list[str]
    examples: list[str]
    questions: list[str]
    bullet_draft: str
    final_output: str


def log(message: str) -> None:
    print(f"[parallel] {message}", flush=True)


def build_model() -> AzureChatOpenAI:
    log("Checking Azure OpenAI environment variables.")
    required_env = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    log("Creating AzureChatOpenAI client.")
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
    )


def outline_branch(state: ParallelState) -> ParallelState:
    log("Outline branch started.")
    outline = get_topic_outline.invoke({"topic": state["topic"]})
    log(f"Outline branch result: {outline}")
    return {"outline": outline}


def examples_branch(state: ParallelState) -> ParallelState:
    log("Examples branch started.")
    examples = get_demo_examples.invoke({"topic": state["topic"]})
    log(f"Examples branch result: {examples}")
    return {"examples": examples}


def questions_branch(state: ParallelState) -> ParallelState:
    log("Questions branch started.")
    questions = get_audience_questions.invoke({"topic": state["topic"]})
    log(f"Questions branch raw questions: {questions}")
    bullet_draft = format_slide_bullets.invoke({"items": questions})
    log(f"Questions branch formatted bullets:\n{bullet_draft}")
    return {"questions": questions, "bullet_draft": bullet_draft}


def combine(state: ParallelState) -> ParallelState:
    log("Combine node started.")
    log(f"Combine received outline: {state['outline']}")
    log(f"Combine received examples: {state['examples']}")
    log(f"Combine received questions: {state['questions']}")
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
    log(f"Combine prompt:\n{prompt}")
    response = model.invoke(prompt)
    final_output = str(response.content)
    log(f"Combine response:\n{final_output}")
    return {"final_output": final_output}


def build_graph():
    log("Building parallel graph.")
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
    log("Parallel graph compiled.")
    return graph.compile()


if __name__ == "__main__":
    log("Loading environment variables from .env if present.")
    app = build_graph()
    topic = "LangGraph demo for business teams"
    log(f"Running graph with topic: {topic}")
    result = app.invoke({"topic": topic})
    log("Graph run complete.")
    log(f"Final state keys: {list(result.keys())}")
    print("\n=== FINAL OUTPUT ===", flush=True)
    print(result["final_output"])
