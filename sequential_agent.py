import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from langgraph_tools import format_slide_bullets, get_demo_examples, get_topic_outline

load_dotenv()


class SequentialState(TypedDict, total=False):
    topic: str
    meeting_goal: str
    outline: list[str]
    examples: list[str]
    tool_output: str
    final_output: str


def log(message: str) -> None:
    print(f"[sequential] {message}", flush=True)


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


def planner(state: SequentialState) -> SequentialState:
    log("Planner node started.")
    log(f"Planner received topic: {state['topic']}")
    model = build_model()
    prompt = (
        "Write one short meeting goal for this presentation topic.\n"
        f"Topic: {state['topic']}"
    )
    log(f"Planner prompt:\n{prompt}")
    response = model.invoke(prompt)
    meeting_goal = str(response.content)
    log(f"Planner response: {meeting_goal}")
    return {"meeting_goal": meeting_goal}


def tool_step(state: SequentialState) -> SequentialState:
    log("Tool step started.")
    outline = get_topic_outline.invoke({"topic": state["topic"]})
    log(f"Tool step outline: {outline}")
    examples = get_demo_examples.invoke({"topic": state["topic"]})
    log(f"Tool step examples: {examples}")
    tool_output = format_slide_bullets.invoke(
        {"items": [state["meeting_goal"], *outline[:2], examples[0]]}
    )
    log(f"Tool step formatted output:\n{tool_output}")
    return {
        "outline": outline,
        "examples": examples,
        "tool_output": tool_output,
    }


def writer(state: SequentialState) -> SequentialState:
    log("Writer node started.")
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
    log(f"Writer prompt:\n{prompt}")
    response = model.invoke(prompt)
    final_output = str(response.content)
    log(f"Writer response:\n{final_output}")
    return {"final_output": final_output}


def build_graph():
    log("Building sequential graph.")
    graph = StateGraph(SequentialState)
    graph.add_node("planner", planner)
    graph.add_node("tool_step", tool_step)
    graph.add_node("writer", writer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tool_step")
    graph.add_edge("tool_step", "writer")
    graph.add_edge("writer", END)
    log("Sequential graph compiled.")
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
