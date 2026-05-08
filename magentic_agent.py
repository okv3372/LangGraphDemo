import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from langgraph_tools import (
    format_slide_bullets,
    get_demo_examples,
    get_topic_outline,
)

load_dotenv()


class MagenticState(TypedDict, total=False):
    topic: str
    coordinator_note: str
    research_brief: str
    example_brief: str
    final_output: str


def log(message: str) -> None:
    print(f"[magentic] {message}", flush=True)


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


def coordinator(state: MagenticState) -> MagenticState:
    log("Coordinator node started.")
    log(f"Coordinator received topic: {state['topic']}")
    model = build_model()
    prompt = (
        "You are a coordinator for a tiny multi-agent demo.\n"
        f"Topic: {state['topic']}\n"
        "Write one short note telling two specialists what to focus on."
    )
    log(f"Coordinator prompt:\n{prompt}")
    response = model.invoke(prompt)
    coordinator_note = str(response.content)
    log(f"Coordinator response: {coordinator_note}")
    return {"coordinator_note": coordinator_note}


def research_specialist(state: MagenticState) -> MagenticState:
    log("Research specialist started.")
    model = build_model()
    outline = get_topic_outline.invoke({"topic": state["topic"]})
    log(f"Research specialist outline: {outline}")
    prompt = f"""
You are the research specialist.

Coordinator note: {state['coordinator_note']}
Topic: {state['topic']}
Outline: {outline}

Write two short bullets with the most important points.
""".strip()
    log(f"Research specialist prompt:\n{prompt}")
    response = model.invoke(prompt)
    research_brief = str(response.content)
    log(f"Research specialist response:\n{research_brief}")
    return {"research_brief": research_brief}


def examples_specialist(state: MagenticState) -> MagenticState:
    log("Examples specialist started.")
    model = build_model()
    examples = get_demo_examples.invoke({"topic": state["topic"]})
    log(f"Examples specialist examples: {examples}")
    prompt = f"""
You are the demo examples specialist.

Coordinator note: {state['coordinator_note']}
Topic: {state['topic']}
Examples: {examples}

Write two short bullets with the best demo examples.
""".strip()
    log(f"Examples specialist prompt:\n{prompt}")
    response = model.invoke(prompt)
    example_brief = str(response.content)
    log(f"Examples specialist response:\n{example_brief}")
    return {"example_brief": example_brief}


def coordinator_finalize(state: MagenticState) -> MagenticState:
    log("Coordinator finalize node started.")
    model = build_model()
    bullet_draft = format_slide_bullets.invoke(
        {"items": [state["research_brief"], state["example_brief"]]}
    )
    log(f"Coordinator finalize combined specialist notes:\n{bullet_draft}")
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
    log(f"Coordinator finalize prompt:\n{prompt}")
    response = model.invoke(prompt)
    final_output = str(response.content)
    log(f"Coordinator finalize response:\n{final_output}")
    return {"final_output": final_output}


def build_graph():
    log("Building magentic graph.")
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
    log("Magentic graph compiled.")
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
