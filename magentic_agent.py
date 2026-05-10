"""
Magentic-style demo:
1. A coordinator decides the next step.
2. It can send work to a researcher or a planner.
3. The researcher uses tools.
4. The graph loops until the coordinator decides to finish.
"""

import os
from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph_tools import (
    get_audience_questions,
    get_demo_examples,
    get_topic_outline,
)

load_dotenv()

TOPIC = "LangGraph demo for business teams"
MAX_ROUNDS = 4
RESEARCH_TOOLS = [get_topic_outline, get_demo_examples, get_audience_questions]
NOTES = Annotated[list[str], add]


class MagenticState(TypedDict, total=False):
    topic: str
    round_count: int
    next_step: str
    coordinator_note: str
    research_messages: list
    research_notes: NOTES
    plan_notes: NOTES
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


def format_notes(notes: list[str] | None) -> str:
    if not notes:
        return "None yet."
    return "\n\n".join(notes)


def last_tool_outputs(messages: list) -> str:
    outputs = []
    for message in messages:
        if message.__class__.__name__ == "ToolMessage":
            outputs.append(f"{message.name}: {message.content}")
    return "\n".join(outputs)


def parse_coordinator_response(text: str) -> tuple[str, str]:
    step = "finish"
    note = text.strip()

    for line in text.splitlines():
        if line.upper().startswith("STEP:"):
            step = line.split(":", 1)[1].strip().lower()
        if line.upper().startswith("NOTE:"):
            note = line.split(":", 1)[1].strip()

    if step not in {"research", "plan", "finish"}:
        step = "finish"

    return step, note


def coordinator(state: MagenticState) -> MagenticState:
    round_number = state.get("round_count", 0) + 1
    print(f"1️⃣🎯 Coordinator (round {round_number})")

    if round_number > MAX_ROUNDS:
        print("Max rounds reached. Finishing with current notes.")
        return {
            "round_count": round_number,
            "next_step": "finish",
            "coordinator_note": "We have enough information. Finish the answer.",
        }

    model = build_model()
    response = model.invoke(
        f"""
        You are coordinating a small research workflow.

        Topic: {state['topic']}
        Round: {round_number} of {MAX_ROUNDS}

        Research notes so far:
        {format_notes(state.get('research_notes'))}

        Plan notes so far:
        {format_notes(state.get('plan_notes'))}

        Choose the next step:
        - research: gather more information
        - plan: organize the information and identify gaps
        - finish: stop when the work is complete

        Reply in exactly this format:
        STEP: research OR plan OR finish
        NOTE: one short instruction for the next node
        """.strip()
    )

    step, note = parse_coordinator_response(str(response.content))
    print("Coordinator decision:", step)
    print("Coordinator note:", note)

    return {
        "round_count": round_number,
        "next_step": step,
        "coordinator_note": note,
    }


def route_from_coordinator(state: MagenticState) -> str:
    return state["next_step"]


def researcher(state: MagenticState) -> MagenticState:
    print("2️⃣🔎 Researcher")
    model = build_model().bind_tools(RESEARCH_TOOLS, tool_choice="required")

    response = model.invoke(
        f"""
        Topic: {state['topic']}
        Coordinator instruction: {state['coordinator_note']}

        Research notes so far:
        {format_notes(state.get('research_notes'))}

        Call one or more tools to gather the next useful information.
        Do not write the final answer.
        """.strip()
    )

    print("Requested tools:", [call["name"] for call in response.tool_calls])
    return {"research_messages": [response]}


research_tools = ToolNode(RESEARCH_TOOLS, messages_key="research_messages")


def research_summary(state: MagenticState) -> MagenticState:
    print("3️⃣📚 Research summary")
    tool_outputs = last_tool_outputs(state["research_messages"])
    print("Latest tool outputs:")
    print(tool_outputs)

    model = build_model()
    response = model.invoke(
        f"""
        Topic: {state['topic']}
        Coordinator instruction: {state['coordinator_note']}
        Tool outputs:
        {tool_outputs}

        Write 2 short bullets with the most useful new findings.
        """.strip()
    )

    note = str(response.content)
    print("Research note:")
    print(note)
    return {"research_notes": [note]}


def planner(state: MagenticState) -> MagenticState:
    print("2️⃣🗂️ Planner")
    model = build_model()

    response = model.invoke(
        f"""
        Topic: {state['topic']}
        Coordinator instruction: {state['coordinator_note']}

        Research notes so far:
        {format_notes(state.get('research_notes'))}

        Write:
        1. A short working plan
        2. What is still missing, if anything
        """.strip()
    )

    note = str(response.content)
    print("Plan note:")
    print(note)
    return {"plan_notes": [note]}


def finish(state: MagenticState) -> MagenticState:
    print("4️⃣✅ Finish")
    model = build_model()

    response = model.invoke(
        f"""
        Topic: {state['topic']}

        Research notes:
        {format_notes(state.get('research_notes'))}

        Plan notes:
        {format_notes(state.get('plan_notes'))}

        Write:
        1. A short title
        2. Three slide bullets
        3. One sentence explaining why iterative orchestration is useful
        """.strip()
    )

    return {"final_output": str(response.content)}


def build_graph():
    graph = StateGraph(MagenticState)

    graph.add_node("coordinator", coordinator)
    graph.add_node("researcher", researcher)
    graph.add_node("research_tools", research_tools)
    graph.add_node("research_summary", research_summary)
    graph.add_node("planner", planner)
    graph.add_node("finish", finish)

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "research": "researcher",
            "plan": "planner",
            "finish": "finish",
        },
    )

    graph.add_edge("researcher", "research_tools")
    graph.add_edge("research_tools", "research_summary")
    graph.add_edge("research_summary", "coordinator")
    graph.add_edge("planner", "coordinator")
    graph.add_edge("finish", END)

    return graph.compile()


if __name__ == "__main__":
    print("🚀 Magentic-style demo")
    print("Path: coordinator -> research or plan -> loop -> finish")

    app = build_graph()
    result = app.invoke({"topic": TOPIC})

    print("\n=== FINAL OUTPUT ===")
    print(result["final_output"])
