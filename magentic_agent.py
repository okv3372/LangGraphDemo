"""
Magentic-style demo:
1. A coordinator decides the next step.
2. It can send work to a researcher or a planner.
3. The researcher uses tools.
4. The graph loops until the coordinator decides to finish.
"""

import os
from operator import add
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

TOPIC = "LangGraph's use for building agents"
MAX_ROUNDS = 6
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


class CoordinatorDecision(TypedDict):
    step: Literal["research", "plan", "finish"]
    note: str


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


def coordinator(state: MagenticState) -> MagenticState:
    round_number = state.get("round_count", 0) + 1
    print(f"🎯 Coordinator (round {round_number})")

    if round_number > MAX_ROUNDS:
        print("Max rounds reached. Finishing with current notes.")
        return {
            "round_count": round_number,
            "next_step": "finish",
            "coordinator_note": "We have enough information. Finish the answer.",
        }

    model = build_model().with_structured_output(CoordinatorDecision)
    decision = model.invoke(
        f"""
        You are coordinating a small research workflow.

        Topic: {state['topic']}
        Round: {round_number} of {MAX_ROUNDS}
        Previous step: {state.get('next_step', 'Start')}

        Research notes so far:
        {format_notes(state.get('research_notes'))}

        Plan notes so far:
        {format_notes(state.get('plan_notes'))}

        Always start with a planning round.
        Each round of research should be followed by a round of planning to organize the findings.
        You should always do at least two rounds of research before finishing.
        Never do more than one round of planning in a row.

        Choose the next step:
        - research: gather more information
        - plan: organize the information and identify gaps
        - finish: stop when the work is complete
        """.strip()
    )

    print("Structured decision:", decision["step"])

    return {
        "round_count": round_number,
        "next_step": decision["step"],
        "coordinator_note": decision["note"],
    }


def route_from_coordinator(state: MagenticState) -> str:
    return state["next_step"]


def researcher(state: MagenticState) -> MagenticState:
    print("🔎 Researcher")
    model = build_model()

    response = model.invoke(
        f"""
        You are a helpful researcher. Your goal is to provide information about the given topic.

        Topic: {state['topic']}
        Coordinator instruction: {state['coordinator_note']}

        Research notes so far:
        {format_notes(state.get('research_notes'))}
        """.strip()
    )

    print("Researcher response:")
    print(response.content)

    return {"research_messages": [response]}

def research_summary(state: MagenticState) -> MagenticState:
    print("📚 Research summary")

    model = build_model()
    response = model.invoke(
        f"""
        You are a research summarizer. Your goal is to read through the most recent research messages and summarize the most useful findings in a concise way.
        Write 2 short bullets with the most useful new findings given the following topic and coordinator instructions:

        Topic: {state['topic']}
        Coordinator instruction: {state['coordinator_note']}
        Research messages: {state['research_messages']}
        """.strip()
    )

    print("Research summarization:")
    print(response.content)
    return {"research_notes": [response.content]}


def planner(state: MagenticState) -> MagenticState:
    print("🗂️ Planner")
    model = build_model()

    response = model.invoke(
        f"""
        Topic: {state['topic']}
        Coordinator instruction: {state['coordinator_note']}

        Research notes so far:
        {format_notes(state.get('research_notes'))}

        Give a list of 1-2 more things to research based on the research notes, and any planning thoughts you have about how to organize the information or identify gaps.
        Make your response short and sweet - just a few bullets. Do not write the final answer.
        """.strip()
    )

    note = str(response.content)
    print("Plan note:")
    print(note)
    return {"plan_notes": [note]}


def finish(state: MagenticState) -> MagenticState:
    print("✅ Finish")
    model = build_model()

    response = model.invoke(
        f"""
        Topic: {state['topic']}

        Research notes:
        {format_notes(state.get('research_notes'))}

        Plan notes:
        {format_notes(state.get('plan_notes'))}

        Write a research report about:
        1. A final summary of the findings based on the research and plan notes.
        2. A reflection on the research process and how the notes evolved over time.
        3. Any remaining gaps or questions that could be explored in the future.
        """.strip()
    )

    return {"final_output": str(response.content)}


def build_graph():
    graph = StateGraph(MagenticState)

    graph.add_node("coordinator", coordinator)
    graph.add_node("researcher", researcher)
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

    graph.add_edge("researcher", "research_summary")
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
