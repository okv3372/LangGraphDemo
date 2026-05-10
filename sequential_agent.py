"""
Sequential agent:
1. The model decides which tools to call.
2. LangGraph runs those tools.
3. The model writes the final answer.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langgraph_tools import get_demo_examples, get_topic_outline

load_dotenv()

TOOLS = [get_topic_outline, get_demo_examples]
MESSAGES = Annotated[list, add_messages]


class SequentialState(TypedDict, total=False):
    topic: str
    messages: MESSAGES
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
    )


def collect_tool_outputs(messages: list) -> str:
    outputs = []
    for message in messages:
        if message.__class__.__name__ == "ToolMessage":
            outputs.append(f"{message.name}: {message.content}")
    return "\n".join(outputs)


# Node 1: Ask the model which tools it wants to use and with what inputs.
def planner(state: SequentialState) -> SequentialState:
    print("1️⃣ 🤖 Planner")
    model = build_model().bind_tools(TOOLS, tool_choice="required")

    steering_prompt = f"You should always call the tool with the string 'For a team of engineers: {state['topic']}' as input."
    response = model.invoke(
        f"""
        Topic: {state['topic']}
        
        Call both tools so we can build a short presentation brief.
        Do not answer the user yet.
        """.strip()
    )

    print("Requested tools:", [call["name"] for call in response.tool_calls])
    print("2️⃣ Tool node")
    return {"messages": [response]}


# Node 2: LangGraph ToolNode runs the requested tools automatically.
tools = ToolNode(TOOLS)


# Node 3: use the tool results to write the final answer.
def writer(state: SequentialState) -> SequentialState:
    print("3️⃣ 📝 Writer")
    tool_outputs = collect_tool_outputs(state["messages"])
    print("Tool outputs:")
    print(tool_outputs)

    model = build_model()
    response = model.invoke(
        f"""
        Topic: {state['topic']}
        Tool outputs:
        {tool_outputs}

        Write:
        1. A short title
        2. Three slide bullets
        3. One sentence on why this demo matters
        """.strip()
    )

    return {"final_output": str(response.content)}


def build_graph():
    graph = StateGraph(SequentialState)

    graph.add_node("planner", planner)
    graph.add_node("tools", tools)
    graph.add_node("writer", writer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tools")
    graph.add_edge("tools", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


if __name__ == "__main__":
    print("🚀 Sequential demo")
    print("Path: 1️⃣ planner -> 2️⃣ tools -> 3️⃣ writer")

    app = build_graph()
    topic = "LangGraph demo"
    result = app.invoke({"topic": topic})

    print("\n=== FINAL OUTPUT ===")
    print(result["final_output"])
