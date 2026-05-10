# LangGraph Demo

This project contains three very small LangGraph examples plus one shared tools file:

- `sequential_agent.py`
- `parallel_agent.py`
- `magentic_agent.py`
- `langgraph_tools.py`

The goal is to keep the code easy to read so you can quickly show:

- how a graph is constructed
- how tools are shared across agents
- how a graph is run
- how agents automatically request tools through the graph
- how simple the LangGraph building blocks can be

## Prerequisites

- Python 3.10+
- An Azure OpenAI deployment for chat completions

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Update `.env` with your Azure OpenAI values:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name
```

The scripts use `load_dotenv()` and will automatically read values from `.env`.

## Run The Demos

Run the sequential graph:

```bash
python3 sequential_agent.py
```

Run the parallel graph:

```bash
python3 parallel_agent.py
```

Run the magentic-style graph:

```bash
python3 magentic_agent.py
```

Each script uses the same sample topic:

`LangGraph demo for business teams`

If you want a different topic, edit the value in the `app.invoke(...)` call at the bottom of each file.

## What Each File Shows

- `sequential_agent.py`: one agent asks for tools, one `ToolNode` runs them, and one writer finishes the answer
- `parallel_agent.py`: three agents run side by side, each one uses its own tool, then one final node combines the results
- `magentic_agent.py`: a coordinator uses structured output to choose `research`, `plan`, or `finish`, then loops until it decides the work is complete
- `langgraph_tools.py`: shared local tools used by the graphs

## Notes

- The shared tools are intentionally simple local functions so the graph structure stays easy to follow.
- The Azure model setup is repeated in each agent file on purpose so the file can be read from top to bottom without jumping around.
- The scripts are intentionally written like tutorials, with only a few nodes and minimal state.
- Tool execution happens through LangGraph tool-calling flow, not direct Python tool calls inside the agent nodes.
- The magentic demo also shows `with_structured_output(...)` for coordinator routing decisions.
