from langchain_core.tools import tool


def log_tool(tool_name: str, message: str) -> None:
    print(f"[tool:{tool_name}] {message}", flush=True)


@tool
def get_topic_outline(topic: str) -> list[str]:
    """Return a short presentation outline for a topic."""
    log_tool("get_topic_outline", f"Starting with topic: {topic}")
    result = [
        f"What {topic} is",
        f"Why {topic} matters",
        f"How {topic} works in a simple workflow",
        f"One business use case for {topic}",
    ]
    log_tool("get_topic_outline", f"Returning outline: {result}")
    return result


@tool
def get_demo_examples(topic: str) -> list[str]:
    """Return a few easy demo examples for a topic."""
    log_tool("get_demo_examples", f"Starting with topic: {topic}")
    result = [
        f"{topic} for customer support triage",
        f"{topic} for internal research requests",
        f"{topic} for meeting prep and summaries",
    ]
    log_tool("get_demo_examples", f"Returning examples: {result}")
    return result


@tool
def get_audience_questions(topic: str) -> list[str]:
    """Return common audience questions for a topic."""
    log_tool("get_audience_questions", f"Starting with topic: {topic}")
    result = [
        f"What problem does {topic} solve?",
        f"How quickly can we pilot {topic}?",
        f"What tools or data does {topic} need?",
    ]
    log_tool("get_audience_questions", f"Returning questions: {result}")
    return result


@tool
def format_slide_bullets(items: list[str]) -> str:
    """Format text items into simple slide bullets."""
    log_tool("format_slide_bullets", f"Formatting items: {items}")
    result = "\n".join(f"- {item}" for item in items)
    log_tool("format_slide_bullets", f"Returning formatted bullets:\n{result}")
    return result
