from langchain_core.tools import tool


@tool
def get_topic_outline(topic: str) -> list[str]:
    """Return a short presentation outline for a topic."""
    return [
        f"What {topic} is",
        f"Why {topic} matters",
        f"How {topic} works in a simple workflow",
        f"One business use case for {topic}",
    ]


@tool
def get_demo_examples(topic: str) -> list[str]:
    """Return a few easy demo examples for a topic."""
    return [
        f"{topic} for customer support triage",
        f"{topic} for internal research requests",
        f"{topic} for meeting prep and summaries",
    ]


@tool
def get_audience_questions(topic: str) -> list[str]:
    """Return common audience questions for a topic."""
    return [
        f"What problem does {topic} solve?",
        f"How quickly can we pilot {topic}?",
        f"What tools or data does {topic} need?",
    ]


@tool
def format_slide_bullets(items: list[str]) -> str:
    """Format text items into simple slide bullets."""
    return "\n".join(f"- {item}" for item in items)
