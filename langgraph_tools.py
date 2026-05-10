from langchain_core.tools import tool


@tool
def get_topic_outline(topic: str) -> list[str]:
    """Return a short presentation outline for a topic."""
    # """Return a short presentation outline for a topic.
    # The topic input should always be geared towards a team of engineers, 
    # for example with the topic LangGraph: 'For a team of engineers: LangGraph."""
    print(f"🔧 [tool:get_topic_outline] Starting with topic: {topic}")
    result = [
        f"What {topic} is",
        f"Why {topic} matters",
        f"How {topic} works in a simple workflow",
        f"One business use case for {topic}",
    ]
    print(f"🔧 [tool:get_topic_outline] Returning outline: {result}")
    return result


@tool
def get_demo_examples(topic: str) -> list[str]:
    """Return a few easy demo examples for a topic."""
    print(f"🔧 [tool:get_demo_examples] Starting with topic: {topic}")
    result = [
        f"{topic} for customer support triage",
        f"{topic} for internal research requests",
        f"{topic} for meeting prep and summaries",
    ]
    print(f"🔧 [tool:get_demo_examples] Returning examples: {result}")
    return result


@tool
def get_audience_questions(topic: str) -> list[str]:
    """Return common audience questions for a topic."""
    print(f"🔧 [tool:get_audience_questions] Starting with topic: {topic}")
    result = [
        f"What problem does {topic} solve?",
        f"How quickly can we pilot {topic}?",
        f"What tools or data does {topic} need?",
    ]
    print(f"🔧 [tool:get_audience_questions] Returning questions: {result}")
    return result


@tool
def format_slide_bullets(items: list[str]) -> str:
    """Format text items into simple slide bullets."""
    print(f"🔧 [tool:format_slide_bullets] Formatting items: {items}")
    result = "\n".join(f"- {item}" for item in items)
    print(f"🔧 [tool:format_slide_bullets] Returning formatted bullets:\n{result}")
    return result
