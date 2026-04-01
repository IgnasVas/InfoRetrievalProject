"""
Query type classification for query specific evaluation.
"""


def classify_query_type(query: str) -> str:
    """
    Assign a simple query type based on the question text.

    Args:
        query: Raw query string.

    Returns:
        Query type label.
    """
    q = query.lower().strip()

    if q.startswith("why"):
        return "why"

    if q.startswith("how"):
        return "how"

    if q.startswith("what"):
        return "what"

    if q.startswith("when"):
        return "when"

    if q.startswith("where"):
        return "where"

    if q.startswith("who"):
        return "who"

    if any(term in q for term in [" vs ", "versus", "difference", "better", "best", "compare"]):
        return "comparison"

    if any(term in q for term in ["should i", "recommend", "advice", "suggest", "help me"]):
        return "advice"

    return "other"
