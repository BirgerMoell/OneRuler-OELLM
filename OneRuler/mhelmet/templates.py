"""Prompt templates for the multilingual HELMET-style benchmark.

The first version intentionally keeps instructions in English by default while
placing the evidence/context in the target language.  This mirrors HELMET's
application-centric task mix and keeps generation reproducible for all
OpenEuroLLM languages until native prompt translations are available.
"""

from __future__ import annotations


TASK_TEMPLATES = {
    "recall": (
        "Read the text below and remember the identifier facts.\n\n"
        "<text>\n{context}\n</text>\n\n"
        "<Question> What code is assigned to {query}? </Question>\n"
        "Answer with only the code."
    ),
    "rag": (
        "Use the retrieved passages to answer the question.\n\n"
        "{context}\n\n"
        "<Question> Which city hosted the {query} meeting? </Question>\n"
        "Answer with only the city name."
    ),
    "rerank": (
        "Rank the candidate passages by relevance to the question.\n\n"
        "<Question> Which passage identifies the archive key for {query}? </Question>\n\n"
        "{context}\n\n"
        "Return only the passage labels in best-to-worst order, separated by commas."
    ),
    "cite": (
        "Answer the question using the passages and cite the supporting passage label.\n\n"
        "{context}\n\n"
        "<Question> What is the registry value for {query}? </Question>\n"
        "Return the value followed by the citation label in brackets."
    ),
    "longqa": (
        "Read the long document and answer the question.\n\n"
        "<document>\n{context}\n</document>\n\n"
        "<Question> What verification phrase is linked to {query}? </Question>\n"
        "Answer with only the phrase."
    ),
    "summ": (
        "Summarize the document in exactly three bullet points. Preserve the key names, "
        "numbers, and outcomes.\n\n"
        "<document>\n{context}\n</document>"
    ),
    "icl": (
        "Infer the mapping from the examples, then answer the final item.\n\n"
        "{context}\n\n"
        "Final item: {query}\nAnswer with only the mapped label."
    ),
}


def task_template(task: str) -> str:
    try:
        return TASK_TEMPLATES[task]
    except KeyError as exc:
        raise ValueError(f"Unknown mHELMET task: {task}") from exc

