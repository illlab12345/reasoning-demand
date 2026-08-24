"""Versioned prompt rendering."""

from __future__ import annotations


def render_prompt(template_text: str, question: str) -> str:
    """Render a prompt template with {question} replaced verbatim.

    Uses plain replace (not str.format) so LaTeX braces in questions are safe.
    """
    if "{question}" not in template_text:
        raise ValueError("prompt template must contain a {question} placeholder")
    return template_text.replace("{question}", question)

