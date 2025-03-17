"""
Utility functions for the Streamlit application.

This module provides utility functions for text processing and UI helpers.
"""

import re
import pandas as pd
from models import Intervention, InterventionFeedback, SkillMatch, SkillMatchFeedback


def split_text_into_sentences(*, text: str) -> list[str]:
    """
    Splits text into sentences.

    Args:
        text: Text to split.

    Returns:
        List of sentences.
    """
    return re.split(r"(?<=[.!?])\s+", text)


def create_sentence_html(
    *, sentence: str, is_highlighted: bool, color: str = ""
) -> str:
    """
    Creates HTML for a sentence with optional highlighting.

    Args:
        sentence: The sentence text to display.
        is_highlighted: Whether the sentence should be highlighted.
        color: Background color for highlighting.

    Returns:
        HTML string for the formatted sentence.
    """
    base_style = "display:block; margin-bottom:10px; padding:5px;"

    if is_highlighted:
        return f"<span style='background-color:{color}; color:black; {base_style}'>{sentence}</span>"

    return f"<span style='{base_style}'>{sentence}</span>"


def create_intervention_from_series(*, row: pd.Series) -> Intervention:
    """
    Creates an Intervention object from a pandas Series.

    Args:
        row: Series containing intervention data.

    Returns:
        Intervention object.
    """
    return Intervention(
        story_id=row["story_id"],
        skill_id=row["skill_id"],
        sentence=row["sentence"],
        intervention_type=row["intervention_type"],
        intervention=row["intervention"],
        explanation=row["explanation"],
        score=row["score"],
    )


def create_intervention_feedback(
    *,
    intervention: Intervention,
    intervention_key: str,
    story_title: str,
    helpfulness: str,
    feedback_comment: str,
    fit_agreement: str,
) -> InterventionFeedback:
    """
    Creates an InterventionFeedback object.

    Args:
        intervention: Intervention object.
        intervention_key: Unique key for the intervention.
        story_title: Title of the story.
        helpfulness: User's helpfulness rating.
        feedback_comment: User's feedback comment.
        fit_agreement: User's agreement on intervention fit.

    Returns:
        InterventionFeedback object.
    """
    return InterventionFeedback(
        story_id=intervention.story_id,
        story_title=story_title,
        intervention_id=intervention_key,
        sentence=intervention.sentence,
        skill_id=intervention.skill_id,
        intervention_type=intervention.intervention_type,
        intervention=intervention.intervention,
        feedback="positive" if helpfulness == "👍 Helpful" else "negative",
        comment=feedback_comment,
        fit_agreement=fit_agreement if fit_agreement else "Not specified",
    )


def create_skill_match_from_series(*, row: pd.Series) -> SkillMatch:
    """
    Creates a SkillMatch object from a pandas Series.

    Args:
        row: Series containing skill match data.

    Returns:
        SkillMatch object.
    """
    skill_match = SkillMatch(
        story_id=row["story_id"],
        skill_id=row["skill_id"],
        parent_score=float(row["parent_score"]),
        child_score=float(row["child_score"]),
    )

    if "method" in row:
        skill_match.method = row["method"]

    if "parent_explanation" in row:
        skill_match.parent_explanation = row["parent_explanation"]

    if "child_explanation" in row:
        skill_match.child_explanation = row["child_explanation"]

    return skill_match


def create_skill_match_feedback(
    *,
    match: SkillMatch,
    story_title: str,
    accuracy: str,
    feedback_comment: str,
) -> SkillMatchFeedback:
    """
    Creates a SkillMatchFeedback object.

    Args:
        match: SkillMatch object.
        story_title: Title of the story.
        accuracy: User's accuracy rating.
        feedback_comment: User's feedback comment.

    Returns:
        SkillMatchFeedback object.
    """
    return SkillMatchFeedback(
        story_id=match.story_id,
        story_title=story_title,
        skill_id=match.skill_id,
        parent_score=match.parent_score,
        child_score=match.child_score,
        method=match.method,
        accuracy=accuracy,
        comment=feedback_comment,
    )
