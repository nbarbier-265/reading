"""
Story Explorer UI components for the Streamlit application.

This module provides UI rendering functions for the story explorer page.
"""

import streamlit as st
from datetime import datetime
import pandas as pd
from typing import Optional, List, Dict, Tuple

from utils import (
    split_text_into_sentences,
    create_sentence_html,
    create_intervention_from_series,
    create_intervention_feedback,
)
from data import save_feedback, get_story_details
from models import Intervention, InterventionFeedback, FeedbackRating
from ui.common import render_story_header, render_fallback_story, render_app_header


def render_story_only(
    *,
    stories_df: pd.DataFrame,
    selected_story_id: str,
) -> None:
    """
    Renders just the story text without intervention points.
    Used when intervention data is not available.

    Args:
        stories_df: DataFrame containing story data.
        selected_story_id: ID of the currently selected story.

    This function has UI side effects.
    """
    render_app_header()

    story_text, story_title = get_story_details(
        stories_df=stories_df, story_id=selected_story_id
    )

    render_story_header(story_title=story_title, story_id=selected_story_id)

    st.markdown("### Story Text")

    # Display story sentences without highlighting
    sentences = split_text_into_sentences(text=story_text)
    for sentence in sentences:
        st.markdown(
            create_sentence_html(sentence=sentence, is_highlighted=False),
            unsafe_allow_html=True,
        )

    st.info(
        "No intervention points data available. You can generate intervention points in the Process Stories tab."
    )


def render_story_text(
    *,
    sentences: list[str],
    filtered_interventions: pd.DataFrame,
    intervention_colors: list[str],
) -> None:
    """
    Renders story text with highlighted interventions.

    Args:
        sentences: List of sentences from the story.
        filtered_interventions: DataFrame of filtered interventions.
        intervention_colors: List of colors for highlighting.

    This function has UI side effects.
    """
    st.markdown("### Story Text")

    for sentence in sentences:
        highlighted = False

        for idx, intervention in filtered_interventions.iterrows():
            if (
                intervention["sentence"] in sentence
                or sentence in intervention["sentence"]
            ):
                color_idx = idx % len(intervention_colors)
                color = intervention_colors[color_idx]
                st.markdown(
                    create_sentence_html(
                        sentence=sentence, is_highlighted=True, color=color
                    ),
                    unsafe_allow_html=True,
                )
                highlighted = True
                break

        if not highlighted:
            st.markdown(
                create_sentence_html(sentence=sentence, is_highlighted=False),
                unsafe_allow_html=True,
            )


def render_intervention_details(*, intervention: pd.Series) -> None:
    """
    Renders the details of an intervention.

    Args:
        intervention: Series containing intervention data.

    This function has UI side effects.
    """
    st.markdown(f"**Text Snippet:**")
    st.markdown(f"_{intervention['sentence']}_")
    st.markdown(f"**Skill ID:** {intervention['skill_id']}")
    st.markdown(f"**Intervention Type:** {intervention['intervention_type']}")
    st.markdown(f"**Intervention:** {intervention['intervention']}")

    explanation_text = intervention["explanation"]
    st.markdown(
        f"**Explanation:** <div style='word-wrap: break-word; white-space: normal;'>{explanation_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"**Score:** {intervention['score']:.2f}")


def handle_feedback_submission(
    *,
    helpfulness: str,
    intervention: Intervention,
    intervention_key: str,
    story_title: str,
    feedback_comment: str,
    fit_agreement: str,
) -> None:
    """
    Handles feedback submission for an intervention.

    Args:
        helpfulness: User's helpfulness rating.
        intervention: Intervention object.
        intervention_key: Unique key for the intervention.
        story_title: Title of the story.
        feedback_comment: User's feedback comment.
        fit_agreement: User's agreement on intervention fit.

    This function has UI and data persistence side effects.
    """
    if helpfulness:
        feedback = create_intervention_feedback(
            intervention=intervention,
            intervention_key=intervention_key,
            story_title=story_title,
            helpfulness=helpfulness,
            feedback_comment=feedback_comment,
            fit_agreement=fit_agreement,
        )
        save_feedback(feedback=feedback)
        st.success("Feedback saved!")
    else:
        st.warning("Please select whether the intervention was helpful.")


def render_feedback_controls(
    *,
    intervention: Intervention,
    intervention_key: str,
    story_title: str,
    context_id: str = "",  # Add a context identifier to make keys unique
) -> None:
    """
    Renders feedback controls for an intervention.

    Args:
        intervention: Intervention object.
        intervention_key: Unique key for the intervention.
        story_title: Title of the story.
        context_id: Additional context identifier to ensure unique form keys.

    This function has UI side effects.
    """
    st.markdown("**Feedback:**")

    # Create a unique form key that includes the context identifier
    form_key = f"feedback_form_{intervention_key}"
    if context_id:
        form_key = f"feedback_form_{context_id}_{intervention_key}"

    with st.form(key=form_key):
        helpfulness = st.radio(
            "Was this intervention helpful?",
            options=["", "👍 Helpful", "👎 Not Helpful"],
            index=0,
            key=f"helpfulness_{context_id}_{intervention_key}",
        )

        feedback_comment = st.text_area(
            "Comments (optional):", key=f"comment_{context_id}_{intervention_key}"
        )

        fit_agreement = st.radio(
            "Does this intervention fit well with the text?",
            options=["", "Yes", "No", "Unsure"],
            index=0,
            horizontal=True,
            key=f"fit_{context_id}_{intervention_key}",
        )

        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            handle_feedback_submission(
                helpfulness=helpfulness,
                intervention=intervention,
                intervention_key=intervention_key,
                story_title=story_title,
                feedback_comment=feedback_comment,
                fit_agreement=fit_agreement,
            )


def render_intervention_card(
    *,
    intervention_row: pd.Series,
    idx: int,
    color: str,
    story_title: str,
    context_id: str = "",  # Add a context identifier to make keys unique
) -> None:
    """
    Renders an intervention card with feedback controls.

    Args:
        intervention_row: Series containing intervention data.
        idx: Index of the intervention.
        color: Background color for the card.
        story_title: Title of the story.
        context_id: Additional context identifier to ensure unique form keys.

    This function has UI side effects.
    """
    intervention = create_intervention_from_series(row=intervention_row)
    intervention_key = intervention.create_key(idx=idx)

    st.markdown(
        f"<div style='background-color:{color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>",
        unsafe_allow_html=True,
    )

    render_intervention_details(intervention=intervention_row)

    render_feedback_controls(
        intervention=intervention,
        intervention_key=intervention_key,
        story_title=story_title,
        context_id=context_id,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_interventions_section(
    *,
    filtered_interventions: pd.DataFrame,
    intervention_colors: list[str],
    story_title: str,
    context_id: str = "",  # Add a context identifier to make keys unique
) -> None:
    """
    Renders the interventions section.

    Args:
        filtered_interventions: DataFrame of filtered interventions.
        intervention_colors: List of colors for highlighting.
        story_title: Title of the story.
        context_id: Additional context identifier to ensure unique form keys.

    This function has UI side effects.
    """
    st.markdown("### Intervention Points")

    if not filtered_interventions.empty:
        for idx, intervention in filtered_interventions.iterrows():
            color_idx = idx % len(intervention_colors)
            color = intervention_colors[color_idx]

            with st.container():
                render_intervention_card(
                    intervention_row=intervention,
                    idx=idx,
                    color=color,
                    story_title=story_title,
                    context_id=context_id,
                )
    else:
        st.info("No intervention points match your filter criteria.")


def render_story_container(
    *,
    story_text: str,
    filtered_interventions: pd.DataFrame,
    story_title: str,
    context_id: str = "container",
) -> None:
    """
    Renders the main story container with text and interventions.

    Args:
        story_text: Text of the story.
        filtered_interventions: DataFrame of filtered interventions.
        story_title: Title of the story.
        context_id: Additional context identifier to ensure unique form keys.

    This function has UI side effects.
    """
    intervention_colors = ["#FFD580", "#FFAFAF", "#AFFFB7", "#AFAFFF", "#FFFFAF"]
    sentences = split_text_into_sentences(text=story_text)

    story_container = st.container()

    with story_container:
        col1, col2 = st.columns([3, 2])

        with col1:
            render_story_text(
                sentences=sentences,
                filtered_interventions=filtered_interventions,
                intervention_colors=intervention_colors,
            )

        with col2:
            render_interventions_section(
                filtered_interventions=filtered_interventions,
                intervention_colors=intervention_colors,
                story_title=story_title,
                context_id=context_id,
            )


def render_story_explorer(
    *,
    stories_df: pd.DataFrame,
    intervention_points_df: pd.DataFrame,
    selected_story_id: str,
    prompt_version: str = "v1",
) -> None:
    """
    Renders the story explorer page.

    Args:
        stories_df: DataFrame containing story data
        intervention_points_df: DataFrame containing intervention points data
        selected_story_id: ID of the selected story
        prompt_version: The prompt version to use. Defaults to "v1".

    This function has UI side effects.
    """
    story_text, story_title = get_story_details(
        stories_df=stories_df, story_id=selected_story_id
    )

    st.header(f"Story: {story_title}")

    # Filter interventions for the selected story
    story_interventions = intervention_points_df[
        intervention_points_df["story_id"] == selected_story_id
    ]

    # Display the story with interventions
    if not story_interventions.empty:
        # Get unique skills in the interventions
        unique_skills = story_interventions["skill_id"].unique()

        # Use expander for each skill to show its interventions with highlighted text
        st.subheader("Story Interventions by Skill")
        st.info(
            "Explore interventions for each skill. Click on a skill to see its specific interventions highlighted in the story text."
        )

        for skill_id in unique_skills:
            # Filter interventions for this skill
            skill_interventions = story_interventions[
                story_interventions["skill_id"] == skill_id
            ]

            with st.expander(f"Skill: {skill_id}", expanded=False):
                # Create a container for this skill's view
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown("### Story Text")
                    sentences = split_text_into_sentences(text=story_text)
                    intervention_colors = [
                        "#FFD580",
                        "#FFAFAF",
                        "#AFFFB7",
                        "#AFAFFF",
                        "#FFFFAF",
                    ]

                    # Display each sentence, highlighting those that are intervention points for this skill
                    for sentence in sentences:
                        highlighted = False

                        for idx, intervention in skill_interventions.iterrows():
                            # Check if the sentence is part of an intervention
                            if (
                                intervention["sentence"] in sentence
                                or sentence in intervention["sentence"]
                            ):
                                color_idx = idx % len(intervention_colors)
                                color = intervention_colors[color_idx]

                                st.markdown(
                                    create_sentence_html(
                                        sentence=sentence,
                                        is_highlighted=True,
                                        color=color,
                                    ),
                                    unsafe_allow_html=True,
                                )
                                highlighted = True
                                break

                        if not highlighted:
                            st.markdown(
                                create_sentence_html(
                                    sentence=sentence, is_highlighted=False
                                ),
                                unsafe_allow_html=True,
                            )

                with col2:
                    st.markdown("### Intervention Points")

                    if not skill_interventions.empty:
                        for idx, intervention in skill_interventions.iterrows():
                            color_idx = idx % len(intervention_colors)
                            color = intervention_colors[color_idx]

                            with st.container():
                                render_intervention_card(
                                    intervention_row=intervention,
                                    idx=idx,
                                    color=color,
                                    story_title=story_title,
                                    context_id=f"skill_{skill_id}",
                                )
                    else:
                        st.info(f"No intervention points found for skill {skill_id}.")
    else:
        st.info("No intervention points found for this story.")
        render_story_only(stories_df=stories_df, selected_story_id=selected_story_id)
