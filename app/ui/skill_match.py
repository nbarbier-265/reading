"""
Skill Match UI components for the Streamlit application.

This module provides UI rendering functions for the skill match page.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils import create_skill_match_from_series, create_skill_match_feedback
from data import save_skill_match_feedback, get_story_details
from ui.common import render_story_header
from models import FeedbackRating


def render_skill_match_item(
    *,
    match_row: pd.Series,
    story_title: str,
) -> None:
    """
    Renders a single skill match item with feedback form.

    Args:
        match_row: Series containing skill match data.
        story_title: Title of the story.

    This function has UI side effects.
    """
    match = create_skill_match_from_series(row=match_row)

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Skill ID:** {match.skill_id}")
            st.markdown(f"**Parent Score:** {match.parent_score:.4f}")
            if match.parent_explanation:
                st.markdown(
                    f"**Parent Explanation:** <div style='word-wrap: break-word; white-space: normal;'>{match.parent_explanation}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Child Score:** {match.child_score:.4f}")
            if match.child_explanation:
                st.markdown(
                    f"**Child Explanation:** <div style='word-wrap: break-word; white-space: normal;'>{match.child_explanation}</div>",
                    unsafe_allow_html=True,
                )

            if match.method:
                st.markdown(f"**Method:** {match.method}")

        with col2:
            match_key = f"{match.story_id}_{match.skill_id}"

            with st.form(key=f"skill_feedback_form_{match_key}"):
                accuracy = st.radio(
                    "Is this skill match accurate?",
                    options=["", "👍 Accurate", "👎 Inaccurate", "🤔 Unsure"],
                    index=0,
                    key=f"accuracy_{match_key}",
                )

                feedback_comment = st.text_area(
                    "Comments (optional):", key=f"skill_comment_{match_key}"
                )

                submit_button = st.form_submit_button("Submit Feedback")

                if submit_button:
                    if accuracy:
                        feedback = create_skill_match_feedback(
                            match=match,
                            story_title=story_title,
                            accuracy=accuracy,
                            feedback_comment=feedback_comment,
                        )
                        save_skill_match_feedback(feedback=feedback)
                        st.success("Feedback saved!")
                    else:
                        st.warning("Please select an accuracy rating.")

    st.markdown("---")


def render_skill_match_page(
    *,
    stories_df: pd.DataFrame,
    skill_match_df: pd.DataFrame,
    selected_story_id: str,
    prompt_version: str = "v1",
) -> None:
    """
    Renders the skill match feedback page.

    Args:
        stories_df: DataFrame containing story data
        skill_match_df: DataFrame containing skill match data
        selected_story_id: ID of the selected story
        prompt_version: The prompt version to use. Defaults to "v1".

    This function has UI side effects.
    """
    story_text, story_title = get_story_details(
        stories_df=stories_df, story_id=selected_story_id
    )

    st.header(f"Skill Match Feedback: {story_title}")

    # Handle None skill_match_df case
    if skill_match_df is None:
        st.info("No skill matches found for this story.")
        return

    # Filter skill matches for the selected story
    story_skill_matches = skill_match_df[
        skill_match_df["story_id"] == selected_story_id
    ]

    # Display a card for each skill match
    if not story_skill_matches.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Story Text")
            st.write(story_text)

        with col2:
            st.markdown("### Matched Skills")

            for _, skill_match in story_skill_matches.iterrows():
                with st.expander(
                    f"{skill_match['skill_id']} (Score: {skill_match['child_score']:.2f})",
                    expanded=False,
                ):
                    st.markdown(f"**Skill ID:** {skill_match['skill_id']}")

                    # Use get() with a default value to handle missing 'explanation' field
                    explanation = skill_match.get(
                        "explanation",
                        skill_match.get(
                            "child_explanation", "No explanation available"
                        ),
                    )
                    st.markdown(f"**Explanation:** {explanation}")

                    # Add feedback form
                    st.markdown("#### Provide Feedback on this Skill Match")

                    rating = st.radio(
                        "How accurate is this match?",
                        options=[
                            "Very helpful",
                            "Somewhat helpful",
                            "Not helpful",
                        ],
                        index=0,
                        key=f"rating_{skill_match['skill_id']}",
                        horizontal=True,
                    )

                    comments = st.text_area(
                        "Additional Comments (Optional)",
                        key=f"comments_{skill_match['skill_id']}",
                    )

                    if st.button(
                        "Submit Feedback", key=f"submit_{skill_match['skill_id']}"
                    ):
                        # Get parent_score and child_score from the DataFrame
                        # Use child_score if parent_score is not available
                        parent_score = skill_match.get(
                            "parent_score", skill_match.get("child_score", 0.0)
                        )
                        child_score = skill_match.get("child_score", 0.0)

                        feedback = SkillMatchFeedback(
                            story_id=selected_story_id,
                            story_title=story_title,
                            skill_id=skill_match["skill_id"],
                            parent_score=parent_score,
                            child_score=child_score,
                            accuracy=rating,  # Map rating to accuracy field
                            comment=comments,  # Map comments to comment field
                            method=skill_match.get("method", None),
                            timestamp=datetime.now().isoformat(),
                        )

                        save_skill_match_feedback(
                            feedback=feedback, prompt_version=prompt_version
                        )
                        st.success("Feedback saved!")
    else:
        st.info("No skill matches found for this story.")
