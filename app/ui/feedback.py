"""
Feedback Analytics UI components for the Streamlit application.

This module provides UI rendering functions for the feedback analytics page.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import os
import re
from typing import Dict, List, Any

from config import get_processed_data_path


def clean_rating_field(df, field_name):
    """
    Clean up the rating field to handle various formats.

    Args:
        df: DataFrame containing the feedback data
        field_name: Name of the rating field to clean

    Returns:
        DataFrame with cleaned rating field
    """
    if field_name not in df.columns:
        return df

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Function to clean individual rating values
    def clean_value(val):
        if not isinstance(val, str):
            return val

        # Remove duplicates by splitting on emoji or common words
        if "👍" in val or "👎" in val or "🤔" in val:
            # Take just the first emoji rating
            match = re.search(r"(👍|👎|🤔)[\s]?[A-Za-z]+", val)
            if match:
                return match.group(0)

        # For text ratings, take the first word
        if val.lower().startswith(("positive", "negative", "neutral")):
            return val.lower().split()[0]

        # For "Very helpful" type ratings
        if "helpful" in val.lower():
            match = re.search(r"(Very|Somewhat|Not)[\s]?[Hh]elpful", val)
            if match:
                return match.group(0)

        return val

    # Apply the cleaning function
    df_copy[field_name] = df_copy[field_name].apply(clean_value)

    return df_copy


def render_feedback_analytics(prompt_version: str = "v1") -> None:
    """
    Renders the feedback analytics page.

    Args:
        prompt_version: The prompt version to use. Defaults to "v1".

    This function has UI side effects.
    """
    st.header("📈 Feedback Analytics")
    st.markdown("""
    This page shows analytics on the feedback received for skill matches and intervention points.
    """)

    # Get paths based on prompt version
    processed_data_path = get_processed_data_path(prompt_version)
    intervention_feedback_path = Path(
        f"{processed_data_path}/intervention_feedback.json"
    )
    skill_match_feedback_path = Path(f"{processed_data_path}/skill_match_feedback.json")

    # Check if feedback files exist
    has_intervention_feedback = intervention_feedback_path.exists()
    has_skill_match_feedback = skill_match_feedback_path.exists()

    if not has_intervention_feedback and not has_skill_match_feedback:
        st.info(
            f"No feedback data found for prompt version {prompt_version}. You can provide feedback on the Review Results page."
        )
        return

    # Create tabs for different types of feedback
    feedback_tabs = st.tabs(["Intervention Feedback", "Skill Match Feedback"])

    # Intervention Feedback Tab
    with feedback_tabs[0]:
        st.subheader("Intervention Point Feedback")

        if has_intervention_feedback:
            try:
                with open(intervention_feedback_path, "r") as f:
                    intervention_feedback = json.load(f)

                if intervention_feedback:
                    feedback_df = pd.DataFrame(intervention_feedback)

                    # Clean up the rating field
                    rating_field = (
                        "rating" if "rating" in feedback_df.columns else "feedback"
                    )
                    if rating_field in feedback_df.columns:
                        feedback_df = clean_rating_field(feedback_df, rating_field)

                    # Display overall metrics
                    st.markdown("### Overall Metrics")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Feedback Items", len(feedback_df))
                    with col2:
                        st.metric("Unique Stories", feedback_df["story_id"].nunique())
                    with col3:
                        # Check if either 'rating' or 'feedback' exists
                        rating_field = (
                            "rating" if "rating" in feedback_df.columns else "feedback"
                        )
                        if rating_field in feedback_df.columns:
                            # Instead of calculating mean, show most common rating
                            try:
                                most_common = (
                                    feedback_df[rating_field].value_counts().index[0]
                                )
                                st.metric("Most Common Rating", most_common)
                            except:
                                st.metric("Most Common Rating", "N/A")
                        else:
                            st.metric("Most Common Rating", "N/A")

                    # Display distribution of ratings
                    st.markdown("### Rating Distribution")
                    # Check if 'rating' exists, otherwise use 'feedback'
                    rating_field = (
                        "rating" if "rating" in feedback_df.columns else "feedback"
                    )
                    if rating_field in feedback_df.columns:
                        try:
                            rating_counts = (
                                feedback_df[rating_field].value_counts().reset_index()
                            )
                            rating_counts.columns = ["Rating", "Count"]
                            st.bar_chart(rating_counts.set_index("Rating"))
                        except Exception as e:
                            st.error(f"Could not create chart: {e}")
                            st.write(feedback_df[rating_field].value_counts())
                    else:
                        st.info("No rating data available")

                    # Display distribution by intervention type
                    if "intervention_type" in feedback_df.columns:
                        st.markdown("### Feedback by Intervention Type")

                        # Check if either 'rating' or 'feedback' exists
                        rating_field = (
                            "rating" if "rating" in feedback_df.columns else "feedback"
                        )
                        if rating_field in feedback_df.columns:
                            try:
                                # Instead of mean, count by intervention type
                                type_counts = (
                                    feedback_df.groupby("intervention_type")
                                    .size()
                                    .reset_index()
                                )
                                type_counts.columns = ["Intervention Type", "Count"]
                                st.bar_chart(type_counts.set_index("Intervention Type"))
                            except Exception as e:
                                st.error(
                                    f"Could not create intervention type chart: {e}"
                                )
                                st.write(
                                    feedback_df.groupby("intervention_type").size()
                                )
                        else:
                            st.info("No rating data available by intervention type")

                    # Display raw feedback data
                    st.markdown("### Raw Feedback Data")
                    st.dataframe(feedback_df)
                else:
                    st.info("No intervention feedback data available yet.")
            except Exception as e:
                st.error(f"Error loading intervention feedback data: {e}")
        else:
            st.info(
                "No intervention feedback data available yet. You can provide feedback on intervention points on the Story Explorer page."
            )

    # Skill Match Feedback Tab
    with feedback_tabs[1]:
        st.subheader("Skill Match Feedback")

        if has_skill_match_feedback:
            try:
                with open(skill_match_feedback_path, "r") as f:
                    skill_match_feedback = json.load(f)

                if skill_match_feedback:
                    feedback_df = pd.DataFrame(skill_match_feedback)

                    # Clean up the rating field
                    rating_field = (
                        "rating" if "rating" in feedback_df.columns else "accuracy"
                    )
                    if rating_field in feedback_df.columns:
                        feedback_df = clean_rating_field(feedback_df, rating_field)

                    # Display overall metrics
                    st.markdown("### Overall Metrics")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Feedback Items", len(feedback_df))
                    with col2:
                        st.metric("Unique Skills", feedback_df["skill_id"].nunique())
                    with col3:
                        # Check if 'rating' exists, otherwise use 'accuracy'
                        rating_field = (
                            "rating" if "rating" in feedback_df.columns else "accuracy"
                        )
                        if rating_field in feedback_df.columns:
                            # Instead of calculating mean, show most common rating
                            try:
                                most_common = (
                                    feedback_df[rating_field].value_counts().index[0]
                                )
                                st.metric("Most Common Rating", most_common)
                            except:
                                st.metric("Most Common Rating", "N/A")
                        else:
                            st.metric("Most Common Rating", "N/A")

                    # Display distribution of ratings
                    st.markdown("### Rating Distribution")
                    # Check if 'rating' exists, otherwise use 'accuracy'
                    rating_field = (
                        "rating" if "rating" in feedback_df.columns else "accuracy"
                    )
                    if rating_field in feedback_df.columns:
                        try:
                            rating_counts = (
                                feedback_df[rating_field].value_counts().reset_index()
                            )
                            rating_counts.columns = ["Rating", "Count"]
                            st.bar_chart(rating_counts.set_index("Rating"))
                        except Exception as e:
                            st.error(f"Could not create chart: {e}")
                            st.write(feedback_df[rating_field].value_counts())
                    else:
                        st.info("No rating data available")

                    # Display top skills by feedback
                    st.markdown("### Top Skills by Feedback Count")

                    skill_counts = feedback_df["skill_id"].value_counts().reset_index()
                    skill_counts.columns = ["Skill ID", "Count"]

                    st.bar_chart(skill_counts.head(10).set_index("Skill ID"))

                    # Display raw feedback data
                    st.markdown("### Raw Feedback Data")
                    st.dataframe(feedback_df)
                else:
                    st.info("No skill match feedback data available yet.")
            except Exception as e:
                st.error(f"Error loading skill match feedback data: {e}")
        else:
            st.info(
                "No skill match feedback data available yet. You can provide feedback on skill matches on the Skill Match Feedback page."
            )
