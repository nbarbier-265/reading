"""
Common UI components for the Streamlit application.

This module provides common UI rendering functions used across the application.
"""

import streamlit as st


def render_app_header() -> None:
    """
    Renders the application header and description.

    This function has UI side effects.
    """
    st.title("📖 Story Explorer")
    st.markdown("""
    This application allows you to explore stories and their associated intervention points.
    Select a story from the dropdown menu to view its content and highlighted intervention points.
    """)


def render_story_header(*, story_title: str, story_id: str) -> None:
    """
    Renders the story header.

    Args:
        story_title: Title of the story.
        story_id: ID of the story.

    This function has UI side effects.
    """
    st.header(f"Story: {story_title}")
    st.subheader(f"Story ID: {story_id}")


def render_story_selection(*, story_options: list[str], story_ids: list[str]) -> str:
    """
    Renders story selection dropdown and returns the selected story ID.

    Args:
        story_options: List of story display options.
        story_ids: List of corresponding story IDs.

    Returns:
        Selected story ID.

    This function has UI side effects.
    """
    st.sidebar.header("Story Selection")

    selected_index = st.sidebar.selectbox(
        "Choose a story:",
        options=range(len(story_options)),
        format_func=lambda i: story_options[i],
        index=0,
    )

    return story_ids[selected_index]


def render_error_message() -> None:
    """
    Renders an error message when data loading fails.

    This function has UI side effects.
    """
    st.error(
        "Failed to load data. Please check if the data files exist in the correct locations."
    )


def render_fallback_story(*, story_text: str) -> None:
    """
    Renders the story text without interventions.

    Args:
        story_text: Text of the story.

    This function has UI side effects.
    """
    st.info("No intervention points found for this story.")
    st.write(story_text)
