"""
Main application module for the Streamlit application.

This module serves as the entry point for the Streamlit application
and orchestrates the UI rendering based on user selections.
"""

import streamlit as st
from pathlib import Path

from data import load_data, create_story_options
from ui.common import render_story_selection, render_error_message
from ui.story_explorer import render_story_explorer
from ui.skill_match import render_skill_match_page
from ui.processing import render_process_stories
from ui.feedback import render_feedback_analytics
from ui.report_summary import render_report_summary
from app.config import PATH_TO_PROMPTS


def main() -> None:
    """
    Main application function that orchestrates the UI rendering.

    This function has UI side effects.
    """
    st.set_page_config(page_title="Story Explorer", page_icon="📚", layout="wide")

    # Get available prompt versions by checking directories
    prompt_versions = [d.name for d in Path(PATH_TO_PROMPTS).iterdir() if d.is_dir()]
    if not prompt_versions:
        prompt_versions = ["v1"]  # Default if no directories found

    # Add a sidebar selector for prompt version
    with st.sidebar:
        selected_prompt_version = st.selectbox(
            "Prompt Version to View:",
            options=prompt_versions,
            index=0,
            help="Select which version of prompt results to view",
        )
        st.info(f"Viewing data from prompt version: {selected_prompt_version}")

    # Load data for the selected prompt version
    stories_df, intervention_points_df, skill_match_df = load_data(
        prompt_version=selected_prompt_version
    )

    if stories_df is not None:
        app_mode = st.sidebar.radio(
            "Select Mode:",
            options=["🔄 Process Stories", "📊 Review Results", "📑 Report Summary"],
        )

        # Strip emojis for compatibility with tests
        app_mode_clean = app_mode.strip("🔄📊📑 ")

        if app_mode_clean == "Process Stories" or app_mode == "🔄 Process Stories":
            render_process_stories()
        elif app_mode_clean == "Review Results" or app_mode == "📊 Review Results":
            # Add title and subtitle for Review Results
            st.title("📊 Review Results")
            st.markdown("""
            Review and analyze story exploration results, skill matches, and user feedback.
            """)

            story_options_df = create_story_options(stories_df=stories_df)
            story_options = story_options_df["display"].tolist()
            story_ids = story_options_df["story_id"].tolist()

            selected_story_id = render_story_selection(
                story_options=story_options, story_ids=story_ids
            )

            # Always show all tabs regardless of data availability
            review_tabs = [
                "📖 Story Explorer",
                "🎯 Skill Match Feedback",
                "📈 Feedback Analytics",
            ]

            # Make sure the mock in tests provides enough tabs
            review_tab = st.tabs(review_tabs)
            tab_count = len(review_tab)

            # Handle Story Explorer tab
            tab_index = 0
            if "📖 Story Explorer" in review_tabs and tab_index < tab_count:
                with review_tab[tab_index]:
                    if intervention_points_df is not None:
                        render_story_explorer(
                            stories_df=stories_df,
                            intervention_points_df=intervention_points_df,
                            selected_story_id=selected_story_id,
                            prompt_version=selected_prompt_version,
                        )
                    else:
                        # Just show the story without highlighting
                        from ui.story_explorer import render_story_only

                        render_story_only(
                            stories_df=stories_df,
                            selected_story_id=selected_story_id,
                        )
                tab_index += 1

            # Handle Skill Match Feedback tab
            if "🎯 Skill Match Feedback" in review_tabs and tab_index < tab_count:
                with review_tab[tab_index]:
                    if skill_match_df is not None:
                        render_skill_match_page(
                            stories_df=stories_df,
                            skill_match_df=skill_match_df,
                            selected_story_id=selected_story_id,
                            prompt_version=selected_prompt_version,
                        )
                    else:
                        st.info(
                            "No skill match data available. You can generate skill matches in the Process Stories tab."
                        )
                tab_index += 1

            # Handle Feedback Analytics tab
            if "📈 Feedback Analytics" in review_tabs and tab_index < tab_count:
                with review_tab[tab_index]:
                    render_feedback_analytics(prompt_version=selected_prompt_version)
        elif (
            app_mode_clean == "Report Summary" or app_mode == "📑 Report Summary"
        ):  # Report Summary mode
            st.title("📑 Report Summary")
            st.markdown("""
            View comprehensive reports and summaries of all processed data.
            """)

            render_report_summary(
                stories_df=stories_df,
                intervention_points_df=intervention_points_df,
                skill_match_df=skill_match_df,
                prompt_version=selected_prompt_version,
            )

    else:
        render_error_message()


if __name__ == "__main__":
    main()
