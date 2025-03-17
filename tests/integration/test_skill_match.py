"""
Integration tests for the app/ui/skill_match.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.append(str(app_dir))

from ui.skill_match import render_skill_match_item, render_skill_match_page
from models import SkillMatchFeedback


@pytest.mark.integration
class TestSkillMatch:
    """Tests for the skill match UI components."""

    def test_render_skill_match_item(self, mock_streamlit, sample_skill_match_df):
        """Test rendering a skill match item."""
        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.container", return_value=MagicMock()) as mock_container,
            patch(
                "streamlit.columns", return_value=[MagicMock(), MagicMock()]
            ) as mock_columns,
            patch("streamlit.form", return_value=MagicMock()) as mock_form,
            patch("streamlit.radio") as mock_radio,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.form_submit_button", return_value=False) as mock_button,
            patch(
                "app.ui.skill_match.create_skill_match_feedback"
            ) as mock_create_feedback,
            patch("app.ui.skill_match.save_skill_match_feedback") as mock_save_feedback,
            patch("streamlit.success") as mock_success,
            patch("streamlit.warning") as mock_warning,
        ):
            # Get a sample skill match row
            match_row = sample_skill_match_df.iloc[0]

            # Call the function
            render_skill_match_item(match_row=match_row, story_title="Test Story 1")

            # Check that the UI components were rendered
            mock_container.assert_called_once()
            mock_columns.assert_called_once()
            assert (
                mock_markdown.call_count >= 4
            )  # Should be called for multiple pieces of information
            mock_form.assert_called_once()
            mock_radio.assert_called_once()
            mock_text_area.assert_called_once()
            mock_button.assert_called_once()

            # Since we mocked button to return False, feedback shouldn't be created or saved
            mock_create_feedback.assert_not_called()
            mock_save_feedback.assert_not_called()
            mock_success.assert_not_called()
            mock_warning.assert_not_called()

    def test_render_skill_match_item_with_submit(
        self, mock_streamlit, sample_skill_match_df, mock_open_json
    ):
        """Test rendering a skill match item with form submission."""
        # Import the function to be tested
        from app.ui.skill_match import render_skill_match_item

        # Create a feedback object to be returned by the mock
        feedback = SkillMatchFeedback(
            story_id="story1",
            story_title="Test Story 1",
            skill_id="skill1",
            parent_score=0.82,
            child_score=0.89,
            accuracy="👍 Accurate",
            comment="Good match",
            method="embedding",
            timestamp="2023-01-01T12:00:00",
        )

        # Get a sample skill match row
        match_row = sample_skill_match_df.iloc[0]

        # Create a mock for the create_skill_match_feedback function
        mock_create_feedback = MagicMock(return_value=feedback)
        mock_save_feedback = MagicMock()

        # We need to patch the actual function that's called, not the imported one
        with (
            patch(
                "app.ui.skill_match.create_skill_match_feedback", mock_create_feedback
            ),
            patch("app.ui.skill_match.save_skill_match_feedback", mock_save_feedback),
            patch(
                "app.ui.skill_match.create_skill_match_from_series",
                return_value=MagicMock(
                    story_id="story1",
                    skill_id="skill1",
                    parent_score=0.82,
                    child_score=0.89,
                    parent_explanation="The story demonstrates this specific skill.",
                    child_explanation="The story demonstrates this specific skill.",
                    method="embedding",
                ),
            ),
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.container", return_value=MagicMock()) as mock_container,
            patch(
                "streamlit.columns", return_value=[MagicMock(), MagicMock()]
            ) as mock_columns,
            patch("streamlit.form", return_value=MagicMock()) as mock_form,
            patch("streamlit.radio", return_value="👍 Accurate") as mock_radio,
            patch("streamlit.text_area", return_value="Good match") as mock_text_area,
            patch("streamlit.form_submit_button", return_value=True) as mock_button,
            patch("streamlit.success") as mock_success,
        ):
            # Call the function
            render_skill_match_item(match_row=match_row, story_title="Test Story 1")

            # Check that the feedback was created and saved
            mock_create_feedback.assert_called_once()
            mock_save_feedback.assert_called_once_with(feedback=feedback)
            mock_success.assert_called_once()

    def test_render_skill_match_page(
        self, mock_streamlit, sample_stories_df, sample_skill_match_df, mock_open_json
    ):
        """Test rendering the skill match feedback page."""
        # Import the function to be tested
        from app.ui.skill_match import render_skill_match_page

        # Create mocks for the functions called by render_skill_match_page
        mock_get_story_details = MagicMock(
            return_value=("This is a test story.", "Test Story 1")
        )
        mock_render_story_header = MagicMock()
        mock_render_item = MagicMock()

        # We need to patch the actual functions that are called, not the imported ones
        with (
            patch("app.ui.skill_match.get_story_details", mock_get_story_details),
            patch("app.ui.skill_match.render_story_header", mock_render_story_header),
            patch("app.ui.skill_match.render_skill_match_item", mock_render_item),
            patch("streamlit.header") as mock_header,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.expander", return_value=MagicMock()) as mock_expander,
            patch("streamlit.write") as mock_write,
            patch("streamlit.subheader") as mock_subheader,
            patch("streamlit.info") as mock_info,
            patch(
                "streamlit.columns", return_value=[MagicMock(), MagicMock()]
            ) as mock_columns,
        ):
            # Filter skill match data to include only matches for story1
            story_skill_matches = sample_skill_match_df[
                sample_skill_match_df["story_id"] == "story1"
            ]

            # Call the function
            render_skill_match_page(
                stories_df=sample_stories_df,
                skill_match_df=story_skill_matches,
                selected_story_id="story1",
                prompt_version="v1",
            )

            # Check that UI components were rendered correctly
            mock_header.assert_called_once()
            mock_get_story_details.assert_called_once_with(
                stories_df=sample_stories_df, story_id="story1"
            )
            mock_columns.assert_called_once()
            mock_expander.assert_called()

            # Test with empty skill matches
            mock_header.reset_mock()
            mock_get_story_details.reset_mock()
            mock_render_story_header.reset_mock()
            mock_expander.reset_mock()
            mock_subheader.reset_mock()
            mock_info.reset_mock()

            # Call the function with empty skill matches
            render_skill_match_page(
                stories_df=sample_stories_df,
                skill_match_df=sample_skill_match_df[
                    sample_skill_match_df["story_id"] == "nonexistent"
                ],
                selected_story_id="story1",
            )

            # Check that info message was shown
            mock_info.assert_called_once()
            mock_render_item.assert_not_called()

    def test_render_skill_match_page_empty(
        self, mock_streamlit, sample_stories_df, sample_skill_match_df
    ):
        """Test rendering the skill match page with no matches for the story."""
        # Create a mock for the render_skill_match_item function
        mock_render_item = MagicMock()

        # We need to patch the actual function that's called, not the imported one
        with (
            patch("app.ui.skill_match.render_skill_match_item", mock_render_item),
            patch("streamlit.header") as mock_header,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.expander", return_value=MagicMock()) as mock_expander,
            patch("streamlit.write") as mock_write,
            patch("streamlit.info") as mock_info,
        ):
            # Filter to create an empty DataFrame (no matches for story3)
            empty_matches = sample_skill_match_df[
                sample_skill_match_df["story_id"] == "story3"
            ]

            # Call the function
            render_skill_match_page(
                stories_df=sample_stories_df,
                skill_match_df=empty_matches,
                selected_story_id="story1",
                prompt_version="v1",
            )

            # Check that UI components were rendered correctly
            mock_header.assert_called_once()
            mock_info.assert_called_once()  # Should show "No skill matches found" message
            mock_render_item.assert_not_called()  # Should not render any items

    def test_render_skill_match_page_none(self, mock_streamlit, sample_stories_df):
        """Test rendering the skill match page with None for skill_match_df."""
        with (
            patch("streamlit.header") as mock_header,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.info") as mock_info,
        ):
            # Call the function with None for skill_match_df
            render_skill_match_page(
                stories_df=sample_stories_df,
                skill_match_df=None,
                selected_story_id="story1",
                prompt_version="v1",
            )

            # Check that header and info message were shown
            mock_header.assert_called_once()
            # markdown may not be called in the updated UI
            # assert mock_markdown.call_count >= 1
            mock_info.assert_called_once()
