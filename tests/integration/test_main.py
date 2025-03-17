"""
Integration tests for the app/main.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.append(str(app_dir))

from main import main


@pytest.mark.integration
class TestMainApplication:
    """Tests for the main application orchestration."""

    def test_main_success(
        self,
        sample_stories_df,
        sample_interventions_df,
        sample_skill_match_df,
        mock_streamlit,
    ):
        """Test main function with successful data loading."""
        # Import the main function
        from app.main import main

        # Use MagicMock to track calls
        load_data_mock = MagicMock(
            return_value=(
                sample_stories_df,
                sample_interventions_df,
                sample_skill_match_df,
            )
        )
        create_options_mock = MagicMock(
            return_value=sample_stories_df[["story_id", "title"]].assign(display="Test")
        )
        selection_mock = MagicMock(return_value="story1")
        explorer_mock = MagicMock()
        skill_match_mock = MagicMock()
        process_mock = MagicMock()
        error_mock = MagicMock()
        sidebar_radio_mock = MagicMock(return_value="Review Results")
        tabs_mock = MagicMock(return_value=[MagicMock(), MagicMock()])

        # Apply all patches
        with (
            patch("app.main.load_data", load_data_mock),
            patch("app.main.create_story_options", create_options_mock),
            patch("app.main.render_story_selection", selection_mock),
            patch("app.main.render_story_explorer", explorer_mock),
            patch("app.main.render_skill_match_page", skill_match_mock),
            patch("app.main.render_process_stories", process_mock),
            patch("app.main.render_error_message", error_mock),
            patch("streamlit.sidebar.radio", sidebar_radio_mock),
            patch("streamlit.tabs", tabs_mock),
        ):
            # Call the main function
            main()

            # Verify the mocks were called correctly
            load_data_mock.assert_called_once()
            create_options_mock.assert_called_once()
            selection_mock.assert_called_once()
            explorer_mock.assert_called_once()
            skill_match_mock.assert_called_once()
            process_mock.assert_not_called()
            error_mock.assert_not_called()

    def test_main_process_stories(
        self,
        sample_stories_df,
        sample_interventions_df,
        sample_skill_match_df,
        mock_streamlit,
    ):
        """Test main function with Process Stories mode."""
        # Use MagicMock to track calls
        load_data_mock = MagicMock(
            return_value=(
                sample_stories_df,
                sample_interventions_df,
                sample_skill_match_df,
            )
        )
        process_mock = MagicMock()
        error_mock = MagicMock()
        sidebar_radio_mock = MagicMock(return_value="🔄 Process Stories")

        # Create a list of MagicMock objects to represent the tabs
        tabs_mock = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        # Apply all patches
        with (
            patch("app.main.load_data", load_data_mock),
            patch("app.main.render_process_stories", process_mock),
            patch("app.main.render_error_message", error_mock),
            patch("streamlit.sidebar.radio", sidebar_radio_mock),
            patch("streamlit.tabs", return_value=tabs_mock),
        ):
            # Import main function inside the patch context
            from app.main import main

            # Call the main function
            main()

            # Verify the mocks were called correctly
            load_data_mock.assert_called_once()
            process_mock.assert_called_once()
            error_mock.assert_not_called()

    def test_main_missing_data(self, sample_stories_df, mock_streamlit):
        """Test main function with missing processed data."""
        # Import the main function
        from app.main import main

        # Use MagicMock to track calls
        load_data_mock = MagicMock(return_value=(sample_stories_df, None, None))
        create_options_mock = MagicMock(
            return_value=sample_stories_df[["story_id", "title"]].assign(display="Test")
        )
        selection_mock = MagicMock(return_value="story1")
        explorer_mock = MagicMock()
        skill_match_mock = MagicMock()
        render_story_only_mock = MagicMock()
        error_mock = MagicMock()
        sidebar_radio_mock = MagicMock(return_value="Review Results")

        # Create a mock for the tabs that returns a context manager
        tab_context = MagicMock()
        tab_context.__enter__ = MagicMock(return_value=None)
        tab_context.__exit__ = MagicMock(return_value=None)
        tabs_mock = MagicMock(return_value=[tab_context])

        # Create a mock module for ui.story_explorer
        mock_story_explorer = MagicMock()
        mock_story_explorer.render_story_only = render_story_only_mock

        # Apply all patches
        with (
            patch("app.main.load_data", load_data_mock),
            patch("app.main.create_story_options", create_options_mock),
            patch("app.main.render_story_selection", selection_mock),
            patch("app.main.render_story_explorer", explorer_mock),
            patch("app.main.render_skill_match_page", skill_match_mock),
            patch("app.main.render_error_message", error_mock),
            patch.dict("sys.modules", {"ui.story_explorer": mock_story_explorer}),
            patch("streamlit.sidebar.radio", sidebar_radio_mock),
            patch("streamlit.tabs", tabs_mock),
        ):
            # Call the main function
            main()

            # Verify the mocks were called correctly
            load_data_mock.assert_called_once()
            create_options_mock.assert_called_once()
            selection_mock.assert_called_once()
            explorer_mock.assert_not_called()  # Should not be called with missing intervention data
            render_story_only_mock.assert_called_once_with(
                stories_df=sample_stories_df, selected_story_id="story1"
            )  # Should show story only
            error_mock.assert_not_called()  # No error should be shown

    def test_main_load_data_error(self, mock_streamlit):
        """Test main function when data loading fails."""
        # Use MagicMock to track calls
        load_data_mock = MagicMock(return_value=(None, None, None))
        error_mock = MagicMock()
        sidebar_radio_mock = MagicMock(return_value="Review Results")

        # Apply all patches
        with (
            patch("app.main.load_data", load_data_mock),
            patch("app.main.render_error_message", error_mock),
            patch("streamlit.sidebar.radio", sidebar_radio_mock),
        ):
            # Import main function inside the patch context
            from app.main import main

            # Call the main function
            main()

            # Verify the mocks were called correctly
            load_data_mock.assert_called_once()
            error_mock.assert_called_once()
            sidebar_radio_mock.assert_not_called()  # Should not be called when data loading fails
