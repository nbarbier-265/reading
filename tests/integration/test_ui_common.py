"""
Integration tests for the app/ui/common.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.append(str(app_dir))
ui_dir = app_dir / "ui"
sys.path.append(str(ui_dir))

from ui.common import (
    render_app_header,
    render_story_header,
    render_story_selection,
    render_error_message,
    render_fallback_story,
)


@pytest.mark.integration
class TestCommonUIComponents:
    """Tests for common UI components."""

    def test_render_app_header(self, mock_streamlit):
        """Test rendering the app header."""
        with (
            patch("streamlit.title") as mock_title,
            patch("streamlit.markdown") as mock_markdown,
        ):
            render_app_header()

            # Check that streamlit.title was called with the correct title
            mock_title.assert_called_once()
            args, _ = mock_title.call_args
            assert "Story Explorer" in args[0]

            # Check that streamlit.markdown was called with some text
            mock_markdown.assert_called_once()
            args, _ = mock_markdown.call_args
            assert isinstance(args[0], str)
            assert len(args[0]) > 0

    def test_render_story_header(self, mock_streamlit):
        """Test rendering the story header."""
        with (
            patch("streamlit.header") as mock_header,
            patch("streamlit.subheader") as mock_subheader,
        ):
            # Call the function with test data
            story_title = "Test Story"
            story_id = "test123"
            render_story_header(story_title=story_title, story_id=story_id)

            # Check that streamlit.header was called with the correct title
            mock_header.assert_called_once()
            args, _ = mock_header.call_args
            assert story_title in args[0]

            # Check that streamlit.subheader was called with the correct ID
            mock_subheader.assert_called_once()
            args, _ = mock_subheader.call_args
            assert story_id in args[0]

    def test_render_story_selection(self, mock_streamlit):
        """Test rendering the story selection dropdown."""
        with (
            patch("streamlit.sidebar.header") as mock_header,
            patch("streamlit.sidebar.selectbox", return_value=1) as mock_selectbox,
        ):
            # Call the function with test data
            story_options = ["Story 1", "Story 2", "Story 3"]
            story_ids = ["id1", "id2", "id3"]
            result = render_story_selection(
                story_options=story_options, story_ids=story_ids
            )

            # Check that streamlit.sidebar.header was called
            mock_header.assert_called_once()

            # Check that streamlit.sidebar.selectbox was called with the correct options
            mock_selectbox.assert_called_once()
            args, kwargs = mock_selectbox.call_args
            assert "options" in kwargs
            assert len(kwargs["options"]) == len(story_options)

            # Check that the function returns the correct story ID
            assert result == "id2"  # Since we mocked selectbox to return index 1

    def test_render_error_message(self, mock_streamlit):
        """Test rendering an error message."""
        with patch("streamlit.error") as mock_error:
            render_error_message()

            # Check that streamlit.error was called with some text
            mock_error.assert_called_once()
            args, _ = mock_error.call_args
            assert isinstance(args[0], str)
            assert len(args[0]) > 0
            assert "Failed to load data" in args[0]

    def test_render_fallback_story(self, mock_streamlit):
        """Test rendering a fallback story view."""
        with (
            patch("streamlit.info") as mock_info,
            patch("streamlit.write") as mock_write,
        ):
            story_text = "This is a test story."
            render_fallback_story(story_text=story_text)

            # Check that streamlit.info was called with the correct message
            mock_info.assert_called_once()
            args, _ = mock_info.call_args
            assert "No intervention points" in args[0]

            # Check that streamlit.write was called with the story text
            mock_write.assert_called_once()
            args, _ = mock_write.call_args
            assert args[0] == story_text
