"""
Integration tests for the app/ui/processing.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.append(str(app_dir))

from ui.processing import render_process_stories


@pytest.mark.integration
class TestProcessing:
    """Tests for the story processing UI components."""

    def test_render_process_stories(self, mock_streamlit):
        """Test rendering the process stories page."""
        with (
            patch("streamlit.title") as mock_title,
            patch("streamlit.markdown") as mock_markdown,
            patch(
                "streamlit.tabs",
                return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
            ) as mock_tabs,
        ):
            # Call the function
            render_process_stories()

            # Check that the title and tabs were rendered
            mock_title.assert_called_once()
            mock_markdown.assert_called()
            # The function now calls st.tabs twice (once for main tabs, once for analysis tabs)
            assert mock_tabs.call_count >= 1

            # Check that the first tabs call contains the expected tab names
            first_tabs_arg = mock_tabs.call_args_list[0][0][0]
            assert "📤 Upload New Stories" in first_tabs_arg
            assert "🔍 Assign Skills to Stories" in first_tabs_arg
            assert "💡 Identify Intervention Points" in first_tabs_arg
            assert "📊 Skill Analytics" in first_tabs_arg
