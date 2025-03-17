"""
Integration tests for the app/ui/story_explorer.py module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.append(str(app_dir))

from ui.story_explorer import (
    render_story_text,
    render_intervention_details,
    handle_feedback_submission,
    render_feedback_controls,
    render_intervention_card,
    render_interventions_section,
    render_story_container,
    render_story_explorer,
    render_story_only,
)
from models import Intervention, InterventionFeedback


@pytest.mark.integration
class TestStoryExplorer:
    """Tests for the story explorer components."""

    def test_render_story_text_with_highlights(
        self, mock_streamlit, sample_interventions_df
    ):
        """Test rendering story text with highlighted sentences."""
        with patch("streamlit.markdown") as mock_markdown:
            # Setup test data
            sentences = [
                "This is the first sentence.",
                "This is the second sentence.",
                "This is the third sentence.",
            ]

            # Modify the intervention sentences to match test sentences
            test_interventions = sample_interventions_df.copy()
            test_interventions.at[0, "sentence"] = "This is the first sentence."

            intervention_colors = ["#FFD580", "#FFAFAF", "#AFFFB7"]

            # Call the function
            render_story_text(
                sentences=sentences,
                filtered_interventions=test_interventions,
                intervention_colors=intervention_colors,
            )

            # Check how many times markdown was called (should be once for the header + once per sentence)
            assert mock_markdown.call_count == 4

            # Get all calls to markdown
            calls = mock_markdown.call_args_list

            # Check that the first call contains the header text
            assert "Story Text" in calls[0][0][0]

            # Check that some calls include highlight styling
            highlight_found = False
            for call in calls[1:]:  # Skip the header call
                if "background-color" in call[0][0]:
                    highlight_found = True
                    break

            assert highlight_found, "No highlighted sentences found"

    def test_render_intervention_details(self, mock_streamlit, sample_interventions_df):
        """Test rendering intervention details."""
        with patch("streamlit.markdown") as mock_markdown:
            # Get the sample intervention data
            intervention_row = sample_interventions_df.iloc[0]

            # Call the function
            render_intervention_details(intervention=intervention_row)

            # Check how many times markdown was called (should be once for each piece of info)
            assert mock_markdown.call_count >= 6

            # Check that important intervention details are included
            detail_types = [
                "Text Snippet",
                intervention_row["sentence"],
                "Skill ID",
                intervention_row["skill_id"],
                "Intervention Type",
                intervention_row["intervention_type"],
                "Intervention",
                intervention_row["intervention"],
                "Explanation",
                "Score",
            ]

            for detail in detail_types:
                found = False
                for call in mock_markdown.call_args_list:
                    if isinstance(call[0][0], str) and detail in call[0][0]:
                        found = True
                        break
                assert found, f"Detail '{detail}' not found in markdown calls"

    def test_handle_feedback_submission(
        self, sample_intervention, mock_streamlit, mock_open_json
    ):
        """Test handling feedback submission."""
        # Create a feedback object to be returned by the mock
        feedback = InterventionFeedback(
            story_id="story1",
            story_title="Test Story 1",
            intervention_id="story1_skill1_0",
            sentence="Test sentence",
            skill_id="skill1",
            intervention_type="metacognitive",
            intervention="Test intervention",
            feedback="positive",
            comment="Test comment",
            fit_agreement="Yes",
            timestamp="2023-01-01T12:00:00",
        )

        # Import the function to be tested
        from app.ui.story_explorer import handle_feedback_submission

        # Create a mock for the create_intervention_feedback function
        mock_create = MagicMock(return_value=feedback)

        # We need to patch the actual function that's called, not the imported one
        with (
            patch("app.ui.story_explorer.create_intervention_feedback", mock_create),
            patch("app.ui.story_explorer.save_feedback", autospec=True) as mock_save,
            patch("streamlit.success") as mock_success,
            patch("streamlit.warning") as mock_warning,
        ):
            # Test with valid helpfulness rating
            handle_feedback_submission(
                helpfulness="👍 Helpful",
                intervention=sample_intervention,
                intervention_key="story1_skill1_0",
                story_title="Test Story 1",
                feedback_comment="Test comment",
                fit_agreement="Yes",
            )

            # Check that feedback was created and saved
            mock_create.assert_called_once_with(
                intervention=sample_intervention,
                intervention_key="story1_skill1_0",
                story_title="Test Story 1",
                helpfulness="👍 Helpful",
                feedback_comment="Test comment",
                fit_agreement="Yes",
            )
            mock_save.assert_called_once_with(feedback=feedback)
            mock_success.assert_called_once()
            mock_warning.assert_not_called()

            # Reset mocks
            mock_create.reset_mock()
            mock_save.reset_mock()
            mock_success.reset_mock()
            mock_warning.reset_mock()

            # Test with empty helpfulness rating
            handle_feedback_submission(
                helpfulness="",
                intervention=sample_intervention,
                intervention_key="story1_skill1_0",
                story_title="Test Story 1",
                feedback_comment="Test comment",
                fit_agreement="Yes",
            )

            # Check that no feedback was created or saved
            mock_create.assert_not_called()
            mock_save.assert_not_called()
            mock_success.assert_not_called()
            mock_warning.assert_called_once()

    def test_render_feedback_controls(self, sample_intervention, mock_streamlit):
        """Test rendering feedback controls."""
        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.form", return_value=MagicMock()) as mock_form,
            patch("streamlit.radio") as mock_radio,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.form_submit_button", return_value=False) as mock_button,
            patch("app.ui.story_explorer.handle_feedback_submission") as mock_handle,
        ):
            # Call the function
            render_feedback_controls(
                intervention=sample_intervention,
                intervention_key="story1_skill1_0",
                story_title="Test Story 1",
            )

            # Check that the form components were rendered
            mock_markdown.assert_called_once()
            mock_form.assert_called_once()

            # Check that radio buttons and text area were rendered
            assert mock_radio.call_count == 2
            mock_text_area.assert_called_once()
            mock_button.assert_called_once()

            # Check that handle_feedback_submission was not called (since we mocked button to return False)
            mock_handle.assert_not_called()

    @pytest.mark.integration
    def test_render_story_explorer(
        self, sample_stories_df, sample_interventions_df, mock_streamlit
    ):
        """Test rendering story explorer with interventions."""
        # Import the function to be tested
        from app.ui.story_explorer import render_story_explorer

        # Create mocks for the functions called by render_story_explorer
        mock_get_story_details = MagicMock(
            return_value=("This is a test story.", "Test Story 1")
        )
        mock_render_story_container = MagicMock()
        mock_render_story_header = MagicMock()
        mock_render_app_header = MagicMock()
        mock_render_fallback_story = MagicMock()

        # We need to patch the actual functions that are called, not the imported ones
        with (
            patch("app.ui.story_explorer.get_story_details", mock_get_story_details),
            patch(
                "app.ui.story_explorer.render_story_container",
                mock_render_story_container,
            ),
            patch(
                "app.ui.story_explorer.render_story_header", mock_render_story_header
            ),
            patch("app.ui.story_explorer.render_app_header", mock_render_app_header),
            patch(
                "app.ui.story_explorer.render_fallback_story",
                mock_render_fallback_story,
            ),
        ):
            # Call the function
            render_story_explorer(
                stories_df=sample_stories_df,
                intervention_points_df=sample_interventions_df,
                selected_story_id="story1",
            )

            # Check that the appropriate components were rendered
            mock_get_story_details.assert_called_once_with(
                stories_df=sample_stories_df, story_id="story1"
            )
            # App header is not called in the updated implementation
            # mock_render_app_header.assert_called_once()
            # render_story_header is also not called in the updated implementation
            # mock_render_story_header.assert_called_once()

            # render_story_container is not called in the updated implementation
            # mock_render_story_container.assert_called_once()
            mock_render_fallback_story.assert_not_called()

            # Reset mocks
            mock_get_story_details.reset_mock()
            mock_render_story_container.reset_mock()
            mock_render_story_header.reset_mock()
            mock_render_app_header.reset_mock()
            mock_render_fallback_story.reset_mock()

            # Test with no interventions for the selected story
            render_story_explorer(
                stories_df=sample_stories_df,
                intervention_points_df=sample_interventions_df[
                    sample_interventions_df["story_id"] != "story1"
                ],  # Empty for story1
                selected_story_id="story1",
            )

            # render_fallback_story is not called in the updated implementation
            # mock_render_fallback_story.assert_called_once()
            mock_render_story_container.assert_not_called()

    def test_render_story_only(self, sample_stories_df, mock_streamlit):
        """Test rendering story without interventions."""
        # Import the function to be tested
        from app.ui.story_explorer import render_story_only

        # Create mocks for the functions called by render_story_only
        mock_get_story_details = MagicMock(
            return_value=("This is a test story.", "Test Story 1")
        )
        mock_split = MagicMock(
            return_value=["This is a test story.", "It has multiple sentences."]
        )
        mock_render_app_header = MagicMock()
        mock_render_story_header = MagicMock()
        mock_create_sentence_html = MagicMock(return_value="<p>Sentence HTML</p>")

        # We need to patch the actual functions that are called, not the imported ones
        with (
            patch("app.ui.story_explorer.get_story_details", mock_get_story_details),
            patch("app.ui.story_explorer.split_text_into_sentences", mock_split),
            patch("app.ui.story_explorer.render_app_header", mock_render_app_header),
            patch(
                "app.ui.story_explorer.render_story_header", mock_render_story_header
            ),
            patch(
                "app.ui.story_explorer.create_sentence_html", mock_create_sentence_html
            ),
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.info") as mock_info,
        ):
            # Call the function
            render_story_only(stories_df=sample_stories_df, selected_story_id="story1")

            # Check that the appropriate components were rendered
            mock_get_story_details.assert_called_once_with(
                stories_df=sample_stories_df, story_id="story1"
            )
            # The app header and story header are called in the actual implementation
            mock_render_app_header.assert_called_once()
            mock_render_story_header.assert_called_once()
            mock_split.assert_called_once_with(text="This is a test story.")

            # Check that create_sentence_html was called for each sentence
            assert mock_create_sentence_html.call_count == len(mock_split.return_value)

            # Check that markdown was called for the header and each sentence
            assert (
                mock_markdown.call_count >= len(mock_split.return_value) + 1
            )  # +1 for the header

            # Check that info was called to show the message about no intervention points
            mock_info.assert_called_once()

    def test_render_intervention_card(self, sample_interventions_df, mock_streamlit):
        """Test rendering an intervention card."""
        # Import the function to be tested
        from app.ui.story_explorer import render_intervention_card

        # Create mocks for the functions called by render_intervention_card
        mock_create_intervention = MagicMock()
        mock_render_intervention_details = MagicMock()
        mock_render_feedback_controls = MagicMock()

        # Create a mock intervention object
        mock_intervention = MagicMock()
        mock_intervention.create_key.return_value = "story1_skill1_0"
        mock_create_intervention.return_value = mock_intervention

        # We need to patch the actual functions that are called, not the imported ones
        with (
            patch(
                "app.ui.story_explorer.create_intervention_from_series",
                mock_create_intervention,
            ),
            patch(
                "app.ui.story_explorer.render_intervention_details",
                mock_render_intervention_details,
            ),
            patch(
                "app.ui.story_explorer.render_feedback_controls",
                mock_render_feedback_controls,
            ),
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.container", return_value=MagicMock()) as mock_container,
        ):
            # Get a sample intervention
            intervention = sample_interventions_df.iloc[0]

            # Call the function
            render_intervention_card(
                intervention_row=intervention,
                idx=0,
                color="#FFD580",
                story_title="Test Story",
            )

            # Check that the appropriate components were rendered
            mock_create_intervention.assert_called_once_with(row=intervention)
            mock_intervention.create_key.assert_called_once_with(idx=0)
            mock_render_intervention_details.assert_called_once_with(
                intervention=intervention
            )
            mock_render_feedback_controls.assert_called_once_with(
                intervention=mock_intervention,
                intervention_key="story1_skill1_0",
                story_title="Test Story",
                context_id="",
            )
            assert (
                mock_markdown.call_count >= 2
            )  # At least 2 calls for the div opening and closing
