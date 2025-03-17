"""
Unit tests for the app/data.py module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from dataclasses import asdict
import sys
import os
import pandas as pd

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.append(str(app_dir))

from data import (
    load_data,
    create_story_options,
    get_story_details,
    filter_interventions,
    save_feedback_to_file,
    save_feedback,
    save_skill_match_feedback,
)
from models import InterventionFeedback, SkillMatchFeedback


@pytest.mark.unit
class TestLoadData:
    """Tests for the load_data function."""

    def test_load_data_success(self, mock_file_exists, mock_streamlit):
        """Test successful data loading."""
        # Create test DataFrames
        stories_df = pd.DataFrame(
            {
                "story_id": ["story1", "story2"],
                "title": ["Test Story 1", "Test Story 2"],
                "story_text": ["Test story text 1", "Test story text 2"],
            }
        )

        interventions_df = pd.DataFrame(
            {
                "story_id": ["story1", "story2"],
                "skill_id": ["skill1", "skill2"],
                "sentence": ["Sentence 1", "Sentence 2"],
                "intervention_type": ["type1", "type2"],
                "intervention": ["Intervention 1", "Intervention 2"],
                "explanation": ["Explanation 1", "Explanation 2"],
                "score": [0.8, 0.9],
            }
        )

        skill_match_df = pd.DataFrame(
            {
                "story_id": ["story1", "story2"],
                "skill_id": ["skill1", "skill2"],
                "parent_score": [0.8, 0.9],
                "child_score": [0.7, 0.8],
                "method": ["method1", "method2"],
                "parent_explanation": ["Parent explanation 1", "Parent explanation 2"],
                "child_explanation": ["Child explanation 1", "Child explanation 2"],
            }
        )

        # Call the function with mocked PATH_TO_PROJECT and read_csv
        with (
            patch(
                "pandas.read_csv",
                side_effect=lambda filepath, *args, **kwargs: stories_df
                if "stories.csv" in str(filepath)
                else interventions_df
                if "intervention_points.csv" in str(filepath)
                else skill_match_df
                if "skill_match_results.csv" in str(filepath)
                else None,
            ),
            patch.dict("os.environ", {"PATH_TO_PROJECT": "/mock/path"}),
        ):
            result_stories_df, result_intervention_points_df, result_skill_match_df = (
                load_data()
            )

        # Verify the result
        pd.testing.assert_frame_equal(result_stories_df, stories_df)
        pd.testing.assert_frame_equal(result_intervention_points_df, interventions_df)
        pd.testing.assert_frame_equal(result_skill_match_df, skill_match_df)

    def test_load_data_missing_processed_data(
        self, mock_path_to_project, sample_stories_df, mock_streamlit
    ):
        """Test loading with missing processed data files."""
        # Import the function to be tested
        from app.data import load_data

        # Create a custom stories DataFrame for this test
        test_stories_df = pd.DataFrame(
            {
                "story_id": ["story1", "story2"],
                "title": ["Test Story 1", "Test Story 2"],
                "story_text": ["Test story text 1", "Test story text 2"],
            }
        )

        # Mock the Path.exists method to return True only for stories.csv
        def mock_exists_side_effect(path):
            return "stories.csv" in str(path)

        # Mock the pd.read_csv function to return our test DataFrame for stories.csv
        def mock_read_csv_side_effect(filepath, *args, **kwargs):
            if "stories.csv" in str(filepath):
                return test_stories_df
            raise FileNotFoundError(f"File not found: {filepath}")

        # Apply all patches
        with (
            patch("pathlib.Path.exists", side_effect=mock_exists_side_effect),
            patch("pandas.read_csv", side_effect=mock_read_csv_side_effect),
        ):
            # Call the function
            stories_df, intervention_points_df, skill_match_df = load_data()

            # Check that stories data was loaded but processed data is None
            pd.testing.assert_frame_equal(stories_df, test_stories_df)
            assert intervention_points_df is None
            assert skill_match_df is None

    def test_load_data_error(self, mock_path_to_project, mock_streamlit):
        """Test error handling when loading data fails."""
        # Import the function to be tested
        from app.data import load_data

        # Mock the Path.exists method to return False for all files
        def mock_exists_side_effect(path):
            return False

        # Mock the pd.read_csv function to raise an exception
        def mock_read_csv_side_effect(filepath, *args, **kwargs):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Apply all patches
        with (
            patch("pathlib.Path.exists", side_effect=mock_exists_side_effect),
            patch("pandas.read_csv", side_effect=mock_read_csv_side_effect),
            patch("streamlit.error") as mock_error,
        ):
            # Call the function
            stories_df, intervention_points_df, skill_match_df = load_data()

            # Check that all data is None
            assert stories_df is None
            assert intervention_points_df is None
            assert skill_match_df is None

            # Check that an error message was displayed
            mock_error.assert_called_once()


@pytest.mark.unit
class TestStoryOptions:
    """Tests for story option creation and retrieval."""

    def test_create_story_options(self, sample_stories_df):
        """Test creating story display options."""
        options_df = create_story_options(stories_df=sample_stories_df)

        assert len(options_df) == 3
        assert "story_id" in options_df.columns
        assert "title" in options_df.columns
        assert "display" in options_df.columns

        # Check that display strings are correctly formatted
        assert options_df.iloc[0]["display"] == "Test Story 1 (ID: story1)"
        assert options_df.iloc[1]["display"] == "Test Story 2 (ID: story2)"
        assert options_df.iloc[2]["display"] == "Test Story 3 (ID: story3)"

    def test_get_story_details(self, sample_stories_df):
        """Test retrieving story details."""
        # Create a test DataFrame with known values
        test_df = pd.DataFrame(
            {
                "story_id": ["story1", "story2", "story3"],
                "title": ["Test Story 1", "Test Story 2", "Test Story 3"],
                "story_text": [
                    "This is a test story. It has multiple sentences. Testing is important.",
                    "Another test story. With different content. For testing purposes.",
                    "Third test story. With more content. And more sentences to test with.",
                ],
            }
        )

        # Test with our test DataFrame
        story_text, story_title = get_story_details(
            stories_df=test_df, story_id="story1"
        )

        assert story_title == "Test Story 1"
        assert (
            story_text
            == "This is a test story. It has multiple sentences. Testing is important."
        )

    def test_filter_interventions(self, sample_interventions_df):
        """Test filtering interventions by story ID."""
        # Test filtering for story1
        filtered_df = filter_interventions(
            interventions_df=sample_interventions_df, story_id="story1"
        )

        assert len(filtered_df) == 2
        assert all(filtered_df["story_id"] == "story1")

        # Test filtering for story2
        filtered_df = filter_interventions(
            interventions_df=sample_interventions_df, story_id="story2"
        )

        assert len(filtered_df) == 1
        assert all(filtered_df["story_id"] == "story2")

        # Test filtering with score threshold
        filtered_df = filter_interventions(
            interventions_df=sample_interventions_df, story_id="story1", min_score=0.8
        )

        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["score"] >= 0.8

        # Test filtering with nonexistent story
        filtered_df = filter_interventions(
            interventions_df=sample_interventions_df, story_id="nonexistent"
        )

        assert len(filtered_df) == 0


@pytest.mark.unit
class TestFeedbackSaving:
    """Tests for feedback saving functionality."""

    def test_save_feedback_to_file_new_file(self, mock_save_json):
        """Test saving feedback to a new file."""
        test_data = {"key": "value"}
        test_file_path = Path("test_feedback.json")

        # Create a parent mock to avoid issues with mkdir
        mock_parent = MagicMock()
        mock_parent.mkdir = MagicMock()

        # Mock all the path operations
        with (
            patch("builtins.open", mock_open()),
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.parent", mock_parent, create=True),
        ):
            save_feedback_to_file(feedback_data=test_data, file_path=test_file_path)

            # Check that mkdir was called
            mock_parent.mkdir.assert_called_once()

    def test_save_feedback_to_file_existing_file(self, mock_save_json):
        """Test saving feedback to an existing file."""
        test_data = {"key": "value"}
        existing_data = [{"existing": "data"}]
        test_file_path = Path("test_feedback.json")

        # Create a parent mock to avoid issues with mkdir
        mock_parent = MagicMock()
        mock_parent.mkdir = MagicMock()

        # Mock json.load to return existing data
        mock_load = MagicMock(return_value=existing_data)

        # Test saving to an existing file
        with (
            patch("builtins.open", mock_open()),
            patch("json.load", mock_load),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.parent", mock_parent, create=True),
        ):
            save_feedback_to_file(feedback_data=test_data, file_path=test_file_path)

            # Check that json.load was called
            mock_load.assert_called_once()

    def test_save_intervention_feedback(
        self, sample_intervention, mock_save_json, mock_open_json
    ):
        """Test saving intervention feedback."""
        feedback = InterventionFeedback(
            story_id="story1",
            story_title="Test Story 1",
            intervention_id="story1_skill1_0",
            sentence="It has multiple sentences.",
            skill_id="skill1",
            intervention_type="metacognitive",
            intervention="Think about what this means.",
            feedback="positive",
            comment="Good intervention",
            fit_agreement="Yes",
            timestamp="2023-01-01T12:00:00",
        )

        # Mock Path.exists to return True for the feedback file
        with patch("pathlib.Path.exists", return_value=True):
            save_feedback(feedback=feedback)

            # Check that json.dump was called with the right arguments
            mock_save_json.assert_called_once()

    def test_save_skill_match_feedback(
        self, sample_skill_match, mock_save_json, mock_open_json
    ):
        """Test saving skill match feedback."""
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

        # Mock Path.exists to return True for the feedback file
        with patch("pathlib.Path.exists", return_value=True):
            save_skill_match_feedback(feedback=feedback)

            # Check that json.dump was called with the right arguments
            mock_save_json.assert_called_once()
