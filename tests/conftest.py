"""
Common test fixtures and configuration for pytest.
"""

import os
import pandas as pd
import pytest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent / "app"
sys.path.append(str(app_dir))

from models import (
    Story,
    Intervention,
    SkillMatch,
    InterventionFeedback,
    SkillMatchFeedback,
)

TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_streamlit():
    """Mock all streamlit functions to prevent UI side effects during tests."""
    streamlit_patches = [
        patch("streamlit.set_page_config"),
        patch("streamlit.title"),
        patch("streamlit.header"),
        patch("streamlit.subheader"),
        patch("streamlit.markdown"),
        patch("streamlit.write"),
        patch("streamlit.info"),
        patch("streamlit.success"),
        patch("streamlit.warning"),
        patch("streamlit.error"),
        patch("streamlit.sidebar.radio", return_value="Review Results"),
        patch("streamlit.sidebar.selectbox", return_value=0),
        patch("streamlit.sidebar.header"),
        patch("streamlit.tabs", return_value=[MagicMock(), MagicMock()]),
        patch("streamlit.container", return_value=MagicMock()),
        patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]),
        patch("streamlit.expander", return_value=MagicMock()),
        patch("streamlit.form", return_value=MagicMock()),
        patch("streamlit.radio", return_value="👍 Helpful"),
        patch("streamlit.text_area", return_value="Test comment"),
        patch("streamlit.form_submit_button", return_value=True),
        patch("streamlit.metric"),
        patch("streamlit.bar_chart"),
        patch("streamlit.slider", return_value=0.7),
        patch("streamlit.number_input", return_value=3),
        patch("streamlit.button", return_value=False),
        patch("streamlit.file_uploader", return_value=None),
        patch("streamlit.progress", return_value=MagicMock()),
        patch("streamlit.empty", return_value=MagicMock()),
        patch("streamlit.text"),
        patch("streamlit.cache_data", lambda f: f),
    ]

    for p in streamlit_patches:
        p.start()

    yield

    for p in streamlit_patches:
        p.stop()


@pytest.fixture
def sample_stories_df():
    """Create a sample stories DataFrame for testing."""
    return pd.DataFrame(
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


@pytest.fixture
def sample_interventions_df():
    """Create a sample interventions DataFrame for testing."""
    return pd.DataFrame(
        {
            "story_id": ["story1", "story1", "story2"],
            "skill_id": ["skill1", "skill2", "skill1"],
            "sentence": [
                "It has multiple sentences.",
                "Testing is important.",
                "With different content.",
            ],
            "intervention_type": ["metacognitive", "conceptual", "vocabulary"],
            "intervention": [
                "Think about what this means.",
                "Consider the importance of testing.",
                "Let's explore the meaning of different content.",
            ],
            "explanation": [
                "This intervention helps readers understand sentence structure.",
                "This intervention reinforces the importance of testing.",
                "This intervention focuses on vocabulary development.",
            ],
            "score": [0.85, 0.76, 0.92],
        }
    )


@pytest.fixture
def sample_skill_match_df():
    """Create a sample skill match DataFrame for testing."""
    return pd.DataFrame(
        {
            "story_id": ["story1", "story1", "story2"],
            "skill_id": ["skill1", "skill2", "skill1"],
            "parent_score": [0.82, 0.75, 0.90],
            "child_score": [0.89, 0.78, 0.93],
            "method": ["embedding", "embedding", "embedding"],
            "parent_explanation": [
                "The story relates strongly to this parent skill.",
                "The story somewhat relates to this parent skill.",
                "The story relates very strongly to this parent skill.",
            ],
            "child_explanation": [
                "The story demonstrates this specific skill.",
                "The story somewhat demonstrates this specific skill.",
                "The story clearly demonstrates this specific skill.",
            ],
        }
    )


@pytest.fixture
def sample_story():
    """Create a sample Story instance for testing."""
    return Story(
        story_id="story1",
        title="Test Story 1",
        story_text="This is a test story. It has multiple sentences. Testing is important.",
    )


@pytest.fixture
def sample_intervention():
    """Create a sample Intervention instance for testing."""
    return Intervention(
        story_id="story1",
        skill_id="skill1",
        sentence="It has multiple sentences.",
        intervention_type="metacognitive",
        intervention="Think about what this means.",
        explanation="This intervention helps readers understand sentence structure.",
        score=0.85,
    )


@pytest.fixture
def sample_skill_match():
    """Create a sample SkillMatch instance for testing."""
    return SkillMatch(
        story_id="story1",
        skill_id="skill1",
        parent_score=0.82,
        child_score=0.89,
        method="embedding",
        parent_explanation="The story relates strongly to this parent skill.",
        child_explanation="The story demonstrates this specific skill.",
    )


@pytest.fixture
def mock_path_to_project():
    """Mock the PATH_TO_PROJECT environment variable."""
    with patch.dict(os.environ, {"PATH_TO_PROJECT": str(TEST_DATA_DIR)}):
        yield TEST_DATA_DIR


@pytest.fixture
def mock_file_exists():
    """Mock Path.exists to return True for test files."""
    mock = MagicMock()
    mock.return_value = True

    with patch("pathlib.Path.exists", mock):
        yield mock


@pytest.fixture
def mock_read_csv():
    """Mock pandas.read_csv to return test DataFrames."""

    def _mock_read_csv(filepath, *args, **kwargs):
        path = Path(filepath)
        if "stories.csv" in str(path):
            return pd.DataFrame(
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
        elif "intervention_points_llm.csv" in str(path):
            return pd.DataFrame(
                {
                    "story_id": ["story1", "story1", "story2"],
                    "skill_id": ["skill1", "skill2", "skill1"],
                    "sentence": [
                        "It has multiple sentences.",
                        "Testing is important.",
                        "With different content.",
                    ],
                    "intervention_type": ["metacognitive", "conceptual", "vocabulary"],
                    "intervention": [
                        "Think about what this means.",
                        "Consider the importance of testing.",
                        "Let's explore the meaning of different content.",
                    ],
                    "explanation": [
                        "This intervention helps readers understand sentence structure.",
                        "This intervention reinforces the importance of testing.",
                        "This intervention focuses on vocabulary development.",
                    ],
                    "score": [0.85, 0.76, 0.92],
                }
            )
        elif "skill_match_results.csv" in str(path):
            return pd.DataFrame(
                {
                    "story_id": ["story1", "story1", "story2"],
                    "skill_id": ["skill1", "skill2", "skill1"],
                    "parent_score": [0.82, 0.75, 0.90],
                    "child_score": [0.89, 0.78, 0.93],
                    "method": ["embedding", "embedding", "embedding"],
                    "parent_explanation": [
                        "The story relates strongly to this parent skill.",
                        "The story somewhat relates to this parent skill.",
                        "The story relates very strongly to this parent skill.",
                    ],
                    "child_explanation": [
                        "The story demonstrates this specific skill.",
                        "The story somewhat demonstrates this specific skill.",
                        "The story clearly demonstrates this specific skill.",
                    ],
                }
            )
        else:
            return pd.DataFrame()

    with patch("pandas.read_csv", _mock_read_csv):
        yield


@pytest.fixture
def mock_save_json():
    """Mock json.dump to prevent writing files during tests."""
    with patch("json.dump") as mock_dump:
        yield mock_dump


@pytest.fixture
def mock_open_json():
    """Mock open and json.load to return test data during tests."""

    def _mock_json_data(filepath):
        if "intervention_feedback.json" in str(filepath):
            return [
                asdict(
                    InterventionFeedback(
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
                )
            ]
        elif "skill_match_feedback.json" in str(filepath):
            return [
                asdict(
                    SkillMatchFeedback(
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
                )
            ]
        else:
            return []

    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=None)

    with (
        patch("builtins.open", return_value=mock_file),
        patch("json.load", side_effect=_mock_json_data),
    ):
        yield


@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data for testing."""
    return {
        "story_id": "story1",
        "story_title": "Test Story 1",
        "intervention_id": "story1_skill1_0",
        "sentence": "It has multiple sentences.",
        "skill_id": "skill1",
        "intervention_type": "metacognitive",
        "intervention": "Think about what this means.",
        "feedback": "positive",
        "comment": "Good intervention",
        "fit_agreement": "Yes",
        "timestamp": "2023-01-01T12:00:00",
    }
