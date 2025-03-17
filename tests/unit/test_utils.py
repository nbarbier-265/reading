"""
Unit tests for the app/utils.py module.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.append(str(app_dir))

from utils import (
    split_text_into_sentences,
    create_sentence_html,
    create_intervention_from_series,
    create_intervention_feedback,
    create_skill_match_from_series,
    create_skill_match_feedback,
)
from models import Intervention, SkillMatch


class TestTextProcessing:
    """Tests for text processing utilities."""

    def test_split_text_into_sentences(self):
        """Test splitting text into sentences."""
        text = "This is a test. It has multiple sentences. Testing is important!"
        sentences = split_text_into_sentences(text=text)

        assert len(sentences) == 3
        assert sentences[0] == "This is a test."
        assert sentences[1] == "It has multiple sentences."
        assert sentences[2] == "Testing is important!"

    def test_split_text_empty(self):
        """Test splitting empty text."""
        sentences = split_text_into_sentences(text="")
        assert len(sentences) == 1
        assert sentences[0] == ""

    def test_split_text_single_sentence(self):
        """Test splitting text with a single sentence."""
        text = "This is a single sentence without ending punctuation"
        sentences = split_text_into_sentences(text=text)

        assert len(sentences) == 1
        assert sentences[0] == text

    def test_create_sentence_html_no_highlight(self):
        """Test creating HTML for a non-highlighted sentence."""
        sentence = "This is a test sentence."
        html = create_sentence_html(sentence=sentence, is_highlighted=False)

        assert "background-color" not in html
        assert sentence in html
        assert "display:block" in html

    def test_create_sentence_html_with_highlight(self):
        """Test creating HTML for a highlighted sentence."""
        sentence = "This is a test sentence."
        color = "#FFD580"
        html = create_sentence_html(sentence=sentence, is_highlighted=True, color=color)

        assert f"background-color:{color}" in html
        assert "color:black" in html
        assert sentence in html
        assert "display:block" in html


class TestModelCreation:
    """Tests for model creation utilities."""

    def test_create_intervention_from_series(self):
        """Test creating an Intervention from a pandas Series."""
        series = pd.Series(
            {
                "story_id": "story1",
                "skill_id": "skill1",
                "sentence": "This is a test sentence.",
                "intervention_type": "metacognitive",
                "intervention": "Think about this concept.",
                "explanation": "This helps understand the concept.",
                "score": 0.85,
            }
        )

        intervention = create_intervention_from_series(row=series)

        assert isinstance(intervention, Intervention)
        assert intervention.story_id == "story1"
        assert intervention.skill_id == "skill1"
        assert intervention.sentence == "This is a test sentence."
        assert intervention.intervention_type == "metacognitive"
        assert intervention.intervention == "Think about this concept."
        assert intervention.explanation == "This helps understand the concept."
        assert intervention.score == 0.85

    def test_create_intervention_feedback(self, sample_intervention):
        """Test creating an InterventionFeedback object."""
        feedback = create_intervention_feedback(
            intervention=sample_intervention,
            intervention_key="story1_skill1_0",
            story_title="Test Story 1",
            helpfulness="👍 Helpful",
            feedback_comment="Good intervention",
            fit_agreement="Yes",
        )

        assert feedback.story_id == "story1"
        assert feedback.story_title == "Test Story 1"
        assert feedback.intervention_id == "story1_skill1_0"
        assert feedback.sentence == "It has multiple sentences."
        assert feedback.skill_id == "skill1"
        assert feedback.intervention_type == "metacognitive"
        assert feedback.intervention == "Think about what this means."
        assert feedback.feedback == "positive"
        assert feedback.comment == "Good intervention"
        assert feedback.fit_agreement == "Yes"
        assert feedback.timestamp is not None

    def test_create_intervention_feedback_not_helpful(self, sample_intervention):
        """Test creating an InterventionFeedback object with 'not helpful' rating."""
        feedback = create_intervention_feedback(
            intervention=sample_intervention,
            intervention_key="story1_skill1_0",
            story_title="Test Story 1",
            helpfulness="👎 Not Helpful",
            feedback_comment="Not a good intervention",
            fit_agreement="No",
        )

        assert feedback.feedback == "negative"
        assert feedback.comment == "Not a good intervention"
        assert feedback.fit_agreement == "No"

    def test_create_skill_match_from_series_minimal(self):
        """Test creating a SkillMatch from a pandas Series with minimal data."""
        series = pd.Series(
            {
                "story_id": "story1",
                "skill_id": "skill1",
                "parent_score": 0.82,
                "child_score": 0.89,
            }
        )

        skill_match = create_skill_match_from_series(row=series)

        assert isinstance(skill_match, SkillMatch)
        assert skill_match.story_id == "story1"
        assert skill_match.skill_id == "skill1"
        assert skill_match.parent_score == 0.82
        assert skill_match.child_score == 0.89
        assert skill_match.method is None
        assert skill_match.parent_explanation is None
        assert skill_match.child_explanation is None

    def test_create_skill_match_from_series_complete(self):
        """Test creating a SkillMatch from a pandas Series with complete data."""
        series = pd.Series(
            {
                "story_id": "story1",
                "skill_id": "skill1",
                "parent_score": 0.82,
                "child_score": 0.89,
                "method": "embedding",
                "parent_explanation": "The story relates strongly to this parent skill.",
                "child_explanation": "The story demonstrates this specific skill.",
            }
        )

        skill_match = create_skill_match_from_series(row=series)

        assert isinstance(skill_match, SkillMatch)
        assert skill_match.story_id == "story1"
        assert skill_match.skill_id == "skill1"
        assert skill_match.parent_score == 0.82
        assert skill_match.child_score == 0.89
        assert skill_match.method == "embedding"
        assert (
            skill_match.parent_explanation
            == "The story relates strongly to this parent skill."
        )
        assert (
            skill_match.child_explanation
            == "The story demonstrates this specific skill."
        )

    def test_create_skill_match_feedback(self, sample_skill_match):
        """Test creating a SkillMatchFeedback object."""
        feedback = create_skill_match_feedback(
            match=sample_skill_match,
            story_title="Test Story 1",
            accuracy="👍 Accurate",
            feedback_comment="Good match",
        )

        assert feedback.story_id == "story1"
        assert feedback.story_title == "Test Story 1"
        assert feedback.skill_id == "skill1"
        assert feedback.parent_score == 0.82
        assert feedback.child_score == 0.89
        assert feedback.method == "embedding"
        assert feedback.accuracy == "👍 Accurate"
        assert feedback.comment == "Good match"
        assert feedback.timestamp is not None
