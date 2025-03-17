"""
Unit tests for the app/models.py module.
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.append(str(app_dir))

from models import (
    Story,
    Intervention,
    InterventionFeedback,
    SkillMatch,
    SkillMatchFeedback,
)


class TestStory:
    """Tests for the Story class."""

    def test_init(self):
        """Test Story initialization."""
        story = Story(
            story_id="test123", title="Test Story", story_text="This is a test story."
        )

        assert story.story_id == "test123"
        assert story.title == "Test Story"
        assert story.story_text == "This is a test story."

    def test_display_name(self):
        """Test the display_name property."""
        story = Story(
            story_id="test123", title="Test Story", story_text="This is a test story."
        )

        assert story.display_name == "Test Story (ID: test123)"


class TestIntervention:
    """Tests for the Intervention class."""

    def test_init(self):
        """Test Intervention initialization."""
        intervention = Intervention(
            story_id="test123",
            skill_id="skill456",
            sentence="This is a test sentence.",
            intervention_type="metacognitive",
            intervention="Think about this concept.",
            explanation="This helps understand the concept.",
            score=0.85,
        )

        assert intervention.story_id == "test123"
        assert intervention.skill_id == "skill456"
        assert intervention.sentence == "This is a test sentence."
        assert intervention.intervention_type == "metacognitive"
        assert intervention.intervention == "Think about this concept."
        assert intervention.explanation == "This helps understand the concept."
        assert intervention.score == 0.85

    def test_create_key(self):
        """Test the create_key method."""
        intervention = Intervention(
            story_id="test123",
            skill_id="skill456",
            sentence="This is a test sentence.",
            intervention_type="metacognitive",
            intervention="Think about this concept.",
            explanation="This helps understand the concept.",
            score=0.85,
        )

        # Test with different indices
        assert intervention.create_key(idx=0) == "test123_skill456_0"
        assert intervention.create_key(idx=1) == "test123_skill456_1"
        assert intervention.create_key(idx=999) == "test123_skill456_999"


class TestInterventionFeedback:
    """Tests for the InterventionFeedback class."""

    def test_init(self):
        """Test InterventionFeedback initialization."""
        feedback = InterventionFeedback(
            story_id="test123",
            story_title="Test Story",
            intervention_id="test123_skill456_0",
            sentence="This is a test sentence.",
            skill_id="skill456",
            intervention_type="metacognitive",
            intervention="Think about this concept.",
            feedback="positive",
            comment="Great intervention!",
            fit_agreement="Yes",
        )

        assert feedback.story_id == "test123"
        assert feedback.story_title == "Test Story"
        assert feedback.intervention_id == "test123_skill456_0"
        assert feedback.sentence == "This is a test sentence."
        assert feedback.skill_id == "skill456"
        assert feedback.intervention_type == "metacognitive"
        assert feedback.intervention == "Think about this concept."
        assert feedback.feedback == "positive"
        assert feedback.comment == "Great intervention!"
        assert feedback.fit_agreement == "Yes"

        # Check that timestamp is generated automatically
        assert feedback.timestamp
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(feedback.timestamp)

    def test_init_with_timestamp(self):
        """Test InterventionFeedback initialization with a provided timestamp."""
        feedback = InterventionFeedback(
            story_id="test123",
            story_title="Test Story",
            intervention_id="test123_skill456_0",
            sentence="This is a test sentence.",
            skill_id="skill456",
            intervention_type="metacognitive",
            intervention="Think about this concept.",
            feedback="positive",
            comment="Great intervention!",
            fit_agreement="Yes",
            timestamp="2023-01-01T12:00:00",
        )

        assert feedback.timestamp == "2023-01-01T12:00:00"


class TestSkillMatch:
    """Tests for the SkillMatch class."""

    def test_init_minimal(self):
        """Test SkillMatch initialization with minimal parameters."""
        skill_match = SkillMatch(
            story_id="test123", skill_id="skill456", parent_score=0.82, child_score=0.89
        )

        assert skill_match.story_id == "test123"
        assert skill_match.skill_id == "skill456"
        assert skill_match.parent_score == 0.82
        assert skill_match.child_score == 0.89
        assert skill_match.method is None
        assert skill_match.parent_explanation is None
        assert skill_match.child_explanation is None

    def test_init_complete(self):
        """Test SkillMatch initialization with all parameters."""
        skill_match = SkillMatch(
            story_id="test123",
            skill_id="skill456",
            parent_score=0.82,
            child_score=0.89,
            method="embedding",
            parent_explanation="Good match for parent skill",
            child_explanation="Excellent match for child skill",
        )

        assert skill_match.story_id == "test123"
        assert skill_match.skill_id == "skill456"
        assert skill_match.parent_score == 0.82
        assert skill_match.child_score == 0.89
        assert skill_match.method == "embedding"
        assert skill_match.parent_explanation == "Good match for parent skill"
        assert skill_match.child_explanation == "Excellent match for child skill"


class TestSkillMatchFeedback:
    """Tests for the SkillMatchFeedback class."""

    def test_init(self):
        """Test SkillMatchFeedback initialization."""
        feedback = SkillMatchFeedback(
            story_id="test123",
            story_title="Test Story",
            skill_id="skill456",
            parent_score=0.82,
            child_score=0.89,
            accuracy="👍 Accurate",
            comment="Great match!",
            method="embedding",
        )

        assert feedback.story_id == "test123"
        assert feedback.story_title == "Test Story"
        assert feedback.skill_id == "skill456"
        assert feedback.parent_score == 0.82
        assert feedback.child_score == 0.89
        assert feedback.accuracy == "👍 Accurate"
        assert feedback.comment == "Great match!"
        assert feedback.method == "embedding"

        # Check that timestamp is generated automatically
        assert feedback.timestamp
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(feedback.timestamp)

    def test_init_with_timestamp(self):
        """Test SkillMatchFeedback initialization with a provided timestamp."""
        feedback = SkillMatchFeedback(
            story_id="test123",
            story_title="Test Story",
            skill_id="skill456",
            parent_score=0.82,
            child_score=0.89,
            accuracy="👍 Accurate",
            comment="Great match!",
            method="embedding",
            timestamp="2023-01-01T12:00:00",
        )

        assert feedback.timestamp == "2023-01-01T12:00:00"
