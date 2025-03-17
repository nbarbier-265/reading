"""
Data models for the Streamlit application.

This module defines the data models used in the Streamlit application
for representing stories, interventions, and user feedback.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Story:
    """Represents a story with its metadata and content."""

    story_id: str
    title: str
    story_text: str

    @property
    def display_name(self) -> str:
        """Returns a formatted display name for the story."""
        return f"{self.title} (ID: {self.story_id})"


@dataclass
class Intervention:
    """Represents an intervention point in a story."""

    story_id: str
    skill_id: str
    sentence: str
    intervention_type: str
    intervention: str
    explanation: str
    score: float

    def create_key(self, *, idx: int) -> str:
        """
        Creates a unique key for this intervention.

        Args:
            idx: Index of the intervention.

        Returns:
            A unique string identifier.
        """
        return f"{self.story_id}_{self.skill_id}_{idx}"


@dataclass
class InterventionFeedback:
    """Represents user feedback on an intervention."""

    story_id: str
    story_title: str
    intervention_id: str
    sentence: str
    skill_id: str
    intervention_type: str
    intervention: str
    feedback: str
    comment: str
    fit_agreement: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SkillMatch:
    """Represents a skill match for a story."""

    story_id: str
    skill_id: str
    parent_score: float
    child_score: float
    method: str | None = None
    parent_explanation: str | None = None
    child_explanation: str | None = None


from enum import Enum


class FeedbackRating(Enum):
    """Represents a feedback rating."""

    VERY_HELPFUL = "Very helpful"
    SOMEWHAT_HELPFUL = "Somewhat helpful"
    NOT_HELPFUL = "Not helpful"


@dataclass
class SkillMatchFeedback:
    """Represents user feedback on a skill match."""

    story_id: str
    story_title: str
    skill_id: str
    parent_score: float
    child_score: float
    accuracy: str
    comment: str
    method: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
