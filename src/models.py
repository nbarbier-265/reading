"""
Models for the project.

This module defines the data models used throughout the project. It includes models for structured LLM responses,
and data models for the project.

- SkillScores: Represents the scores for a skill (parent or child)
- ReadingLevel: Represents the reading level of a story
- ChildSkill: Represents a child skill with its ID and description
- SkillMatchResult: Represents a skill match result with its story ID, title, skill ID and scores

"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from enum import Enum
from typing import Any
import pandas as pd

from config import PATH_TO_PROJECT

skill_ids: list[str] = pd.read_csv(f"{PATH_TO_PROJECT}/data/source_data/skills.csv")[
    "skill_id"
].tolist()
skill_ids_parent: list[str] = [
    skill_id.split(".", 1)[0] for skill_id in skill_ids if "." in skill_id
]
skill_ids_parent = list(set(skill_ids_parent))
skill_ids_child: list[str] = skill_ids
skill_ids_all: list[str] = skill_ids_parent + skill_ids_child


class SkillScores(BaseModel):
    """Scores for a skill (parent or child)."""

    parent_score: float = Field(
        ..., description="Parent score between 0 and 1 indicating relevance"
    )
    child_score: float = Field(
        ..., description="Child score between 0 and 1 indicating relevance"
    )
    reading_level: int = Field(..., description="Reading level from K through 12")
    parent_explanation: str = Field(
        ..., description="Brief explanation for the parent score"
    )
    child_explanation: str = Field(
        ..., description="Brief explanation for the child score"
    )


class SingleSkillScore(BaseModel):
    """Score for a single skill."""

    skill_id: str = Field(..., description="The skill ID")
    score: float = Field(
        ..., description="Score between 0 and 1 indicating relevance", ge=0, le=1
    )
    explanation: str = Field(
        ..., description="Brief explanation for the assigned score"
    )


class ReadingLevel(BaseModel):
    level: int = Field(
        ...,
        description="Reading level from 0 (Kindergarten) to 12 (12th grade)",
        ge=0,
        le=12,
    )


@dataclass(frozen=True)
class ChildSkill:
    """Represents a child skill with its ID and description."""

    skill_id: str
    description: str

    def __post_init__(self) -> None:
        """Validate that skill_id is in the list of valid skill IDs."""
        if self.skill_id not in skill_ids_child:
            raise ValueError(
                f"Invalid skill_id: {self.skill_id}. Must be one of the valid skill IDs."
            )


@dataclass
class SkillMatchResult:
    """Represents a skill match result with its story ID, title, skill ID and scores."""

    story_id: str
    story_title: str
    skill_id: str
    scores: SkillScores
    parent_score: float = field(init=False)
    child_score: float = field(init=False)
    reading_level: int = field(init=False)
    parent_explanation: str = field(init=False)
    child_explanation: str = field(init=False)

    def __post_init__(self) -> None:
        """Extract scores from the scores dictionary and validate skill_id."""
        if self.skill_id not in skill_ids_all:
            raise ValueError(
                f"Invalid skill_id: {self.skill_id}. Must be one of the valid skill IDs."
            )

        self.parent_score = self.scores.parent_score
        self.child_score = self.scores.child_score
        self.reading_level = self.scores.reading_level
        self.parent_explanation = self.scores.parent_explanation
        self.child_explanation = self.scores.child_explanation


@dataclass
class StoryTask:
    """Represents a story processing task."""

    story_id: str
    title: str
    story_text: str


@dataclass
class BatchCount:
    """Represents batch processing counts."""

    total: int
    to_process: int


@dataclass
class EmbeddingResult:
    """Represents a standardized embedding result."""

    story_id: str
    story_title: str
    skill_id: str
    parent_score: float
    child_score: float
    method: str = "embedding"


from pydantic import BaseModel, Field, validator
from typing import TypedDict


class RelevantTextResponse(BaseModel):
    """Response model for relevant text extraction from chunks."""

    text: str = Field(
        ...,
        description="Extracted text segments demonstrating reading comprehension skills",
    )

    @validator("text")
    def validate_text_length(cls, v: str) -> str:
        char_count: int = len(v)
        word_count: int = len(v.split())
        if not (20 <= word_count <= 400):
            raise ValueError(
                f"Text must be between 20-400 words (currently {word_count})"
            )
        if not (150 <= char_count <= 4000):
            raise ValueError(
                f"Text must be between 150-4000 characters (currently {char_count})"
            )
        return v


class SkillMatch(TypedDict):
    skill_id: str
    skill_name: str
    score: float
    confidence: float


class TextSnippetResult(TypedDict):
    text: str
    skill_matches: list[SkillMatch]


class ChunkResult(TypedDict):
    chunk_index: int
    token_start: int
    token_end: int
    text_snippets: list[TextSnippetResult]


class InterventionType(str, Enum):
    METACOGNITIVE = "metacognitive"
    CONCEPTUAL = "conceptual"
    APPLICATION = "application"
    VOCABULARY = "vocabulary"


class InterventionPointIdentification(BaseModel):
    """Identified intervention point in a text."""

    sentence: str = Field(
        ..., description="The sentence that serves as an intervention point"
    )
    position: int = Field(
        ..., description="The position of the sentence in the text (0-indexed)"
    )
    skill_id: str = Field(
        ...,
        description="The ID of the skill that the intervention is targeting",
        enum=skill_ids_all,
    )
    intervention: str = Field(..., description="The specific intervention to be used")
    intervention_type: InterventionType = Field(
        ...,
        description="The type of intervention (metacognitive, conceptual, application, or vocabulary)",
    )
    explanation: str = Field(
        ..., description="Explanation for why this is a good intervention point"
    )
    score: float = Field(
        ...,
        description="Score between 0 and 1 indicating strength as an intervention point",
        ge=0,
        le=1,
    )

    def post_init(self, __context: Any) -> None:
        """Validate that sentence positions are within the range of the text."""
        if self.position is not None and self.sentence is not None:
            sentences = self.sentence.split(".")
            if self.position < 0 or self.position >= len(sentences):
                raise ValueError(
                    f"Position {self.position} is out of range for the text with {len(sentences)} sentences"
                )

    def validate_sentence_in_text(self, text: str) -> None:
        """Validate that the sentence is actually a substring of the text.

        This method should be called after creating an InterventionPointIdentification
        instance to ensure the sentence exists in the original text.

        Args:
            text: The original text to check against

        Raises:
            ValueError: If the sentence is not found in the text
        """
        """Validate that the sentence is actually a substring of the text."""
        if self.sentence and text:
            if self.sentence not in text:
                raise ValueError(f"Sentence '{self.sentence}' is not found in the text")


class InterventionPointSet(BaseModel):
    """Set of intervention points for a text."""

    story_id: str = Field(..., description="ID of the story")
    intervention_points: list[InterventionPointIdentification] = Field(
        ..., description="List of identified intervention points"
    )
