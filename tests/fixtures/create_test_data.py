"""
Script to generate sample test data files for the test fixtures.

This script creates sample CSV and JSON files that can be used for testing.
Run this script to regenerate test data files after making changes to the data model.
"""

import os
import pandas as pd
import json
from pathlib import Path

# Define the fixtures directory
FIXTURES_DIR = Path(__file__).parent
DATA_DIR = FIXTURES_DIR / "data"
SOURCE_DATA_DIR = DATA_DIR / "source_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

# Ensure directories exist
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def create_sample_stories():
    """Create a sample stories DataFrame and save as CSV."""
    stories_df = pd.DataFrame(
        {
            "story_id": ["story1", "story2", "story3"],
            "title": ["Test Story 1", "Test Story 2", "Test Story 3"],
            "story_text": [
                "This is the first test story. It contains multiple sentences. Each sentence can be processed separately.",
                "This is the second test story. It has fewer sentences but is still useful for testing.",
                "This is the third test story with some content for testing.",
            ],
        }
    )

    # Save to CSV
    stories_df.to_csv(SOURCE_DATA_DIR / "stories.csv", index=False)
    stories_df.to_csv(PROCESSED_DATA_DIR / "stories.csv", index=False)

    return stories_df


def create_sample_interventions(stories_df):
    """Create a sample interventions DataFrame and save as CSV."""
    interventions_df = pd.DataFrame(
        {
            "story_id": ["story1", "story1", "story2", "story2"],
            "skill_id": ["skill1", "skill2", "skill1", "skill3"],
            "sentence": [
                "It contains multiple sentences.",
                "Each sentence can be processed separately.",
                "It has fewer sentences but is still useful for testing.",
                "This is the second test story.",
            ],
            "intervention_type": [
                "metacognitive",
                "conceptual",
                "vocabulary",
                "question",
            ],
            "intervention": [
                "Consider how the character feels in this situation.",
                "What do you think happens next in the story?",
                "This part of the story introduces a key theme.",
                "How would you describe the character's motivation?",
            ],
            "explanation": [
                "This intervention helps readers understand the character's emotions.",
                "This intervention encourages prediction skills.",
                "This intervention helps identify themes in the story.",
                "This intervention focuses on character analysis.",
            ],
            "score": [0.85, 0.92, 0.78, 0.88],
        }
    )

    # Save to CSV with the filename expected by the app
    interventions_df.to_csv(
        PROCESSED_DATA_DIR / "intervention_points_llm.csv", index=False
    )

    return interventions_df


def create_sample_skill_matches(stories_df):
    """Create a sample skill matches DataFrame and save as CSV."""
    skill_match_df = pd.DataFrame(
        {
            "story_id": ["story1", "story1", "story2", "story2"],
            "skill_id": ["skill1", "skill2", "skill1", "skill3"],
            "parent_score": [0.82, 0.75, 0.79, 0.88],
            "child_score": [0.89, 0.81, 0.85, 0.92],
            "method": ["embedding", "embedding", "embedding", "embedding"],
            "parent_explanation": [
                "The story relates strongly to this parent skill.",
                "The story somewhat relates to this parent skill.",
                "The story relates moderately to this parent skill.",
                "The story relates very strongly to this parent skill.",
            ],
            "child_explanation": [
                "The story demonstrates this specific skill.",
                "The story somewhat demonstrates this specific skill.",
                "The story moderately demonstrates this specific skill.",
                "The story clearly demonstrates this specific skill.",
            ],
        }
    )

    # Save to CSV with the filename expected by the app
    skill_match_df.to_csv(PROCESSED_DATA_DIR / "skill_match_results.csv", index=False)

    return skill_match_df


def create_sample_feedback_files():
    """Create sample feedback JSON files."""
    # Intervention feedback
    intervention_feedback = [
        {
            "story_id": "story1",
            "story_title": "Test Story 1",
            "intervention_id": "int1",
            "text": "Consider how the character feels in this situation.",
            "type": "suggestion",
            "helpfulness": "👍 Helpful",
            "comment": "Good suggestion",
            "timestamp": "2023-01-01T12:00:00",
        },
        {
            "story_id": "story2",
            "story_title": "Test Story 2",
            "intervention_id": "int3",
            "text": "This part of the story introduces a key theme.",
            "type": "information",
            "helpfulness": "👎 Not Helpful",
            "comment": "Could be more specific",
            "timestamp": "2023-01-02T14:30:00",
        },
    ]

    # Skill match feedback
    skill_match_feedback = [
        {
            "story_id": "story1",
            "story_title": "Test Story 1",
            "skill_id": "skill1",
            "parent_score": 0.82,
            "child_score": 0.89,
            "accuracy": "👍 Accurate",
            "comment": "Good match",
            "method": "embedding",
            "timestamp": "2023-01-01T12:30:00",
        },
        {
            "story_id": "story2",
            "story_title": "Test Story 2",
            "skill_id": "skill3",
            "parent_score": 0.88,
            "child_score": 0.92,
            "accuracy": "👎 Not Accurate",
            "comment": "Not a good match",
            "method": "embedding",
            "timestamp": "2023-01-02T15:00:00",
        },
    ]

    # Save JSON files
    with open(PROCESSED_DATA_DIR / "intervention_feedback.json", "w") as f:
        json.dump(intervention_feedback, f, indent=2)

    with open(PROCESSED_DATA_DIR / "skill_match_feedback.json", "w") as f:
        json.dump(skill_match_feedback, f, indent=2)

    return intervention_feedback, skill_match_feedback


def main():
    """Generate all sample test data files."""
    print("Generating sample test data files...")

    # Create DataFrames and CSV files
    stories_df = create_sample_stories()
    interventions_df = create_sample_interventions(stories_df)
    skill_match_df = create_sample_skill_matches(stories_df)

    # Create feedback JSON files
    intervention_feedback, skill_match_feedback = create_sample_feedback_files()

    print(f"Sample test data files generated in {DATA_DIR}")
    print(f"  - Source data directory: {SOURCE_DATA_DIR}")
    print(f"  - Processed data directory: {PROCESSED_DATA_DIR}")
    print("Files created:")
    print("  - stories.csv")
    print("  - intervention_points_llm.csv")
    print("  - skill_match_results.csv")
    print("  - intervention_feedback.json")
    print("  - skill_match_feedback.json")


if __name__ == "__main__":
    main()
