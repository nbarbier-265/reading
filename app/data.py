"""
Data access module for the Streamlit application.

This module handles loading data from CSV files and saving feedback to JSON files.
"""

import json
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple
import logging
import os

from models import InterventionFeedback, SkillMatchFeedback
from pathlib import Path

from config import PATH_TO_PROCESSED_DATA, PATH_TO_SOURCE_DATA, get_processed_data_path


@st.cache_data
def load_data(
    prompt_version: str = "v1",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads story, intervention point, and skill match data from CSV files.

    Args:
        prompt_version (str): The prompt version to load data for. Defaults to "v1".

    Returns:
        Tuple containing stories DataFrame, intervention points DataFrame, and skill match results DataFrame.
        The stories DataFrame will be None if loading fails. The other DataFrames will be None if their
        respective files are missing, but this is not considered an error.
    """
    try:
        # Source data is the same regardless of prompt version
        stories_path: Path = Path(f"{PATH_TO_SOURCE_DATA}/stories.csv")
        stories_df: pd.DataFrame = pd.read_csv(stories_path)

        # Get the processed data path for the specific prompt version
        processed_data_path = get_processed_data_path(prompt_version)

        intervention_points_df: pd.DataFrame | None = None
        try:
            intervention_points_path: Path = Path(
                f"{processed_data_path}/intervention_points.csv"
            )
            if intervention_points_path.exists():
                from loguru import logger

                logger.info(
                    f"Loading intervention points data from {intervention_points_path}"
                )
                intervention_points_df = pd.read_csv(intervention_points_path)
            else:
                logger.error(
                    f"Intervention points data not found at {intervention_points_path}"
                )
        except Exception:
            pass

        skill_match_df: pd.DataFrame | None = None
        try:
            skill_match_path: Path = Path(
                f"{processed_data_path}/skill_match_results.csv"
            )
            if skill_match_path.exists():
                skill_match_df: pd.DataFrame = pd.read_csv(skill_match_path)
            else:
                logger.error(f"Skill match data not found at {skill_match_path}")
        except Exception:
            pass

        return stories_df, intervention_points_df, skill_match_df
    except Exception as e:
        st.error(f"Error loading stories data: {e}")
        return None, None, None


def create_story_options(*, stories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame with story display options.

    Args:
        stories_df: DataFrame containing story data.

    Returns:
        DataFrame with story display options.
    """
    story_options: pd.DataFrame = stories_df[["story_id", "title"]].drop_duplicates()
    story_options["display"] = (
        story_options["title"] + " (ID: " + story_options["story_id"].astype(str) + ")"
    )
    return story_options


def get_story_details(*, stories_df: pd.DataFrame, story_id: str) -> Tuple[str, str]:
    """
    Retrieves story text and title for a given story ID.

    Args:
        stories_df: DataFrame containing story data.
        story_id: ID of the story to retrieve.

    Returns:
        Tuple containing story text and story title.
    """
    story_data: pd.Series = stories_df[stories_df["story_id"] == story_id].iloc[0]
    return story_data["story_text"], story_data["title"]


def filter_interventions(
    *, interventions_df: pd.DataFrame, story_id: str, min_score: float = 0.0
) -> pd.DataFrame:
    """
    Filters interventions by story ID and minimum score.

    Args:
        interventions_df: DataFrame containing intervention data.
        story_id: ID of the story to filter by.
        min_score: Minimum score threshold (default: 0.0).

    Returns:
        Filtered DataFrame.
    """
    story_interventions: pd.DataFrame = interventions_df[
        interventions_df["story_id"] == story_id
    ]
    return story_interventions[story_interventions["score"] >= min_score]


def save_feedback_to_file(
    feedback_data: dict, file_path: Path, allow_duplicates: bool = False
) -> None:
    """
    Save feedback data to a JSON file.

    Args:
        feedback_data: Dictionary containing feedback data.
        file_path: Path to the JSON file.
        allow_duplicates: Whether to allow duplicate feedback entries.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w") as f:
                json.dump([], f)

        with open(file_path, "r") as f:
            try:
                existing_feedback: list[dict] = json.load(f)
            except json.JSONDecodeError:
                existing_feedback: list[dict] = []

        if not allow_duplicates:
            for entry in existing_feedback:
                if all(
                    entry.get(k) == v
                    for k, v in feedback_data.items()
                    if k != "timestamp"
                ):
                    logging.info(
                        f"Duplicate feedback entry found, not saving: {feedback_data}"
                    )
                    return

        existing_feedback.append(feedback_data)

        with open(file_path, "w") as f:
            json.dump(existing_feedback, f, indent=2)

        logging.info(f"Feedback saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving feedback to {file_path}: {e}")
        raise


def save_feedback(
    *, feedback: InterventionFeedback, prompt_version: str = "v1"
) -> None:
    """
    Save intervention feedback to a file.

    Args:
        feedback: Intervention feedback object.
        prompt_version: The prompt version to save feedback for. Defaults to "v1".
    """
    from dataclasses import asdict

    feedback_data = asdict(feedback)

    # Get the processed data path for the specific prompt version
    processed_data_path = get_processed_data_path(prompt_version)
    os.makedirs(processed_data_path, exist_ok=True)

    file_path = Path(f"{processed_data_path}/intervention_feedback.json")
    save_feedback_to_file(feedback_data=feedback_data, file_path=file_path)


def save_skill_match_feedback(
    *, feedback: SkillMatchFeedback, prompt_version: str = "v1"
) -> None:
    """
    Save skill match feedback to a file.

    Args:
        feedback: Skill match feedback object.
        prompt_version: The prompt version to save feedback for. Defaults to "v1".
    """
    from dataclasses import asdict

    feedback_data = asdict(feedback)

    # Get the processed data path for the specific prompt version
    processed_data_path = get_processed_data_path(prompt_version)
    os.makedirs(processed_data_path, exist_ok=True)

    file_path = Path(f"{processed_data_path}/skill_match_feedback.json")
    save_feedback_to_file(feedback_data=feedback_data, file_path=file_path)
