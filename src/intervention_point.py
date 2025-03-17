#!/usr/bin/env python3
"""
Pedagogical Intervention Point Identifier using LLMs

This module identifies potential pedagogical intervention points in texts based on learning
and literacy science principles. It analyzes texts in relation to their matched skill areas
and identifies moments where educational interventions would be most effective.

Instead of hardcoded logic, this version uses Large Language Models to analyze texts
and identify intervention points based on educational theory. It augments the LLM with
pedagogical theory based in Piaget's theories and developed by modern educational
research promoting four types of interventions:

1. METACOGNITIVE - Interventions that help students reflect on and monitor their own thinking and learning processes
2. CONCEPTUAL - Interventions that develop deeper understanding of core ideas and relationships
3. APPLICATION - Interventions that help students apply knowledge to real-world contexts
4. VOCABULARY - Interventions that build word knowledge and language comprehension
"""

import os
import time
from typing import Any, Optional
import importlib.util
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.models import InterventionPointSet, InterventionType
from src.theories import THEORY_DESCRIPTIONS_COMPLEX

THEORY_DESCRIPTIONS = THEORY_DESCRIPTIONS_COMPLEX
from config import (
    PATH_TO_PROCESSED_DATA,
    PATH_TO_SOURCE_DATA,
    get_prompt_version_path,
    get_processed_data_path,
)

from loguru import logger


class InterventionPoint:
    """
    Identifies potential pedagogical intervention points in texts using LLMs.

    This class analyzes texts based on their matched skill areas and identifies
    specific sentences or passages where educational interventions would be
    most effective for learning.
    """

    def __init__(self, model_name="gpt-4", temperature=0.2, prompt_version="v1"):
        """
        Initialize the intervention point generator.

        Args:
            model_name (str): The LLM model to use for generating interventions
            temperature (float): The temperature setting for the LLM
            prompt_version (str): Version of prompts to use (defaults to v1)
        """
        load_dotenv()

        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.prompt_version = prompt_version

        # Initialize client
        # Initialize the OpenAI client with instructor for response validation
        import instructor

        self.client = instructor.patch(OpenAI(api_key=self.openai_api_key))

        # Data for processing
        self.processed_data_path = get_processed_data_path(prompt_version)
        os.makedirs(self.processed_data_path, exist_ok=True)

        self.results_path = f"{self.processed_data_path}/skill_match_results.csv"
        self.stories_path = f"{PATH_TO_SOURCE_DATA}/stories.csv"
        self.results_df: pd.DataFrame | None = None
        self.stories_df: pd.DataFrame | None = None
        self.top_skills: dict[str, pd.DataFrame] = {}

        # Initialize logger
        self.logger = logger

        # Load the prompt module
        self.prompt_module = self._load_prompt_module()

        # Use the client to check if we can connect to OpenAI
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )

        self.logger.info(
            f"Initialized InterventionPoint generator with model: {model_name}, temperature: {temperature}, prompt version: {prompt_version}"
        )

    def _load_prompt_module(self):
        """
        Load the appropriate prompt module based on the prompt version.
        """
        prompt_path = get_prompt_version_path(self.prompt_version)
        module_path = os.path.join(prompt_path, "intervention_point.py")

        if not os.path.exists(module_path):
            logger.warning(
                f"Prompt module not found at {module_path}, using default prompts"
            )
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                "intervention_point_prompts", module_path
            )
            if spec is None:
                logger.error(f"Could not create spec for module at {module_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            if module is None:
                logger.error(f"Could not create module from spec for {module_path}")
                return None

            sys.modules["intervention_point_prompts"] = module
            spec.loader.exec_module(module)

            # Verify the module has the required function
            if not hasattr(module, "get_intervention_prompt"):
                logger.error(
                    f"Module at {module_path} does not contain get_intervention_prompt function"
                )
                return None

            logger.info(f"Successfully loaded prompt module from {module_path}")
            return module
        except Exception as e:
            logger.error(f"Error loading prompt module: {e}")
            return None

    def load_data(self) -> None:
        """
        Load skill match results and stories data.

        Raises:
            FileNotFoundError: If either file doesn't exist
        """
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        if not os.path.exists(self.stories_path):
            raise FileNotFoundError(f"Stories file not found: {self.stories_path}")

        logger.info(f"Loading skill match results from {self.results_path}")
        self.results_df = pd.read_csv(self.results_path)

        logger.info(f"Loading stories from {self.stories_path}")
        self.stories_df = pd.read_csv(self.stories_path)

        self._extract_top_skills()

    def _extract_top_skills(self) -> None:
        """
        Extract the top 5 skills for each story based on weighted score.
        """
        for story_id in self.results_df["story_id"].unique():
            story_results = self.results_df[self.results_df["story_id"] == story_id]
            top_skills = story_results.nlargest(5, "child_score")
            self.top_skills[story_id] = top_skills

        logger.info(f"Extracted top skills for {len(self.top_skills)} stories")

    def identify_intervention_points_with_llm(
        self, *, story_id: str
    ) -> list[dict[str, Any]]:
        """
        Use a direct LLM call to identify intervention points in a single API call.

        This method processes the entire story with its top skills in one LLM call,
        allowing the model to analyze the complete context and identify the best
        intervention points of each type.

        Args:
            story_id: ID of the story to analyze

        Returns:
            List of dictionaries containing intervention points
        """
        if not self._validate_story_exists(story_id):
            return []

        story_data: dict[str, str] = self._get_story_data(story_id)
        skills_info: list[dict[str, Any]] = self._prepare_skills_info(story_id)
        reading_level: int = self._get_reading_level(story_id)

        prompt: str = self._build_intervention_prompt(story_id, story_data, skills_info)

        return self._execute_llm_intervention_request(prompt, story_data["story_text"])

    def _validate_story_exists(self, story_id: str) -> bool:
        """
        Validate that the story exists and has top skills.

        Args:
            story_id: ID of the story to validate

        Returns:
            True if story exists and has top skills, False otherwise
        """
        if story_id not in self.top_skills:
            logger.warning(f"No top skills found for story {story_id}")
            return False

        story_row: pd.Series | None = self.stories_df[
            self.stories_df["story_id"] == story_id
        ].iloc[0]
        if story_row is None:
            logger.warning(f"Story not found: {story_id}")
            return False

        return True

    def _get_story_data(self, story_id: str) -> dict[str, str]:
        """
        Get story text and title for the given story ID.

        Args:
            story_id: ID of the story

        Returns:
            Dictionary with story_text and title
        """
        story_row: pd.Series | None = self.stories_df[
            self.stories_df["story_id"] == story_id
        ].iloc[0]
        if story_row is None:
            logger.warning(f"Story not found: {story_id}")
            return {}

        return {"story_text": story_row["story_text"], "title": story_row["title"]}

    def _prepare_skills_info(self, story_id: str) -> list[dict[str, Any]]:
        """
        Prepare information about the top skills for a story.

        Args:
            story_id: ID of the story

        Returns:
            List of dictionaries with skill information
        """
        skills_info = []
        top_skills_df = self.top_skills[story_id]

        for _, skill_row in top_skills_df.iterrows():
            skill_id = skill_row["skill_id"]
            child_score = skill_row["child_score"]
            skills_info.append(
                {
                    "skill_id": skill_id,
                    "child_score": child_score,
                    "explanation": skill_row["explanation"],
                }
            )

        return skills_info

    def _get_reading_level(self, story_id: str) -> int:
        """
        Get the reading level for a story.

        Args:
            story_id: ID of the story

        Returns:
            Integer reading level between 1 and 5
        """
        # For now, return a default value
        # This could be enhanced in future versions to infer reading level from the text
        return 3

    def _build_intervention_prompt(
        self,
        story_id: str,
        story_data: dict[str, str],
        skills_info: list[dict[str, Any]],
    ) -> str:
        """
        Build the prompt for the LLM to identify intervention points.

        Args:
            story_id: ID of the story
            story_data: Dictionary with story text and title
            skills_info: List of dictionaries with skill information

        Returns:
            Formatted prompt string
        """
        # Use the versioned prompt if available, otherwise fall back to the default
        if self.prompt_module and hasattr(
            self.prompt_module, "get_intervention_prompt"
        ):
            try:
                # Convert enum keys to string keys for the theory descriptions
                theory_dict = {
                    "METACOGNITIVE": THEORY_DESCRIPTIONS[
                        InterventionType.METACOGNITIVE
                    ],
                    "CONCEPTUAL": THEORY_DESCRIPTIONS[InterventionType.CONCEPTUAL],
                    "APPLICATION": THEORY_DESCRIPTIONS[InterventionType.APPLICATION],
                    "VOCABULARY": THEORY_DESCRIPTIONS[InterventionType.VOCABULARY],
                }

                return self.prompt_module.get_intervention_prompt(
                    story_id=story_id,
                    story_title=story_data["title"],
                    story_text=story_data["story_text"],
                    skills_info=skills_info,
                    theory_descriptions=theory_dict,
                )
            except Exception as e:
                logger.error(f"Error calling get_intervention_prompt: {e}")
                # Fall through to default prompt

        # Default prompt if module not loaded or error occurred
        prompt = f"""
        You are an education expert analyzing a story to identify the best pedagogical intervention points.
        
        Story ID: {story_id}
        Title: {story_data["title"]}
        
        Story Text:
        {story_data["story_text"]}
        
        Top Skills for this story (in order of relevance):
        """

        for i, skill in enumerate(skills_info, 1):
            prompt += f"\n{i}. {skill['skill_id']} (score: {skill['child_score']:.3f})\n   {skill['explanation']}"

        prompt += f"""
        
        Based on educational theory, there are four types of pedagogical interventions that can enhance student learning and comprehension:

        1. METACOGNITIVE - Interventions that help students reflect on and monitor their own thinking and learning processes:
        {THEORY_DESCRIPTIONS[InterventionType.METACOGNITIVE]}
        

        2. CONCEPTUAL - Interventions that develop deeper understanding of core ideas and relationships:
        {THEORY_DESCRIPTIONS[InterventionType.CONCEPTUAL]}
        

        3. APPLICATION - Interventions that help students apply knowledge to real-world contexts:
        {THEORY_DESCRIPTIONS[InterventionType.APPLICATION]}
        

        4. VOCABULARY - Interventions that build word knowledge and language comprehension:
        {THEORY_DESCRIPTIONS[InterventionType.VOCABULARY]}
    

        Your task is to analyze the story and:

        1. Break down the story into individual sentences
        
        2. For each of the four intervention types above:
        - Identify the most impactful sentence where that type of intervention would be most effective
        - Note the 0-based index of the chosen sentence
        - Design a specific intervention (question, prompt, or activity) that:
            * Aligns with that intervention type's pedagogical goals by using the theory descriptions above
            * Directly relates to the story content and identified top skills
            * Can be naturally delivered by an AI tutor during reading
        - Explain why this sentence and intervention were chosen, referencing:
            * Why this moment is pedagogically valuable
            * How the intervention supports learning goals
        
        A good intervention point should:
        - Be clearly relevant to the pedagogical intervention type
        - Have appropriate complexity and length
        - Contain information worth emphasizing
        - Provide opportunity for student engagement
        
        If you cannot find a suitable intervention point for a particular type, you may omit it.
        """

        return prompt

    def _execute_llm_intervention_request(
        self, prompt: str, story_text: str
    ) -> list[dict[str, Any]]:
        """
        Execute the LLM request to identify intervention points.

        Args:
            prompt: Formatted prompt for the LLM
            story_text: The full story text for validation

        Returns:
            List of intervention points
        """
        try:
            response: InterventionPointSet = self.client.chat.completions.create(
                model=self.model_name,
                response_model=InterventionPointSet,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an educational content analysis assistant that identifies pedagogical intervention points.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            for point in response.intervention_points:
                point.validate_sentence_in_text(story_text)

            logger.info(
                f"Identified {len(response.intervention_points)} intervention points for story {story_text[:20]}..."
            )
            return response.intervention_points

        except Exception as e:
            logger.error(f"Error identifying intervention points with LLM: {e}")
            return []

    def identify_intervention_points(self, *, story_id: str) -> list[dict[str, Any]]:
        """
        Identify potential intervention points for a story based on its top skills.

        This method uses the LLM-based approach to identify intervention points.

        Args:
            story_id: ID of the story to analyze

        Returns:
            List of dictionaries containing intervention points
        """
        return self.identify_intervention_points_with_llm(story_id=story_id)

    def process_all_stories(
        self, *, limit: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Process all stories and identify intervention points.

        Args:
            limit: Optional limit on the number of stories to process

        Returns:
            Dictionary mapping story IDs to lists of intervention points
        """
        if self.results_df is None or self.stories_df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        story_ids: list[str] = list(self.top_skills.keys())

        if limit:
            story_ids = story_ids[:limit]

        for story_id in story_ids:
            logger.info(f"Processing story {story_id}")
            try:
                intervention_points: list[dict[str, Any]] = (
                    self.identify_intervention_points(story_id=story_id)
                )
                if intervention_points:
                    result[story_id] = intervention_points
            except Exception as e:
                logger.error(f"Error processing story {story_id}: {e}")

        return result

    def identify_intervention_points_for_skill(
        self, story_id: str, skill_id: str
    ) -> list:
        """
        Identify intervention points for a specific skill in a story.

        Args:
            story_id: ID of the story
            skill_id: ID of the skill

        Returns:
            List of intervention points
        """
        logger.info(
            f"Identifying intervention points for story {story_id}, skill {skill_id}"
        )

        # Get story data
        if self.stories_df is None:
            logger.error("Stories data not loaded. Call load_data() first.")
            return []

        story_data = self.stories_df[self.stories_df["story_id"] == story_id]
        if story_data.empty:
            logger.warning(f"Story {story_id} not found in stories data")
            return []

        story_text = story_data.iloc[0]["story_text"]
        story_title = story_data.iloc[0]["title"]

        # Get skill data
        if self.results_df is None:
            logger.error("Skill results data not loaded. Call load_data() first.")
            return []

        skill_data = self.results_df[
            (self.results_df["story_id"] == story_id)
            & (self.results_df["skill_id"] == skill_id)
        ]
        if skill_data.empty:
            logger.warning(f"Skill {skill_id} not found for story {story_id}")
            return []

        # Building skill info for the prompt
        skill_info = [
            {
                "skill_id": skill_id,
                "child_score": skill_data.iloc[0]["child_score"],
                "explanation": skill_data.iloc[0].get("child_explanation", ""),
            }
        ]

        # Build the prompt for intervention points
        prompt = self._build_intervention_prompt(
            story_id=story_id,
            story_data={"title": story_title, "story_text": story_text},
            skills_info=skill_info,
        )

        # Execute the request
        try:
            interventions = self._execute_llm_intervention_request(prompt, story_text)
            return interventions
        except Exception as e:
            logger.error(
                f"Error executing LLM request for story {story_id}, skill {skill_id}: {e}"
            )
            return []

    def generate_intervention_points(self, story_id: str) -> list:
        """
        Generate intervention points for a specific story.

        Args:
            story_id: ID of the story to generate intervention points for

        Returns:
            List of intervention points
        """
        story_data = self.stories_df[self.stories_df["story_id"] == story_id].iloc[0]
        story_title = story_data["title"]

        # Get skills for the story
        story_skills = self.results_df[self.results_df["story_id"] == story_id]

        if story_skills.empty:
            logger.warning(f"No skills found for story {story_id}")
            return []

        # Sort by child_score and take only the top 3 skills
        top_skills = story_skills.sort_values(by="child_score", ascending=False).head(3)
        logger.info(
            f"Processing top 3 skills for story {story_id} out of {len(story_skills)} total matches"
        )

        all_points = []

        for _, skill_row in top_skills.iterrows():
            skill_id = skill_row["skill_id"]
            try:
                logger.info(
                    f"Processing skill {skill_id} with score {skill_row['child_score']:.3f}"
                )
                points = self.identify_intervention_points_for_skill(story_id, skill_id)

                for point in points:
                    # Safely access attributes with getattr + default value
                    all_points.append(
                        {
                            "story_id": story_id,
                            "story_title": story_title,
                            "sentence": getattr(point, "sentence", ""),
                            "position": getattr(point, "position", 0),
                            "skill_id": skill_id,
                            "intervention_type": getattr(
                                point, "intervention_type", "CONCEPTUAL"
                            ),
                            "intervention": getattr(point, "intervention", ""),
                            "explanation": getattr(point, "explanation", ""),
                            "score": getattr(point, "score", 0.5),
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Error generating intervention points for story {story_id}, skill {skill_id}: {e}"
                )

        return all_points

    def generate_report(
        self,
        output_path: str = f"{PATH_TO_PROCESSED_DATA}/intervention_points.csv",
        limit: Optional[int] = None,
        print_examples: bool = False,
    ) -> pd.DataFrame:
        """
        Generate a report of intervention points and save it to a CSV file.

        Args:
            output_path: Path to save the CSV report
            limit: Optional limit on the number of stories to process
            print_examples: Whether to print example interventions to console

        Returns:
            DataFrame with the intervention points
        """
        self.load_data()

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        story_ids = self.results_df["story_id"].unique()

        if limit:
            story_ids = story_ids[:limit]

        logger.info(f"Processing {len(story_ids)} stories")

        all_intervention_points = []

        for story_id in tqdm(story_ids, desc="Generating intervention points"):
            try:
                intervention_points = self.generate_intervention_points(story_id)
                all_intervention_points.extend(intervention_points)
                time.sleep(0.5)  # Avoid rate limiting
            except Exception as e:
                logger.error(f"Error processing story {story_id}: {e}")

        if all_intervention_points:
            df = pd.DataFrame(all_intervention_points)

            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} intervention points to {output_path}")

            if print_examples and not df.empty:
                print(
                    f"Found {len(df)} intervention points across {df['story_id'].nunique()} stories."
                )

                for int_type in [
                    "METACOGNITIVE",
                    "CONCEPTUAL",
                    "APPLICATION",
                    "VOCABULARY",
                ]:
                    type_df = df[df["intervention_type"] == int_type]
                    if not type_df.empty:
                        top_example = type_df.iloc[0]
                        print(f"\nExample {int_type} intervention:")
                        print(f"Story: {top_example['story_title']}")
                        print(f"Skill: {top_example['skill_id']}")
                        print(f'Sentence: "{top_example["sentence"]}"')
                        print(f'Intervention: "{top_example["intervention"]}"')
                        print(f"Explanation: {top_example['explanation']}")
                        print(f"Score: {top_example['score']:.3f}")

            return df
        else:
            logger.warning("No intervention points were generated")
            return pd.DataFrame()


def run_intervention_analysis(
    limit=None,
    print_examples=False,
    output_path=None,
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    prompt_version="v1",
):
    """
    Run the intervention point analysis process with configurable parameters.

    Args:
        limit (int, optional): Maximum number of stories to process. Default is None (all).
        print_examples (bool, optional): Whether to print example interventions. Default is False.
        output_path (str, optional): Path to save results CSV. Default is None (uses default path).
        model_name (str, optional): Name of the model to use. Default is "gpt-4".
        temperature (float, optional): Temperature for model generation. Default is 0.2.
        prompt_version (str, optional): Version of prompts to use. Default is "v1".

    Returns:
        pd.DataFrame: DataFrame with intervention points.
    """
    try:
        if not output_path:
            processed_data_path = get_processed_data_path(prompt_version)
            os.makedirs(processed_data_path, exist_ok=True)
            output_path = f"{processed_data_path}/intervention_points.csv"
        else:
            output_path = output_path

        generator = InterventionPoint(
            model_name=model_name,
            temperature=temperature,
            prompt_version=prompt_version,
        )

        # Load data
        generator.load_data()

        # Generate the report with the intervention points
        report_df = generator.generate_report(
            output_path=output_path, limit=limit, print_examples=print_examples
        )

        return report_df
    except Exception as e:
        logger.error(f"Error running intervention analysis: {e}")
        return None


# Main execution block remains the same, but add prompt_version parameter to the argument parser

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate pedagogical intervention points"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of stories to process"
    )
    parser.add_argument(
        "--examples", action="store_true", help="Print example interventions"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results CSV"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4", help="Model to use for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for generation"
    )
    parser.add_argument(
        "--prompt-version", type=str, default="v1", help="Version of prompts to use"
    )

    args = parser.parse_args()

    run_intervention_analysis(
        limit=args.limit,
        print_examples=args.examples,
        output_path=args.output,
        model_name=args.model,
        temperature=args.temperature,
        prompt_version=args.prompt_version,
    )
