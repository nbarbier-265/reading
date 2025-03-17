#!/usr/bin/env python3
"""
Script to match stories to skills using an LLM with structured outputs.
Computes a weighted score for each story-skill pair.
"""

import os
import asyncio
from typing import Any
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import importlib.util
import sys

import instructor
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI, AsyncOpenAI

from src.models import (
    ChildSkill,
    SkillScores,
    SingleSkillScore,
    ReadingLevel,
    SkillMatchResult,
    StoryTask,
    BatchCount,
    EmbeddingResult,
)
from config import (
    PATH_TO_PROCESSED_DATA,
    PATH_TO_SOURCE_DATA,
    get_prompt_version_path,
    get_processed_data_path,
)


class SkillMatcher:
    """Match stories to skills using an LLM.

    Attributes:
        client: OpenAI client
        async_client: AsyncOpenAI client
        instructor_client: instructor client
        async_instructor_client: async instructor client
        skills_df: skills DataFrame
        stories_df: stories DataFrame
        parent_skills: parent skills dictionary
        child_skills: child skills dictionary
        skill_embeddings: skill embeddings dictionary
        model: OpenAI model
        temperature: Temperature setting for OpenAI model
        prompt_version: Version of prompts to use

    Methods:
        # Data loading methods
        load_data: Load skills and stories data from CSV files
        # Skill matching methods
        get_skill_score_async: Get score for a single skill for a given story
        get_all_skill_scores_async: Get scores for all skills for a given story
        match_story_to_skills_async: Match a story to skills with weighted scores
        process_all_stories_async: Process all stories and match them to skills
        process_all_stories_with_embeddings: Process all stories and match them to skills using embeddings
        generate_skill_embeddings: Generate skill embeddings
        match_story_to_skills_with_embeddings: Match a story to skills with weighted scores using embeddings
        calculate_batch_count: Calculate the number of batches to process
        get_batch_tasks: Get the tasks for a batch
        process_batch: Process a batch
        log_batch_error: Log an error for a batch
        # Embedding methods
        _format_embedding_results: Format embedding results
        _create_embedding_dataframe: Create an embedding dataframe
        _process_story_batches_with_embeddings: Process story batches with embeddings
        _extract_story_results: Extract story results from a list of dictionaries
        _create_results_dataframe: Create a results dataframe
        # Validation methods
        _validate_data_loaded: Validate that the data has been loaded


    Side effects:
        - Loads the skills and stories data into the class attributes
        - Creates a parent skills dictionary
        - Creates a child skills dictionary
        - Creates instructor clients
        - Creates OpenAI clients
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = os.environ.get("OPENAI_MODEL_NAME"),
        temperature: float = 0.2,
        prompt_version: str = "v1",
    ) -> None:
        """Initialize the skill matcher.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model_name: Name of the OpenAI model to use (defaults to OPENAI_MODEL_NAME environment variable)
            temperature: Temperature setting for OpenAI model (defaults to 0.2)
            prompt_version: Version of prompts to use (defaults to v1)

        Returns:
            None
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client: OpenAI = OpenAI(api_key=api_key)
        self.async_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)
        self.instructor_client: instructor.Client = instructor.patch(self.client)
        self.async_instructor_client: instructor.AsyncClient = instructor.patch(
            self.async_client
        )
        self.model = model_name
        self.temperature = temperature
        self.skills_df: pd.DataFrame | None = None
        self.stories_df: pd.DataFrame | None = None
        self.parent_skills: dict[str, str] = {}
        self.child_skills: dict[str, list[ChildSkill]] = {}
        self.skill_embeddings: dict[str, np.ndarray] = {}
        self.prompt_version = prompt_version

        # Load the appropriate prompt module
        self.prompt_module = self._load_prompt_module()

    def _load_prompt_module(self):
        """
        Load the appropriate prompt module based on the prompt version.
        """
        prompt_path = get_prompt_version_path(self.prompt_version)
        module_path = os.path.join(prompt_path, "skill_matcher.py")

        if not os.path.exists(module_path):
            logger.warning(
                f"Prompt module not found at {module_path}, using default prompts"
            )
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                "skill_matcher_prompts", module_path
            )
            if spec is None:
                logger.error(f"Could not create spec for module at {module_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            if module is None:
                logger.error(f"Could not create module from spec for {module_path}")
                return None

            sys.modules["skill_matcher_prompts"] = module
            spec.loader.exec_module(module)

            # Verify the module has the required function
            if not hasattr(module, "get_skill_score_prompt"):
                logger.error(
                    f"Module at {module_path} does not contain get_skill_score_prompt function"
                )
                return None

            logger.info(f"Successfully loaded prompt module from {module_path}")
            return module
        except Exception as e:
            logger.error(f"Error loading prompt module: {e}")
            return None

    def load_data(
        self,
        skills_path: str = f"{PATH_TO_SOURCE_DATA}/skills.csv",
        stories_path: str = f"{PATH_TO_SOURCE_DATA}/stories.csv",
    ) -> None:
        """Load skills and stories data from CSV files.

        Args:
            skills_path: Path to the skills CSV file
            stories_path: Path to the stories CSV file

        Returns:
            None

        Side effects:
            - Loads the skills and stories data into the class attributes
        """
        logger.info(f"Loading skills from {skills_path}")
        self.skills_df = pd.read_csv(skills_path)

        logger.info(f"Loading stories from {stories_path}")
        self.stories_df = pd.read_csv(stories_path)

        self._extract_skills()

    def _extract_skills(self) -> None:
        """Extract parent and child skills from the skills DataFrame."""
        for _, row in self.skills_df.iterrows():
            skill_id: str = row["skill_id"]
            description: str = row["description"]

            if "." in skill_id:
                parent_id, _ = skill_id.split(".", 1)
                if parent_id not in self.parent_skills:
                    self.parent_skills[parent_id] = parent_id

                if parent_id not in self.child_skills:
                    self.child_skills[parent_id] = []
                self.child_skills[parent_id].append(
                    ChildSkill(skill_id=skill_id, description=description)
                )
            else:
                self.parent_skills[skill_id] = skill_id

        logger.info(
            f"Extracted {len(self.parent_skills)} parent skills and child skills for {len(self.child_skills)} parent categories"
        )

    async def get_skill_score_async(
        self,
        story_id: str,
        story_text: str,
        title: str,
        skill_id: str,
        description: str = "",
    ) -> SingleSkillScore | None:
        """Get score for a single skill for a given story asynchronously.

        Args:
            story_id: ID of the story
            story_text: Text content of the story
            title: Title of the story
            skill_id: ID of the skill to score
            description: Optional description of the skill

        Returns:
            SingleSkillScore object or None if an error occurred
        """
        logger.info(f"Getting score for skill {skill_id} for story {story_id}")

        skill_type = "parent category" if "." not in skill_id else "specific skill"

        # Use the versioned prompt if available, otherwise fall back to the default
        prompt = None
        if self.prompt_module and hasattr(self.prompt_module, "get_skill_score_prompt"):
            try:
                prompt_function = getattr(self.prompt_module, "get_skill_score_prompt")
                if callable(prompt_function):
                    prompt = prompt_function(
                        story_id=story_id,
                        title=title,
                        story_text=story_text,
                        skill_id=skill_id,
                        description=description,
                        skill_type=skill_type,
                    )
                else:
                    logger.error(f"get_skill_score_prompt is not callable in module")
            except Exception as e:
                logger.error(f"Error calling get_skill_score_prompt: {e}")
                prompt = None

        # If prompt is None or not a string, use the default
        if prompt is None or not isinstance(prompt, str):
            # Default prompt if module not loaded or error occurred
            desc_text = f": {description}" if description else ""
            prompt = f"""
            You are an education expert analyzing a story to identify its relevance to a specific skill area.
            
            Story ID: {story_id}
            Title: {title}
            Story Text:
            {story_text}
            
            Evaluate how relevant this story is to the following {skill_type}:
            Skill ID: {skill_id}

            {desc_text}
            
            Assign a score between 0 and 1 indicating relevance, where:
            - 0 means not relevant at all
            - 1 means extremely relevant
            
            This should be a continuous number between 0 and 1, not just 0 or 1.
            
            Provide a brief explanation for your score.
            """

        try:
            logger.debug(
                f"Sending prompt to OpenAI API for skill {skill_id} and story {story_id}"
            )

            response: SingleSkillScore = await self.async_instructor_client.chat.completions.create(
                model=self.model,
                response_model=SingleSkillScore,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an education content analysis assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            logger.debug(
                f"Received score for skill {skill_id} and story {story_id}: {response.score}"
            )
            return response

        except Exception as e:
            logger.error(
                f"Error getting score for skill {skill_id} and story {story_id}: {e}"
            )
            logger.debug(f"Exception details: {str(e)}", exc_info=True)
            return SingleSkillScore(
                skill_id=skill_id,
                score=float("nan"),
                explanation=f"Error occurred: {str(e)}",
            )

    async def get_all_skill_scores_async(
        self, story_id: str, story_text: str, title: str
    ) -> tuple[dict[str, float], dict[str, str], int]:
        """Get scores for all skills (parent and child) for a given story asynchronously.

        Args:
            story_id: ID of the story
            story_text: Text content of the story
            title: Title of the story

        Returns:
            Tuple of (dictionary mapping skill IDs to scores, dictionary mapping skill IDs to explanations, reading level)
        """
        logger.info(f"Getting all skill scores for story {story_id}")

        reading_level_prompt: str = f"""
        You are an education expert analyzing the reading level of a story.
        
        Story ID: {story_id}
        Title: {title}
        Story Text:
        {story_text}
        
        Evaluate the reading level of this text on a scale from K (Kindergarten) through 12 (12th grade).
        For Kindergarten, return 0. For all other grades, return the corresponding number (1-12).
        """

        reading_level_response = (
            await self.async_instructor_client.chat.completions.create(
                model=self.model,
                response_model=ReadingLevel,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an education content analysis assistant.",
                    },
                    {"role": "user", "content": reading_level_prompt},
                ],
            )
        )

        reading_level: int = reading_level_response.level
        logger.info(f"Determined reading level for story {story_id}: {reading_level}")

        parent_tasks: list[SingleSkillScore] = []
        for parent_id in self.parent_skills.values():
            parent_tasks.append(
                self.get_skill_score_async(story_id, story_text, title, parent_id)
            )

        child_tasks: list[SingleSkillScore] = []
        for parent_id, children in self.child_skills.items():
            for child in children:
                child_tasks.append(
                    self.get_skill_score_async(
                        story_id, story_text, title, child.skill_id, child.description
                    )
                )

        all_tasks = parent_tasks + child_tasks
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        scores: dict[str, float] = {}
        explanations: dict[str, str] = {}

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Error in skill scoring task: {result}")
                continue

            scores[result.skill_id] = result.score
            explanations[result.skill_id] = result.explanation

        logger.info(f"Retrieved scores for {len(scores)} skills for story {story_id}")
        return scores, explanations, reading_level

    async def match_story_to_skills_async(
        self, story_id: str, story_text: str, title: str
    ) -> dict[str, dict[str, float | str | int]]:
        """Match a story to skills with weighted scores asynchronously.

        Args:
            story_id: ID of the story
            story_text: Text content of the story
            title: Title of the story

        Returns:
            Dictionary mapping skill IDs to scores
        """
        logger.info(f"Matching story {story_id} to skills")

        (
            all_scores,
            all_explanations,
            reading_level,
        ) = await self.get_all_skill_scores_async(story_id, story_text, title)

        if not all_scores:
            logger.error(f"Failed to get skill scores for story {story_id}")
            return {}

        skill_matches: dict[str, dict[str, float | str | int]] = {}

        for child_id in all_scores:
            if "." in child_id:
                parent_id: str = child_id.split(".", 1)[0]
                if parent_id in all_scores:
                    parent_score: float = all_scores[parent_id]
                    child_score: float = all_scores[child_id]

                    skill_scores = SkillScores(
                        parent_score=parent_score,
                        child_score=child_score,
                        reading_level=reading_level,
                        parent_explanation=all_explanations.get(parent_id, ""),
                        child_explanation=all_explanations.get(child_id, ""),
                    )

                    try:
                        result = SkillMatchResult(
                            story_id=story_id,
                            story_title=title,
                            skill_id=child_id,
                            scores=skill_scores,
                        )

                        skill_matches[child_id] = result.__dict__
                    except ValueError as e:
                        logger.error(
                            f"Error creating SkillMatchResult for skill {child_id} and story {story_id}: {e}"
                        )
                        continue

        logger.info(
            f"Matched story {title} with ID {story_id} to {len(skill_matches)} skills"
        )
        return skill_matches

    async def process_all_stories_async(
        self,
        limit: int | None = None,
        batch_size: int = 5,
        num_batches: int | None = None,
        sleep_time: int = 5,
    ) -> pd.DataFrame | None:
        """Process all stories and match them to skills asynchronously.

        Args:
            limit: Optional limit on the number of stories to process
            batch_size: Number of stories to process in each batch
            num_batches: Optional limit on number of batches to process
            sleep_time: Time to sleep between batches in seconds

        Returns:
            DataFrame with story-skill matches and scores
        """
        if not self._validate_data_loaded():
            return None

        tasks = self._create_story_tasks(limit)
        all_results = await self._process_story_batches(
            tasks, batch_size, num_batches, sleep_time
        )
        story_results = self._extract_story_results(tasks, all_results)

        return self._create_results_dataframe(story_results)

    def _validate_data_loaded(self) -> bool:
        """Check if required data is loaded."""
        if self.stories_df is None or self.skills_df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        return True

    def _create_story_tasks(self, limit: int | None) -> list[StoryTask]:
        """Create list of story processing tasks."""
        stories_to_process = self.stories_df.iloc[:limit] if limit else self.stories_df

        tasks: list[StoryTask] = []
        for _, story in stories_to_process.iterrows():
            story_id = story["story_id"]
            story_text = story["story_text"]
            title = story["title"]

            logger.info(f"Creating task for story {story_id}: {title}")
            tasks.append(
                StoryTask(story_id=story_id, title=title, story_text=story_text)
            )

        return tasks

    async def _process_story_batches(
        self,
        tasks: list[StoryTask],
        batch_size: int,
        num_batches: int | None,
        sleep_time: int,
    ) -> list[dict[str, dict[str, float | str | int]]]:
        """Process stories in batches."""
        if not tasks:
            logger.warning("No tasks to process")
            return []

        batch_count: BatchCount = self.calculate_batch_count(
            len(tasks), batch_size, num_batches
        )

        logger.info(
            f"Processing {batch_count.to_process} batches out of {batch_count.total} total batches"
        )

        all_results: list[dict[str, dict[str, float | str | int]]] = []

        for idx in range(batch_count.to_process):
            batch_tasks = self.get_batch_tasks(tasks, idx, batch_size)
            logger.info(
                f"Processing batch {idx + 1}/{batch_count.to_process} with {len(batch_tasks)} stories"
            )

            async_tasks = [
                self.match_story_to_skills_async(
                    task.story_id, task.story_text, task.title
                )
                for task in batch_tasks
            ]
            batch_results = await asyncio.gather(*async_tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.log_batch_error(batch_tasks[i], result)
                else:
                    all_results.append(result)

            logger.info(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)

        logger.info(f"Processed {len(all_results)} stories")
        return all_results

    def _extract_story_results(
        self,
        tasks: list[StoryTask],
        all_results: list[dict[str, dict[str, float | str | int]]],
    ) -> list[dict[str, Any]]:
        """Extract results for each story-skill pair."""
        story_results = []

        for idx, skill_scores in enumerate(all_results):
            if isinstance(skill_scores, Exception):
                logger.error(
                    f"Error processing story {tasks[idx].title} (ID: {tasks[idx].story_id}): {skill_scores}"
                )
                continue

            for skill_id, scores_dict in skill_scores.items():
                result_row = {
                    "story_id": tasks[idx].story_id,
                    "story_title": tasks[idx].title,
                    "skill_id": skill_id,
                    "parent_score": scores_dict["parent_score"],
                    "child_score": scores_dict["child_score"],
                    "reading_level": scores_dict["reading_level"],
                    "parent_explanation": scores_dict["parent_explanation"],
                    "child_explanation": scores_dict["child_explanation"],
                }
                story_results.append(result_row)

        return story_results

    def _create_results_dataframe(
        self, story_results: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Create DataFrame from story results."""
        if story_results:
            return pd.DataFrame(story_results)

        logger.warning("No results generated")
        return pd.DataFrame()

    async def generate_skill_embeddings(self) -> None:
        """Generate embeddings for all skills."""
        cache_file = f"{PATH_TO_PROCESSED_DATA}/skill_embeddings_cache.npy"

        if os.path.exists(cache_file):
            try:
                logger.info("Loading skill embeddings from cache")
                cached_data = np.load(cache_file, allow_pickle=True).item()
                self.skill_embeddings = cached_data
                logger.info(
                    f"Loaded {len(self.skill_embeddings)} skill embeddings from cache"
                )
                return
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}")

        logger.info("Generating embeddings for all skills")

        if not self._validate_data_loaded():
            return

        self.skill_embeddings = {}  # Initialize the dictionary if it doesn't exist

        for _, row in self.skills_df.iterrows():
            skill_id: str = row["skill_id"]
            description: str = row["description"]

            try:
                response = await self.async_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=f"Skill ID: {skill_id}. Description: {description}",
                )
                self.skill_embeddings[skill_id] = np.array(response.data[0].embedding)
                logger.debug(f"Generated embedding for skill {skill_id}")
            except Exception as e:
                logger.error(f"Error generating embedding for skill {skill_id}: {e}")

        parent_ids_to_generate = set()
        for skill_id in self.skill_embeddings.keys():
            if "." in skill_id:
                parent_id = skill_id.split(".", 1)[0]
                if parent_id not in self.skill_embeddings:
                    parent_ids_to_generate.add(parent_id)

        for parent_id in parent_ids_to_generate:
            parent_description = ""
            parent_row = self.skills_df[self.skills_df["skill_id"] == parent_id]
            if not parent_row.empty:
                parent_description = parent_row.iloc[0]["description"]

            try:
                logger.info(f"Generating embedding for parent skill {parent_id}")
                response = await self.async_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=f"Skill ID: {parent_id}. Description: {parent_description}",
                )
                self.skill_embeddings[parent_id] = np.array(response.data[0].embedding)
                logger.debug(f"Generated embedding for parent skill {parent_id}")
            except Exception as e:
                logger.error(
                    f"Error generating embedding for parent skill {parent_id}: {e}"
                )

        logger.info(f"Generated embeddings for {len(self.skill_embeddings)} skills")

        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, self.skill_embeddings)
            logger.info(f"Saved skill embeddings to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")

    async def match_story_to_skills_with_embeddings(
        self, story_id: str, story_text: str, title: str
    ) -> dict[str, dict[str, float]]:
        """Match a story to skills using embeddings and cosine similarity.

        Args:
            story_id: ID of the story
            story_text: Text content of the story
            title: Title of the story

        Returns:
            Dictionary mapping skill IDs to similarity scores
        """
        logger.info(f"Matching story {story_id} to skills using embeddings")

        if not hasattr(self, "skill_embeddings") or not self.skill_embeddings:
            logger.info("No skill embeddings found, generating them now")
            await self.generate_skill_embeddings()

        try:
            story_embedding_response = await self.async_client.embeddings.create(
                model="text-embedding-ada-002",
                input=f"Title: {title}. Story: {story_text}",
            )
            story_embedding = np.array(story_embedding_response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for story {story_id}: {e}")
            return {}

        similarity_scores: dict[str, float] = {}
        for skill_id, skill_embedding in self.skill_embeddings.items():
            similarity = cosine_similarity(
                story_embedding.reshape(1, -1), skill_embedding.reshape(1, -1)
            )[0][0]
            similarity_scores[skill_id] = float(similarity)
            logger.info(f"Similarity score for skill {skill_id}: {similarity}")

        skill_matches: dict[str, dict[str, float]] = {}
        for child_id, child_score in similarity_scores.items():
            logger.info(f"Child ID: {child_id}, Child Score: {child_score}")
            if "." in child_id:
                parent_id = child_id.split(".", 1)[0]
                if parent_id in similarity_scores:
                    parent_score = similarity_scores[parent_id]
                    logger.info(f"Parent ID: {parent_id}, Parent Score: {parent_score}")
                    skill_matches[child_id] = {
                        "story_id": story_id,
                        "story_title": title,
                        "skill_id": child_id,
                        "parent_score": parent_score,
                        "child_score": child_score,
                        "method": "embedding",
                    }

        logger.info(
            f"Matched story {story_id} to {len(skill_matches)} skills using embeddings"
        )
        return skill_matches

    async def process_all_stories_with_embeddings(
        self,
        limit: int | None = None,
        batch_size: int = 5,
        num_batches: int | None = None,
        sleep_time: int = 5,
    ) -> pd.DataFrame | None:
        """Process all stories and match them to skills using embeddings.

        Args:
            limit: Optional limit on the number of stories to process
            batch_size: Number of stories to process in each batch
            num_batches: Optional limit on number of batches to process
            sleep_time: Time to sleep between batches in seconds

        Returns:
            DataFrame with story-skill matches and similarity scores
        """
        if not self._validate_data_loaded():
            return None

        if not hasattr(self, "skill_embeddings") or not self.skill_embeddings:
            await self.generate_skill_embeddings()

        tasks: list[StoryTask] = self._create_story_tasks(limit)

        if not tasks:
            logger.warning(
                "No story tasks created. Check if stories data is loaded correctly."
            )
            return pd.DataFrame()

        all_results = await self._process_story_batches_with_embeddings(
            tasks, batch_size, num_batches, sleep_time
        )

        if not all_results:
            logger.warning("No results returned from processing story batches")
            return pd.DataFrame()

        embedding_results = self._format_embedding_results(all_results)

        return self._create_embedding_dataframe(embedding_results)

    async def _process_story_batches_with_embeddings(
        self,
        tasks: list[StoryTask],
        batch_size: int,
        num_batches: int | None,
        sleep_time: int,
    ) -> list[dict[str, dict[str, float]]]:
        """Process stories in batches using embeddings.

        Args:
            tasks: List of StoryTask objects
            batch_size: Number of stories to process in each batch
            num_batches: Optional limit on number of batches to process
            sleep_time: Time to sleep between batches in seconds

        Returns:
            List of results from embedding matching
        """
        if not tasks:
            logger.warning("No tasks to process")
            return []

        batch_count = self.calculate_batch_count(len(tasks), batch_size, num_batches)

        logger.info(
            f"Processing {batch_count.to_process} batches out of {batch_count.total} total batches"
        )

        all_results: list[dict[str, dict[str, float]]] = []

        for idx in range(batch_count.to_process):
            batch_tasks = self.get_batch_tasks(tasks, idx, batch_size)

            logger.info(
                f"Processing batch {idx + 1}/{batch_count.to_process} with {len(batch_tasks)} stories using embeddings"
            )

            batch_results = await self.process_batch(batch_tasks)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.log_batch_error(batch_tasks[i], result)
                else:
                    all_results.append(result)

            logger.info(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)

        logger.info(f"Processed {len(all_results)} story results")
        return all_results

    def calculate_batch_count(
        self, tasks_length: int, batch_size: int, num_batches: int | None
    ) -> BatchCount:
        """Calculate the total number of batches and how many to process."""
        total_batches = tasks_length // batch_size + (
            1 if tasks_length % batch_size > 0 else 0
        )
        to_process = (
            min(total_batches, num_batches)
            if num_batches is not None
            else total_batches
        )
        return BatchCount(total=total_batches, to_process=to_process)

    def get_batch_tasks(
        self, tasks: list[StoryTask], batch_idx: int, batch_size: int
    ) -> list[StoryTask]:
        """Get a slice of tasks for the current batch."""
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tasks))
        return tasks[start_idx:end_idx]

    async def process_batch(
        self, batch_tasks: list[StoryTask]
    ) -> list[dict[str, dict[str, float]]]:
        """Process a batch of stories asynchronously.

        Args:
            batch_tasks: List of StoryTask objects

        Returns:
            List of results from embedding matching
        """
        async_tasks = [
            self.match_story_to_skills_with_embeddings(
                task.story_id, task.story_text, task.title
            )
            for task in batch_tasks
        ]
        return await asyncio.gather(*async_tasks, return_exceptions=True)

    def log_batch_error(self, task: StoryTask, error: Exception) -> None:
        """Log an error that occurred during batch processing.

        Args:
            task: StoryTask object
            error: Exception that occurred

        Side effects:
            Logs an error to the logger
        """
        logger.error(
            f"Error processing story {task.title} (ID: {task.story_id}): {error}"
        )

    def _format_embedding_results(
        self, results: list[dict[str, dict[str, float]]]
    ) -> list[EmbeddingResult]:
        """Format embedding results into a standardized list of EmbeddingResult objects.

        Args:
            results: List of embedding matching results

        Returns:
            List of standardized EmbeddingResult objects
        """
        embedding_results: list[EmbeddingResult] = []

        if not results:
            logger.warning("No results to format")
            return embedding_results

        for result_dict in results:
            if not result_dict:
                continue

            for skill_id, data in result_dict.items():
                embedding_results.append(
                    EmbeddingResult(
                        story_id=data["story_id"],
                        story_title=data["story_title"],
                        skill_id=skill_id,
                        parent_score=data["parent_score"],
                        child_score=data["child_score"],
                    )
                )

        logger.info(f"Formatted {len(embedding_results)} embedding results")
        return embedding_results

    def _create_embedding_dataframe(
        self, embedding_results: list[EmbeddingResult]
    ) -> pd.DataFrame:
        """Create DataFrame from embedding results.

        Args:
            embedding_results: List of standardized EmbeddingResult objects

        Returns:
            DataFrame with embedding results
        """
        if embedding_results:
            logger.info(f"Creating DataFrame with {len(embedding_results)} rows")
            return pd.DataFrame([result.__dict__ for result in embedding_results])

        logger.warning("No embedding results generated")
        return pd.DataFrame()


async def run_skill_matcher(
    *,
    limit: int | None = None,
    batch_size: int = 1,
    num_batches: int = 1,
    sleep_time: int = 5,
    output_path: str = None,
    use_embeddings: bool = False,
    embeddings_output_path: str = None,
    model_name: str = os.environ.get("OPENAI_MODEL_NAME"),
    temperature: float = 0.2,
    prompt_version: str = "v1",
) -> pd.DataFrame | None:
    """
    Run the skill matcher with configurable parameters.

    Args:
        limit: Maximum number of stories to process
        batch_size: Number of stories to process in parallel
        num_batches: Number of batches to process
        sleep_time: Sleep time between batches
        output_path: Path to save the results CSV
        use_embeddings: Whether to use embeddings
        embeddings_output_path: Path to save embedding results CSV
        model_name: Name of the OpenAI model to use
        temperature: Temperature setting for OpenAI model
        prompt_version: Version of prompts to use

    Returns:
        DataFrame with skill match results or None if an error occurred
    """
    try:
        if not output_path:
            # Use versioned path if not specified
            processed_data_path = get_processed_data_path(prompt_version)
            # Make sure directory exists
            os.makedirs(processed_data_path, exist_ok=True)
            output_path = f"{processed_data_path}/skill_match_results.csv"

        if not embeddings_output_path and use_embeddings:
            # Use versioned path if not specified
            processed_data_path = get_processed_data_path(prompt_version)
            # Make sure directory exists
            os.makedirs(processed_data_path, exist_ok=True)
            embeddings_output_path = (
                f"{processed_data_path}/skill_match_embedding_results.csv"
            )

        matcher = SkillMatcher(
            model_name=model_name,
            temperature=temperature,
            prompt_version=prompt_version,
        )
        matcher.load_data()

        if use_embeddings:
            print(f"Using embeddings to match stories to skills")
            await matcher.generate_skill_embeddings()
            results_df = await matcher.process_all_stories_with_embeddings(
                limit=limit,
                batch_size=batch_size,
                num_batches=num_batches,
                sleep_time=sleep_time,
            )

            if results_df is not None:
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")

                # Format results for embedding
                embedding_results = matcher._format_embedding_results(
                    matcher._extract_story_results(
                        matcher._create_story_tasks(limit), results_df
                    )
                )
                embedding_df = matcher._create_embedding_dataframe(embedding_results)
                embedding_df.to_csv(embeddings_output_path, index=False)
                print(f"Embedding results saved to {embeddings_output_path}")

            return results_df
        else:
            print(f"Processing all stories to match skills")
            results_df = await matcher.process_all_stories_async(
                limit=limit,
                batch_size=batch_size,
                num_batches=num_batches,
                sleep_time=sleep_time,
            )

            if results_df is not None:
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")

            return results_df
    except Exception as e:
        logger.error(f"Error running skill matcher: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match stories to skills using an LLM with structured outputs."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of stories to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of stories to process in parallel",
    )
    parser.add_argument(
        "--num-batches", type=int, default=1, help="Number of batches to process"
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=10,
        help="Sleep time between batches",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{PATH_TO_PROCESSED_DATA}/skill_match_results.csv",
        help="Path to save the results CSV",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use embeddings for skill matching",
    )
    parser.add_argument(
        "--embeddings-output",
        type=str,
        default=f"{PATH_TO_PROCESSED_DATA}/skill_match_embedding_results.csv",
        help="Path to save embedding results CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL_NAME"),
        help="Name of the OpenAI model to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature setting for OpenAI model",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1",
        help="Version of prompts to use",
    )

    args = parser.parse_args()

    asyncio.run(
        run_skill_matcher(
            limit=args.limit,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            sleep_time=args.sleep_time,
            output_path=args.output,
            use_embeddings=args.use_embeddings,
            embeddings_output_path=args.embeddings_output,
            model_name=args.model,
            temperature=args.temperature,
            prompt_version=args.prompt_version,
        )
    )
