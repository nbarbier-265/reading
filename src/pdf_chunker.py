#!/usr/bin/env python3
"""
PDF Chunker and Skill Matcher

This module processes PDF documents by chunking them into manageable segments,
sending these chunks to the OpenAI API for analysis, and identifying text segments
that are relevant to skills defined in a skills CSV file.

The module uses asynchronous processing to efficiently handle large documents
while respecting the token window limitations of the OpenAI models.
"""

import asyncio
import os
import random
from pathlib import Path
from typing import Any, TypedDict

import instructor
import pandas as pd
import PyPDF2
import tiktoken
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from src.skill_matcher import SkillMatcher
from src.models import RelevantTextResponse, TextSnippetResult, ChunkResult

from config import PATH_TO_PROJECT

logger.add("pdf_chunker.log", rotation="10 MB")


class PDFChunker:
    """
    Processes PDF documents by chunking them into segments and analyzing them for skill relevance.

    This class handles the extraction of text from PDFs, chunking the text based on token limits,
    and coordinating the asynchronous processing of these chunks through the OpenAI API.
    """

    def __init__(
        self,
        pdf_path: str,
        skills_csv_path: str,
        model: str = os.environ.get("OPENAI_MODEL_NAME"),
        max_tokens: int = 8000,
        overlap_tokens: int = 500,
        max_chunks: int | None = None,
        random_sampling: bool = False,
        api_key: str | None = None,
        max_text_snippets: int = 10,
    ):
        """
        Initialize the PDF chunker.

        Args:
            pdf_path: Path to the PDF file to process
            skills_csv_path: Path to the CSV file containing skills definitions
            model: OpenAI model to use for analysis
            max_tokens: Maximum tokens per chunk to send to the API
            overlap_tokens: Number of tokens to overlap between chunks
            max_chunks: Maximum number of chunks to process (None for all)
            random_sampling: Whether to randomly sample chunks from the document
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            max_text_snippets: Maximum number of text snippets to extract per chunk (1-10)
        """
        self.pdf_path: Path = Path(pdf_path)
        self.skills_csv_path: Path = Path(skills_csv_path)
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.overlap_tokens: int = overlap_tokens
        self.max_chunks: int | None = max_chunks
        self.random_sampling: bool = random_sampling
        self.max_text_snippets: int = min(
            max(1, max_text_snippets), 10
        )  # Ensure between 1-10
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.skill_matcher: SkillMatcher = SkillMatcher()
        self.skill_matcher.load_data(skills_csv_path)

    def extract_text_from_pdf(self) -> str:
        """
        Extract all text content from the PDF document.

        Returns:
            The complete text content of the PDF

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PyPDF2.errors.PdfReadError: If there's an error reading the PDF
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        logger.info(f"Extracting text from {self.pdf_path}")

        text: str = ""
        try:
            with open(self.pdf_path, "rb") as file:
                reader: PyPDF2.PdfReader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"

            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Error reading PDF: {e}")
            raise

    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """
        Split the text into chunks that respect the token limit.

        Args:
            text: The complete text to chunk

        Returns:
            List of dictionaries containing text chunks and their indices
        """
        tokens: list[int] = self.tokenizer.encode(text)
        chunks: list[dict[str, Any]] = []

        start_idx: int = 0
        chunk_index: int = 0
        while start_idx < len(tokens):
            end_idx: int = min(start_idx + self.max_tokens, len(tokens))

            chunk_tokens: list[int] = tokens[start_idx:end_idx]

            chunk_text: str = self.tokenizer.decode(chunk_tokens)
            chunks.append(
                {
                    "index": chunk_index,
                    "text": chunk_text,
                    "token_start": start_idx,
                    "token_end": end_idx,
                }
            )

            start_idx = (
                end_idx - self.overlap_tokens if end_idx < len(tokens) else len(tokens)
            )
            chunk_index += 1

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def select_chunks_to_process(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Select which chunks to process based on max_chunks and random_sampling settings.

        Args:
            chunks: List of all text chunks

        Returns:
            List of chunks to process
        """
        if not self.max_chunks or self.max_chunks >= len(chunks):
            return chunks

        if self.random_sampling:
            selected_chunks: list[dict[str, Any]] = random.sample(
                chunks, self.max_chunks
            )
            logger.info(
                f"Randomly selected {len(selected_chunks)} chunks for processing"
            )
        else:
            selected_chunks: list[dict[str, Any]] = chunks[: self.max_chunks]
            logger.info(f"Selected first {len(selected_chunks)} chunks for processing")

        return selected_chunks

    async def analyze_chunk(
        self, chunk: dict[str, Any]
    ) -> ChunkResult | dict[str, Any]:
        """
        Send a chunk to the OpenAI API for analysis.

        Args:
            chunk: Dictionary containing text chunk and metadata

        Returns:
            Dictionary containing the analysis results
        """
        chunk_text: str = chunk["text"]
        chunk_index: int = chunk["index"]
        logger.info(f"Analyzing chunk {chunk_index} ({len(chunk_text)} chars)")

        try:
            prompt: str = f"""
            Analyze the following text and identify sentences or paragraphs that demonstrate 
            reading comprehension skills. Focus on segments that show:
            
            1. Understanding of main ideas and supporting details
            2. Character development and motivations
            3. Plot structure and narrative elements
            4. Thematic elements and author's purpose
            5. Vocabulary in context
            6. Inferential reasoning
            
            Return a list of the most relevant sentences or paragraphs that clearly demonstrate 
            these skills, with no additional commentary.

            TEXT EXTRACTION GUIDELINES:
            - Extract only relevant text segments that demonstrate the skills listed above
            - Do not include any analysis, commentary, or skill labels in your response
            - Each extracted text segment must be between 150-4000 characters in length
            - Keep your response concise (20-400 words per item)
            - The list should be between 1 and {self.max_text_snippets} items
            - Focus on the most illustrative examples from the text
            - Do not include very short quotes or dialogue snippets without context
            
            TEXT:
            {chunk_text}
            """

            client = instructor.patch(self.client)
            response = await client.chat.completions.create(
                model=self.model,
                response_model=list[RelevantTextResponse],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in literacy education and reading comprehension.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            text_snippets: list[TextSnippetResult] = []

            for item in response:
                snippet_text: str = item.text

                skill_matches = await self.skill_matcher.match_story_to_skills_async(
                    str(chunk_index), snippet_text, f"Chunk {chunk_index} - Snippet"
                )

                text_snippets.append(
                    {"text": snippet_text, "skill_matches": skill_matches}
                )

            return {
                "chunk_index": chunk_index,
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "text_snippets": text_snippets,
            }

        except Exception as e:
            logger.error(f"Error analyzing chunk {chunk_index}: {e}")
            return {
                "chunk_index": chunk_index,
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "error": str(e),
                "text_snippets": [],
            }

    async def process_pdf(self) -> list[ChunkResult | dict[str, Any]]:
        """
        Process the PDF file and return relevant text segments with skill matches.

        Returns:
            List of dictionaries containing relevant text segments and their skill matches
        """
        full_text: str = self.extract_text_from_pdf()

        all_chunks: list[dict[str, Any]] = self.chunk_text(full_text)

        chunks_to_process: list[dict[str, Any]] = self.select_chunks_to_process(
            all_chunks
        )

        tasks: list = [self.analyze_chunk(chunk) for chunk in chunks_to_process]
        results: list[ChunkResult | dict[str, Any]] = await asyncio.gather(*tasks)

        valid_results: list[ChunkResult | dict[str, Any]] = [
            r for r in results if "error" not in r
        ]

        logger.info(
            f"Successfully processed {len(valid_results)} chunks out of {len(chunks_to_process)}"
        )
        return valid_results

    def save_results(
        self,
        results: list[ChunkResult | dict[str, Any]],
        output_path: str = f"{PATH_TO_PROJECT}/data/processed_data/pdf_analysis_results.csv",
    ):
        """
        Save the analysis results to a CSV file.

        Args:
            results: List of analysis results
            output_path: Path to save the CSV file
        """
        flattened_results: list[dict[str, Any]] = []

        for result in results:
            chunk_index: int = result["chunk_index"]
            token_start: int = result["token_start"]
            token_end: int = result["token_end"]

            for snippet_idx, snippet in enumerate(result["text_snippets"]):
                snippet_text: str = snippet["text"]

                for skill_match in snippet["skill_matches"]:
                    flattened_results.append(
                        {
                            "chunk_index": chunk_index,
                            "snippet_index": snippet_idx,
                            "token_start": token_start,
                            "token_end": token_end,
                            "text": snippet_text,
                            "skill_id": (
                                skill_match["skill_id"]
                                if isinstance(skill_match, dict)
                                else skill_match
                            ),
                            "skill_name": (
                                skill_match["skill_name"]
                                if isinstance(skill_match, dict)
                                else ""
                            ),
                            "score": (
                                skill_match["score"]
                                if isinstance(skill_match, dict)
                                else 0.0
                            ),
                            "confidence": (
                                skill_match["confidence"]
                                if isinstance(skill_match, dict)
                                else 0.0
                            ),
                        }
                    )

        if flattened_results:
            df: pd.DataFrame = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.warning("No results to save")


async def process_pdf_file(
    pdf_path: str,
    skills_csv_path: str,
    max_chunks: int = 2,
    random_sampling: bool = True,
    max_text_snippets: int = 1,
    output_path: str = None,
) -> dict[str, Any]:
    """Process a PDF file and extract skill matches.

    Args:
        pdf_path: Path to the PDF file to process
        skills_csv_path: Path to the CSV file containing skills
        max_chunks: Maximum number of chunks to process
        random_sampling: Whether to randomly sample chunks
        max_text_snippets: Maximum number of text snippets per chunk
        output_path: Optional path to save results

    Returns:
        Dictionary containing processing statistics
    """
    load_dotenv()

    chunker: PDFChunker = PDFChunker(
        pdf_path=pdf_path,
        skills_csv_path=skills_csv_path,
        max_chunks=max_chunks,
        random_sampling=random_sampling,
        max_text_snippets=max_text_snippets,
    )

    results: list[ChunkResult | dict[str, Any]] = await chunker.process_pdf()

    if output_path:
        chunker.save_results(results, output_path)
    else:
        chunker.save_results(results)

    total_snippets: int = sum(len(r["text_snippets"]) for r in results)
    total_skill_matches: int = sum(
        sum(len(snippet["skill_matches"]) for snippet in r["text_snippets"])
        for r in results
    )

    return {
        "chunks_processed": len(results),
        "total_snippets": total_snippets,
        "total_skill_matches": total_skill_matches,
        "results": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a PDF file for skill matching"
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--skills", default="data/skills.csv", help="Path to skills CSV file"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=2, help="Maximum chunks to process"
    )
    parser.add_argument(
        "--random", action="store_true", help="Use random sampling of chunks"
    )
    parser.add_argument(
        "--snippets", type=int, default=1, help="Maximum text snippets per chunk"
    )
    parser.add_argument("--output", help="Path to save results CSV")

    args = parser.parse_args()

    result = asyncio.run(
        process_pdf_file(
            pdf_path=args.pdf,
            skills_csv_path=args.skills,
            max_chunks=args.max_chunks,
            random_sampling=args.random,
            max_text_snippets=args.snippets,
            output_path=args.output,
        )
    )

    print(f"Processed {result['chunks_processed']} chunks from the PDF")
    print(f"Extracted {result['total_snippets']} text snippets")
    print(f"Found {result['total_skill_matches']} skill matches")
