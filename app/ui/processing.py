"""
Processing UI components for the Streamlit application.

This module provides UI rendering functions for the story processing page.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import PyPDF2
from typing import Any
import numpy as np
import asyncio
import os
import tempfile

from src.skill_matcher import run_skill_matcher
from src.intervention_point import run_intervention_analysis
from app.config import PATH_TO_PROCESSED_DATA, PATH_TO_PROMPTS, get_processed_data_path


def render_process_stories() -> None:
    """
    Renders the process stories page.

    This function has UI side effects.
    """
    st.title("🔄 Process Stories")
    st.markdown("""
    This page allows you to process stories to generate skill scores and intervention points.
    """)

    # Get available prompt versions by checking directories
    prompt_versions = [d.name for d in Path(PATH_TO_PROMPTS).iterdir() if d.is_dir()]
    if not prompt_versions:
        prompt_versions = ["v1"]  # Default if no directories found

    # Add a sidebar selector for prompt version
    with st.sidebar:
        st.subheader("A/B Testing Options")
        selected_prompt_version = st.selectbox(
            "Prompt Version to Use:",
            options=prompt_versions,
            index=0,
            help="Select which version of prompts to use for processing",
        )
        st.info(f"Using prompt version: {selected_prompt_version}")

        # Show where data will be stored
        processed_data_path = get_processed_data_path(selected_prompt_version)
        st.caption(f"Data will be stored in: {processed_data_path}")

    tabs = st.tabs(
        [
            "📤 Upload New Stories",
            "🔍 Assign Skills to Stories",
            "💡 Identify Intervention Points",
            "📊 Skill Analytics",
        ]
    )

    with tabs[0]:
        st.header("📤 Upload New Stories")

        with st.expander("Upload CSV", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload a CSV file with stories",
                type=["csv"],
                key="upload_csv_uploader",
            )
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                if st.button("Process Uploaded Stories", key="process_csv_button"):
                    st.info("Processing uploaded stories...")
            else:
                st.button(
                    "Process Uploaded Stories",
                    key="process_csv_disabled_button",
                    disabled=True,
                )

        with st.expander("Generate Story with LLM", expanded=False):
            st.markdown("### Generate a New Story Using AI")
            st.markdown(
                "Use AI to generate a new story based on your selected parameters"
            )

            # Story length slider
            story_length = st.slider(
                "Approximate Story Length (words)",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                help="Choose the approximate length of the generated story in words",
            )

            # Get available skills for selection
            available_skills = pd.read_csv("data/source_data/skills.csv")

            # Skill selection
            selected_skills = st.multiselect(
                "Skills to Include in Story",
                options=available_skills["skill_id"].tolist(),
                format_func=lambda x: f"{x} - {available_skills[available_skills['skill_id'] == x]['description'].iloc[0][:50]}...",
                help="Select skills that should be incorporated into the generated story",
            )

            # Story topic input (optional)
            story_topic = st.text_input(
                "Optional Story Topic/Theme",
                value="",
                help="Provide a topic or theme for the story (optional)",
            )

            # Generate button
            if st.button("Generate Story", key="generate_story_button"):
                st.warning(
                    "This feature is experimental and not yet working. Check back soon!"
                )

                # Replace the nested expander with regular markdown
                st.markdown("---")
                st.markdown("### Story Generation Details")
                st.markdown("When implemented, this feature will:")
                st.markdown("- Generate stories using LLM based on selected skills")
                st.markdown("- Allow customization of story length and topics")
                st.markdown("- Automatically save stories to your database")
                st.markdown("- Pre-process stories for skill matching")

        with st.expander("Create from PDF", expanded=False):
            pdf_dir = Path("data/pdfs")
            if not pdf_dir.exists():
                pdf_dir.mkdir(parents=True, exist_ok=True)
                st.warning(
                    "PDF directory created. Please add PDF files to data/pdfs/ folder."
                )
                pdf_files = []
            else:
                pdf_files = list(pdf_dir.glob("*.pdf"))

            if pdf_files:
                pdf_options = {}
                for pdf_file in pdf_files:
                    try:
                        with open(pdf_file, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            num_pages = len(reader.pages)
                            pdf_options[pdf_file] = (
                                f"{pdf_file.name} ({num_pages} pages)"
                            )
                    except Exception:
                        pdf_options[pdf_file] = f"{pdf_file.name}"

                selected_pdf = st.selectbox(
                    "Select PDF file",
                    options=list(pdf_options.keys()),
                    format_func=lambda x: pdf_options[x],
                    key="pdf_select",
                )

                st.markdown("#### PDF Processing Parameters")
                col1, col2 = st.columns(2)

                with col1:
                    max_chunks = st.number_input(
                        "Maximum chunks to process",
                        min_value=1,
                        max_value=10,
                        value=2,
                        help="Higher values process more of the PDF but take longer",
                        key="pdf_max_chunks",
                    )

                    random_sampling = st.checkbox(
                        "Use random sampling",
                        value=True,
                        help="If checked, chunks will be randomly sampled from the document",
                        key="pdf_random_sampling",
                    )

                with col2:
                    max_text_snippets = st.number_input(
                        "Maximum text snippets per chunk",
                        min_value=1,
                        max_value=10,
                        value=1,
                        help="Higher values extract more content but may include less relevant text",
                        key="pdf_max_snippets",
                    )

                available_skills = pd.read_csv("data/source_data/skills.csv")
                st.markdown("#### Select Skills (Optional)")
                st.markdown("If none selected, all skills will be considered")
                selected_skills = st.multiselect(
                    "Skills to match",
                    options=available_skills["skill_id"].tolist(),
                    format_func=lambda x: f"{x} - {available_skills[available_skills['skill_id'] == x]['description'].iloc[0][:50]}...",
                    key="pdf_selected_skills",
                )

                # Use processed data path based on selected prompt version
                processed_data_dir = get_processed_data_path(selected_prompt_version)
                os.makedirs(f"{processed_data_dir}/processed_pdfs", exist_ok=True)

                output_path = st.text_input(
                    "Output file name",
                    value=f"{processed_data_dir}/processed_pdfs/{selected_pdf.stem}.csv",
                    help="Name of the output file to save results",
                    key="pdf_output_path",
                )

                if st.button("Process PDF", key="process_pdf_button"):
                    st.info("Processing PDF...")

            st.markdown("---")
            st.markdown("### Previously Processed PDFs")

            # List processed PDFs from the selected version directory
            processed_data_dir = get_processed_data_path(selected_prompt_version)
            processed_dir = Path(f"{processed_data_dir}/processed_pdfs")
            os.makedirs(processed_dir, exist_ok=True)

            pdf_results = list(processed_dir.glob("*.csv"))

            if pdf_results:
                selected_result = st.selectbox(
                    "Select a processed PDF result to view",
                    options=pdf_results,
                    format_func=lambda x: x.stem,
                    key="pdf_result_select",
                )

                if st.button("Load Selected Result", key="load_pdf_result_button"):
                    try:
                        result_df = pd.read_csv(selected_result)
                        st.dataframe(result_df, key="pdf_result_dataframe")

                        st.markdown("#### Result Statistics")
                        st.markdown(
                            f"- Total text snippets: {result_df['text'].nunique()}"
                        )
                        st.markdown(f"- Total skill matches: {len(result_df)}")
                        st.markdown(
                            f"- Unique skills matched: {result_df['skill_id'].nunique()}"
                        )

                        st.markdown("#### Skill Distribution")
                        skill_counts = (
                            result_df["skill_id"].value_counts().reset_index()
                        )
                        skill_counts.columns = ["Skill ID", "Count"]
                        st.bar_chart(
                            skill_counts.set_index("Skill ID"),
                            key="pdf_skill_dist_chart",
                        )
                    except Exception as e:
                        st.error(f"Error loading result: {str(e)}")
            else:
                st.info("No processed PDF results found.")

    with tabs[1]:
        st.header("🔍 Assign Skills to Stories")
        st.markdown("""
        Configure parameters to match stories with appropriate skills.
        """)
        skill_threshold: float = st.slider(
            "Skill Match Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="skills_threshold_slider",
        )
        max_skills: int = st.number_input(
            "Maximum Skills per Story",
            min_value=1,
            max_value=10,
            value=3,
            key="skills_max_skills_input",
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_size: int = st.slider(
                "Batch Size",
                min_value=1,
                max_value=4,
                value=1,
                help="Number of stories to process in parallel",
                key="skills_batch_size_slider",
            )
            num_batches: int = st.slider(
                "Number of Batches",
                min_value=1,
                max_value=5,
                value=1,
                help="Number of batches to process",
                key="skills_num_batches_slider",
            )

        with col2:
            limit: int = st.number_input(
                "Number of stories to process (leave at 0 for all)",
                min_value=0,
                value=0,
                help="Set to 0 to process all stories",
                key="skills_limit_input",
            )
            sleep_time: int = st.slider(
                "Sleep Time Between Batches (seconds)",
                min_value=1,
                max_value=30,
                value=5,
                key="skills_sleep_time_slider",
            )

        use_embeddings: bool = st.checkbox(
            "Use Embeddings",
            value=True,
            help="Use embeddings for faster, more efficient skill matching",
            key="skills_use_embeddings_checkbox",
        )

        # Use processed data path based on selected prompt version
        processed_data_dir = get_processed_data_path(selected_prompt_version)
        os.makedirs(processed_data_dir, exist_ok=True)

        output_path: str = st.text_input(
            "Output Path",
            value=f"{processed_data_dir}/skill_match_results.csv",
            help="Path to save the skill matching results",
            key="skills_output_path_input",
        )

        if st.button("Start Skill Matching", key="skills_start_button"):
            with st.spinner("Running skill matching..."):
                st.info(
                    f"Starting skill matching with prompt version: {selected_prompt_version}, batch size: {batch_size}, batches: {num_batches}"
                )

                # Convert limit from 0 to None if needed
                actual_limit = None if limit == 0 else limit

                # Run the skill matcher asynchronously
                try:
                    results_df = asyncio.run(
                        run_skill_matcher(
                            limit=actual_limit,
                            batch_size=batch_size,
                            num_batches=num_batches,
                            sleep_time=sleep_time,
                            output_path=output_path,
                            use_embeddings=use_embeddings,
                            # Use None for the embedding path to use the default
                            model_name="gpt-3.5-turbo",
                            prompt_version=selected_prompt_version,
                        )
                    )

                    if results_df is not None and not results_df.empty:
                        st.success(
                            f"✅ Skill matching completed successfully. {len(results_df)} matches found."
                        )
                        st.dataframe(results_df, key="skills_results_dataframe")
                    else:
                        st.warning("⚠️ No matches found or an error occurred.")
                except Exception as e:
                    st.error(f"Error processing skill matches: {str(e)}")

    with tabs[2]:
        st.header("💡 Identify Intervention Points")
        st.markdown("""
        Generate pedagogical intervention points for stories based on matched skills.
        """)

        col1, col2 = st.columns(2)

        with col1:
            limit_interventions: int = st.number_input(
                "Number of stories to process (leave at 0 for all)",
                min_value=0,
                value=0,
                help="Set to 0 to process all stories",
                key="intervention_limit_input",
            )

            model_name: str = st.selectbox(
                "LLM Model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"],
                index=0,
                help="LLM model to use for generating intervention points",
                key="intervention_model_select",
            )

        with col2:
            temperature: float = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Higher values make output more random",
                key="intervention_temperature_slider",
            )

            print_examples: bool = st.checkbox(
                "Print Example Interventions",
                value=False,
                help="Print example interventions to console",
                key="intervention_examples_checkbox",
            )

        # Use processed data path based on selected prompt version
        processed_data_dir = get_processed_data_path(selected_prompt_version)
        os.makedirs(processed_data_dir, exist_ok=True)

        intervention_output_path: str = st.text_input(
            "Output Path",
            value=f"{processed_data_dir}/intervention_points.csv",
            help="Path to save the intervention points",
            key="intervention_output_path_input",
        )

        if st.button("Generate Intervention Points", key="intervention_start_button"):
            with st.spinner("Generating intervention points..."):
                st.info(
                    f"Starting intervention point generation with prompt version: {selected_prompt_version}, model: {model_name}"
                )

                # Convert limit from 0 to None if needed
                actual_limit = None if limit_interventions == 0 else limit_interventions

                try:
                    intervention_df = run_intervention_analysis(
                        limit=actual_limit,
                        print_examples=print_examples,
                        output_path=intervention_output_path,
                        model_name=model_name,
                        temperature=temperature,
                        prompt_version=selected_prompt_version,
                    )

                    if intervention_df is not None and not intervention_df.empty:
                        st.success(
                            f"✅ Intervention point generation completed successfully. {len(intervention_df)} points identified."
                        )
                        st.dataframe(
                            intervention_df, key="intervention_results_dataframe"
                        )
                    else:
                        st.warning(
                            "⚠️ No intervention points found or an error occurred."
                        )
                except Exception as e:
                    st.error(f"Error generating intervention points: {str(e)}")

    with tabs[3]:
        st.header("📊 Skill Analytics")
        st.markdown("""
        Analyze skill matching results and compare prompt versions.
        """)

        # Create tabs for different analyses
        analysis_tabs = st.tabs(
            ["Version Comparison", "Skill Distribution", "Story Coverage"]
        )

        with analysis_tabs[0]:
            st.subheader("Prompt Version Comparison")

            # Select versions to compare
            versions_to_compare = st.multiselect(
                "Select prompt versions to compare",
                options=prompt_versions,
                default=[prompt_versions[0]] if prompt_versions else [],
                key="compare_versions_select",
            )

            if st.button("Compare Versions", key="compare_versions_button"):
                if len(versions_to_compare) < 2:
                    st.warning("Please select at least two versions to compare")
                else:
                    # Load data for each version
                    version_data = {}
                    for version in versions_to_compare:
                        try:
                            version_path = get_processed_data_path(version)
                            skill_match_path = f"{version_path}/skill_match_results.csv"

                            if os.path.exists(skill_match_path):
                                df = pd.read_csv(skill_match_path)
                                version_data[version] = df
                                st.info(
                                    f"Loaded {len(df)} matches for version {version}"
                                )
                            else:
                                st.warning(f"No data found for version {version}")
                        except Exception as e:
                            st.error(
                                f"Error loading data for version {version}: {str(e)}"
                            )

                    if len(version_data) >= 2:
                        # Compare average scores
                        scores_df = pd.DataFrame(
                            {
                                version: data["child_score"].mean()
                                for version, data in version_data.items()
                            },
                            index=["Average Score"],
                        ).T

                        st.subheader("Average Skill Match Scores")
                        st.bar_chart(scores_df)

                        # Count of matches
                        matches_df = pd.DataFrame(
                            {
                                version: len(data)
                                for version, data in version_data.items()
                            },
                            index=["Total Matches"],
                        ).T

                        st.subheader("Number of Matches")
                        st.bar_chart(matches_df)

                        # Compare distribution of scores
                        st.subheader("Score Distribution")

                        # Create histogram data
                        hist_data = [
                            data["child_score"] for data in version_data.values()
                        ]

                        # Plot histogram using streamlit
                        st.caption("Distribution of Child Scores by Version")
                        hist_df = pd.DataFrame(
                            {
                                version: np.histogram(
                                    data["child_score"], bins=10, range=(0, 1)
                                )[0]
                                for version, data in version_data.items()
                            }
                        )
                        hist_df.index = [
                            f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)
                        ]
                        st.bar_chart(hist_df)

        with analysis_tabs[1]:
            st.subheader("Skill Distribution Analysis")

            # Select a version to analyze
            version_to_analyze = st.selectbox(
                "Select prompt version to analyze",
                options=prompt_versions,
                index=0,
                key="analyze_version_select",
            )

            if st.button("Analyze Skills", key="analyze_skills_button"):
                try:
                    version_path = get_processed_data_path(version_to_analyze)
                    skill_match_path = f"{version_path}/skill_match_results.csv"

                    if os.path.exists(skill_match_path):
                        df = pd.read_csv(skill_match_path)

                        # Count skills
                        skill_counts = df["skill_id"].value_counts().reset_index()
                        skill_counts.columns = ["Skill ID", "Count"]

                        st.subheader(f"Top Skills in Version {version_to_analyze}")
                        st.dataframe(skill_counts.head(20))

                        # Plot skill distribution
                        st.subheader("Skill Distribution")
                        st.bar_chart(skill_counts.set_index("Skill ID").head(10))

                        # Domain analysis
                        if "skill_id" in df.columns:
                            df["domain"] = df["skill_id"].apply(
                                lambda x: x.split(".")[0]
                                if isinstance(x, str) and "." in x
                                else "unknown"
                            )

                            domain_counts = df["domain"].value_counts().reset_index()
                            domain_counts.columns = ["Domain", "Count"]

                            st.subheader("Domain Distribution")
                            st.dataframe(domain_counts)
                            st.bar_chart(domain_counts.set_index("Domain"))
                    else:
                        st.warning(f"No data found for version {version_to_analyze}")
                except Exception as e:
                    st.error(f"Error analyzing data: {str(e)}")

        with analysis_tabs[2]:
            st.subheader("Story Coverage Analysis")

            # Select a version to analyze
            version_to_analyze = st.selectbox(
                "Select prompt version",
                options=prompt_versions,
                index=0,
                key="story_version_select",
            )

            if st.button("Analyze Story Coverage", key="analyze_stories_button"):
                try:
                    version_path = get_processed_data_path(version_to_analyze)
                    skill_match_path = f"{version_path}/skill_match_results.csv"
                    intervention_path = f"{version_path}/intervention_points.csv"

                    # Load skill match data
                    if os.path.exists(skill_match_path):
                        skill_df = pd.read_csv(skill_match_path)

                        # Count stories
                        story_counts = skill_df["story_id"].value_counts().reset_index()
                        story_counts.columns = ["Story ID", "Skill Matches"]

                        # Load intervention data if available
                        if os.path.exists(intervention_path):
                            intervention_df = pd.read_csv(intervention_path)

                            # Count interventions per story
                            intervention_counts = (
                                intervention_df["story_id"].value_counts().reset_index()
                            )
                            intervention_counts.columns = ["Story ID", "Interventions"]

                            # Merge the counts
                            merged_counts = pd.merge(
                                story_counts,
                                intervention_counts,
                                on="Story ID",
                                how="outer",
                            ).fillna(0)

                            st.subheader("Story Coverage Analysis")
                            st.dataframe(merged_counts)

                            # Plot as a grouped bar chart
                            st.subheader("Skill Matches vs Interventions by Story")
                            st.bar_chart(merged_counts.set_index("Story ID"))
                        else:
                            st.subheader("Story Coverage Analysis (Skill Matches Only)")
                            st.dataframe(story_counts)
                            st.bar_chart(story_counts.set_index("Story ID"))
                    else:
                        st.warning(
                            f"No skill match data found for version {version_to_analyze}"
                        )
                except Exception as e:
                    st.error(f"Error analyzing story coverage: {str(e)}")
