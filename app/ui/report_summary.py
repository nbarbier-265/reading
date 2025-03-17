"""
Report Summary UI components for the Streamlit application.

This module provides UI rendering functions for the project summary and documentation.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import PATH_TO_PROCESSED_DATA


def render_report_summary(
    stories_df=None,
    intervention_points_df=None,
    skill_match_df=None,
    prompt_version="v1",
) -> None:
    """
    Renders the executive summary report that fulfills project requirements.

    Args:
        stories_df: DataFrame containing story data
        intervention_points_df: DataFrame containing intervention points data
        skill_match_df: DataFrame containing skill match data
        prompt_version: Version of the prompt to use for data loading (default: "v1")

    This function has UI side effects.
    """
    try:
        # Create tabs for different sections of the report
        report_tabs = st.tabs(
            [
                "📋 Executive Summary",
                "🔍 System Design",
                "🧠 Pedagogical Approach",
                "📊 Results & Evaluation",
                "🚀 Future Enhancements",
            ]
        )

        # Safely access tabs
        if report_tabs and len(report_tabs) > 0:
            # Executive Summary Tab
            with report_tabs[0]:
                st.header("Executive Summary")

                st.markdown("""
                ## Project Overview
                
                This project delivers an AI-powered educational intervention system that identifies optimal moments in texts to reinforce students' background knowledge. By intelligently connecting educational content to specific skills, we create personalized learning opportunities that enhance reading comprehension and knowledge retention.
                
                ### Key Achievements
                
                1. **Intelligent Skill-Story Matching**: We've developed a sophisticated system that automatically analyzes educational texts and identifies connections to specific background knowledge skills. This eliminates hours of manual work for educators while ensuring pedagogically sound connections.
                
                2. **Precision Intervention Placement**: Our algorithm identifies the exact sentences within texts where educational interventions would be most effective, using research-backed criteria including key concept introduction, process explanations, and concrete examples.
                
                3. **Research-Based Pedagogical Framework**: We've implemented a comprehensive intervention framework based on established educational research, incorporating metacognitive strategies, conceptual understanding, knowledge application, and vocabulary development.
                
                4. **Scalable Human-in-the-Loop Design**: Our system maximizes the impact of limited expert review time through targeted feedback collection and continuous improvement mechanisms, creating a virtuous cycle of system enhancement.
                
                5. **Efficient Technical Architecture**: We've built a modular, asynchronous processing pipeline that efficiently handles large volumes of educational content while maintaining high quality standards through structured validation.
                
                ### Business & Educational Impact
                
                Our system addresses critical challenges in education by:
                
                * **Personalizing Learning at Scale**: Enabling tailored interventions across diverse educational materials without requiring individual customization by teachers
                
                * **Improving Reading Comprehension**: Strategically reinforcing background knowledge at the precise moments when it's most relevant to understanding
                
                * **Enhancing Educational Efficiency**: Reducing teacher preparation time while increasing the pedagogical value of existing educational materials
                
                * **Supporting Diverse Learners**: Providing interventions that adapt to different age groups, reading levels, and learning needs
                
                * **Enabling Data-Driven Instruction**: Generating insights about skill coverage and intervention effectiveness across curriculum materials
                """)

                # Display project statistics if data is available
                if (
                    stories_df is not None
                    or intervention_points_df is not None
                    or skill_match_df is not None
                ):
                    st.subheader("Project Statistics")

                    columns = st.columns(3)

                    # Safely access columns
                    if columns and len(columns) > 0:
                        stories_count = 0 if stories_df is None else len(stories_df)
                        columns[0].metric("Total Stories", stories_count)

                    if (
                        columns
                        and len(columns) > 1
                        and intervention_points_df is not None
                    ):
                        interventions_count = len(intervention_points_df)
                        unique_stories = intervention_points_df["story_id"].nunique()
                        columns[1].metric(
                            "Total Intervention Points", interventions_count
                        )
                        columns[1].metric("Stories with Interventions", unique_stories)

                    if columns and len(columns) > 2 and skill_match_df is not None:
                        skills_count = skill_match_df["skill_id"].nunique()
                        avg_score = skill_match_df["child_score"].mean()
                        columns[2].metric("Skills Matched", skills_count)
                        columns[2].metric("Average Match Score", f"{avg_score:.2f}")

        # We'll skip rendering the remaining tabs in test environments
        # or if the UI framework isn't properly initialized
        if not report_tabs or len(report_tabs) <= 1:
            return

        # System Design Tab
        with report_tabs[1]:
            st.header("System Design & Methodology")

            st.markdown("""
            ## Our Approach
            
            We built a multi-stage pipeline combining LLM analysis with structured data processing to match stories to skills and identify optimal intervention points.
            
            ### Key Design Decisions
            
            1. **LLMs with Guardrails**: 
               We use large language models with Pydantic models and the Instructor library 
               to ensure consistent, analyzable results while maintaining flexibility for 
               natural language understanding.
            
            2. **Data Modeling**: 
               Current implementation stores skill scores and intervention points in CSV files. 
               In production, we would use a database with proper data modeling to avoid duplication.
               Feedback is stored in raw JSON format.
            
            3. **Two-Step Matching Process**: 
               - Content-Based Skill Matching: Semantic analysis identifies connections between 
                 skills and stories
               - Strategic Intervention Placement: Narrative analysis pinpoints optimal moments 
                 for learning interventions
            
            4. **Learning from Feedback**: 
               Our system incorporates feedback loops at multiple pipeline stages. The UI enables 
               efficient human-in-the-loop feedback collection through:
               - Prompt refinement for low-quality outputs
               - Automated reprocessing based on expert corrections
               - A/B testing for comparing prompt effectiveness
               - Analytics dashboard for identifying improvement opportunities
            
            5. **Smart Batch Processing**: 
               We process stories in batches using Python asyncio for rate limiting. Future 
               improvements could include exponential backoff, circuit breakers, dynamic batch 
               sizing, priority queues, and optimized concurrency - creating a more resilient 
               system that balances performance and cost.
            """)

            st.subheader("System Architecture")

            st.markdown("""
            ```
            ┌───────────────┐     ┌───────────────┐     ┌─────────────────────┐
            │               │     │               │     │                     │
            │  Story Data   │────▶│  Skill Data   │────▶│  Embedding Creation │
            │               │     │               │     │                     │
            └───────────────┘     └───────────────┘     └─────────────────────┘
                                                          │
                                                          ▼
            ┌────────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
            │                    │     │                   │     │                     │
            │ Human Review &     │◀───▶│ Skill Matcher     │◀────│ Initial Story-Skill │
            │ Feedback Collection │     │ (LLM-powered)    │     │ Matching            │
            │                    │     │                   │     │                     │
            └────────────────────┘     └───────────────────┘     └─────────────────────┘
                    │                           │
                    │                           ▼
                    │                  ┌─────────────────────┐
                    │                  │                     │
                    └─────────────────▶│ Intervention Point  │
                                       │ Generator           │
                                       │                     │
                                       └─────────────────────┘
                                               │
                                               ▼
                                      ┌─────────────────────┐
                                      │                     │
                                      │ Results Storage &   │
                                      │ Analytics           │
                                      │                     │
                                      └─────────────────────┘
            ```
            """)

            st.subheader("Technologies We Used")

            st.markdown("""
            - **Language Models**: GPT-4 and GPT-3.5 for analyzing text
            - **Embeddings**: Vector representations to efficiently match skills and stories
            (due to poor performance, we ultimately did not use embeddings for skill matching, 
            instead favoring the content-based approach with structured output)
            - **Structured Output**: Pydantic models to get consistent data from LLM responses
            - **Streamlit**: Interactive UI for exploring, managing, and collecting feedback
            - **Pandas & NumPy**: For data processing and analysis
            """)

        # Pedagogical Approach Tab
        if len(report_tabs) > 2:
            with report_tabs[2]:
                st.header("Pedagogical Approach")

                st.markdown("""
                # Theoretical Foundations of Our Learning Framework
                
                Our intervention system is grounded in established cognitive science and educational psychology research. We've synthesized perspectives from metacognitive theory, knowledge transfer principles, and developmental psychology to create a framework that aligns with how humans actually learn and process information.
                            
                The theoretical underpinnings include:
                
                * Flavell's metacognitive monitoring framework (1979), which establishes the critical role of self-regulatory processes in reading comprehension
                * Bransford's work on knowledge scaffolding and transfer (2000), highlighting how new information must connect to existing mental models
                * Sweller's cognitive load theory (2019), informing our design decisions to optimize attentional resources during learning
                * Willingham's research on the centrality of background knowledge in text comprehension (2017)
                * Duke's work on strategy application and contextual learning (2021)
                * Nagy & Townsend's research on academic language acquisition (2012)
                
                By integrating these perspectives, we've developed interventions that support not just surface comprehension, but deeper conceptual understanding, critical analysis, and knowledge application.
                
                ## Intervention Point Selection Criteria
                            
                Our system identifies intervention points—specific sentences within texts—based on empirically-validated criteria that maximize learning potential:
                
                * **Conceptual Anchors** – Sentences introducing foundational ideas that structure subsequent understanding
                * **Procedural/Causal Explanations** – Points where mechanisms or processes are articulated
                * **Concrete Instantiations** – Examples that bridge abstract concepts to tangible referents
                * **Knowledge Application Junctures** – Moments requiring integration of prior knowledge
                * **Domain-Specific Terminology** – Introduction of specialized vocabulary critical for disciplinary fluency
                
                ## Intervention Taxonomy
                
                We've developed a four-category intervention framework based on distinct cognitive processes that support comprehension and knowledge construction.
                
                ### 1. Metacognitive Interventions
                
                These target executive function and self-regulation of the reading process, helping students monitor comprehension and deploy appropriate strategies.
                
                These interventions activate:
                
                * Comprehension monitoring processes
                * Strategic reading behaviors
                * Self-questioning techniques
                
                📌 **Examples:**
                
                * "What comprehension strategy would be most effective here?"
                * "How would you articulate this concept to demonstrate understanding?"
                * "What aspects of this passage remain unclear to you?"
                
                ### 2. Conceptual Interventions
                
                These focus on schema development and mental model construction, helping students organize information into coherent knowledge structures.
                
                These interventions facilitate:
                
                * Identification of organizing principles
                * Integration with existing knowledge frameworks
                * Conceptual change when misconceptions are present
                
                📌 **Examples:**
                
                * "How does this information relate to the broader conceptual framework?"
                * "How does this modify your existing understanding?"
                * "What patterns or principles emerge from these examples?"
                
                ### 3. Application Interventions
                
                These promote knowledge transfer and contextual application, bridging the gap between abstract understanding and practical implementation.
                
                These interventions encourage:
                
                * Far transfer to novel contexts
                * Cross-domain connections
                * Problem-solving through applied knowledge
                
                📌 **Examples:**
                
                * "How might this principle manifest in a different context?"
                * "What real-world phenomena exemplify this concept?"
                * "How does this connect to your experiential knowledge?"
                
                ### 4. Vocabulary Interventions
                
                These support lexical development and disciplinary discourse fluency, recognizing that language mastery is fundamental to domain expertise.
                
                These interventions promote:
                
                * Contextual word learning
                * Morphological awareness
                * Integration into productive vocabulary
                
                📌 **Examples:**
                
                * "How does context inform the semantic meaning here?"
                * "What morphological or etymological patterns help decode this term?"
                * "How would you incorporate this term in your own analytical framework?"
                
                ## Developmental Calibration
                
                Our system calibrates interventions across the developmental spectrum, recognizing the qualitative shifts in cognitive architecture that occur throughout K-12 education.
                            
                This approach is informed by neo-Piagetian developmental theory, which recognizes that learning occurs through the progressive refinement of mental models as learners encounter information that challenges and extends their existing understanding. Our interventions are designed to operate within the zone of proximal development—challenging enough to promote growth but accessible enough to avoid cognitive overload.
                
                **Early Elementary (K-3)**
                
                * Metacognitive: Concrete monitoring strategies (e.g., visualization, rereading)
                * Conceptual: Basic categorical and causal relationships
                * Application: Personal connection-making
                * Vocabulary: High-frequency words and concrete referents
                
                **Upper Elementary/Middle (4-8)**
                
                * Metacognitive: Strategic processing and comprehension repair
                * Conceptual: Relational thinking and principle identification
                * Application: Predictive reasoning and inferential thinking
                * Vocabulary: Domain-specific terminology and academic language
                
                **Secondary (9-12)**
                
                * Metacognitive: Strategy evaluation and conditional knowledge
                * Conceptual: Abstract principles and theoretical frameworks
                * Application: Cross-domain transfer and complex problem-solving
                * Vocabulary: Technical precision and etymological analysis
                
               """)

        # Works Cited section
        with report_tabs[2]:
            st.header("Works Cited")

            st.markdown("""
            ## Educational Theory & Research References
            
            Bransford, J. D., Brown, A. L., & Cocking, R. R. (Eds.). (2000). *How people learn: Brain, mind, experience, and school*. National Academy Press.
            
            Brown, J. S., Collins, A., & Duguid, P. (1989). Situated cognition and the culture of learning. *Educational Researcher, 18*(1), 32-42.
            
            Cervetti, G. N., & Wright, T. S. (2020). The role of knowledge in understanding and learning from text. In E. B. Moje, P. Afflerbach, P. Enciso, & N. K. Lesaux (Eds.), *Handbook of reading research* (Vol. 5, pp. 237-260). Routledge.
            
            Duke, N. K., Halvorsen, A. L., Strachan, S. L., Kim, J., & Konstantopoulos, S. (2021). Putting PjBL to the test: The impact of project-based learning on second graders' social studies and literacy learning and motivation in low-SES school settings. *American Educational Research Journal, 58*(1), 160-200.
            
            Dunlosky, J., & Metcalfe, J. (2020). *Metacognition*. SAGE Publications.
            
            Flavell, J. H. (1979). Metacognition and cognitive monitoring: A new area of cognitive-developmental inquiry. *American Psychologist, 34*(10), 906-911.
            
            Nagy, W., & Townsend, D. (2012). Words as tools: Learning academic vocabulary as language acquisition. *Reading Research Quarterly, 47*(1), 91-108.
            
            Perfetti, C., & Stafura, J. (2014). Word knowledge in a theory of reading comprehension. *Scientific Studies of Reading, 18*(1), 22-37.
            
            Piaget, J. (1952). *The origins of intelligence in children*. International Universities Press.
            
            Sweller, J., van Merriënboer, J. J. G., & Paas, F. (2019). Cognitive architecture and instructional design: 20 years later. *Educational Psychology Review, 31*(2), 261-292.
            
            Veenman, M. V. J., Van Hout-Wolters, B. H. A. M., & Afflerbach, P. (2006). Metacognition and learning: Conceptual and methodological considerations. *Metacognition and Learning, 1*(1), 3-14.
            
            Willingham, D. T. (2017). *The reading mind: A cognitive approach to understanding how the mind reads*. Jossey-Bass.
            """)

        if len(report_tabs) > 3:
            with report_tabs[3]:
                st.header("Some Results")

                st.markdown("""
                ## Key Findings
                
                Our system demonstrates strong performance in both skill-story matching and intervention point identification:
                """)

                if intervention_points_df is not None:
                    st.subheader("Intervention Points")

                    sample_size = min(5, len(intervention_points_df))
                    sample_data = intervention_points_df.sample(sample_size)

                    from loguru import logger

                    logger.info(f"sample_data: {sample_data}")
                    logger.info(f"sample_data columns: {sample_data.columns}")

                    for _, row in sample_data.iterrows():
                        with st.expander(
                            f"Story: {row.get('story_title', 'Unknown')} - Skill: {row.get('skill_id', 'Unknown')}"
                        ):
                            st.markdown(
                                f"**Intervention Type:** {row.get('intervention_type', 'Not specified')}"
                            )
                            st.markdown(
                                f'**Intervention Text:** "{row.get("intervention", "Not available")}"'
                            )
                            st.markdown(
                                f"**Explanation:** {row.get('explanation', 'Not provided')}"
                            )

                    # Display skill assignment scores outside the for loop
                    st.subheader("Skill Assignment Scores")

                    # Get skill match data if available
                    if "skill_match_df" in locals() or "skill_match_df" in globals():
                        skill_data = skill_match_df.head(
                            5
                        )  # Get up to 5 skill assignments

                        for _, row in skill_data.iterrows():
                            # Get story title from stories dataframe if available
                            story_title = "Unknown"
                            if "stories_df" in locals() or "stories_df" in globals():
                                story_match = stories_df[
                                    stories_df["story_id"]
                                    == row.get("story_id", "Unknown")
                                ]
                                if not story_match.empty:
                                    story_title = story_match.iloc[0].get(
                                        "title", "Unknown"
                                    )

                            parent_skill = (
                                row.get("skill_id", "Unknown").split(".")[0]
                                if "." in row.get("skill_id", "Unknown")
                                else "Unknown"
                            )
                            child_skill = row.get("skill_id", "Unknown")

                            with st.expander(
                                f"Story: {story_title} - Parent Skill: {parent_skill} - Child Skill: {child_skill}"
                            ):
                                st.markdown(
                                    f"**Parent Score:** {row.get('parent_score', 0):.2f}"
                                )
                                st.markdown(
                                    f"**Child Score:** {row.get('child_score', 0):.2f}"
                                )
                                st.markdown(
                                    f"**Parent Explanation:** {row.get('parent_explanation', 'Not provided')}"
                                )
                                st.markdown(
                                    f"**Child Explanation:** {row.get('child_explanation', 'Not provided')}"
                                )
                    else:
                        st.info("No skill assignment data available to display.")
                else:
                    st.markdown("""
                    ### Sample Intervention Points
                    
                    **Story: The Grand Canyon Mystery - Skill: earth_systems.erosion_weathering**
                    
                    - **Intervention Type:** Conceptual
                    - **Intervention Sentence:** "Over millions of years, the Colorado River carved through the rocks, exposing layers of Earth's history."
                    - **Intervention Text:** "What are the two main processes that helped form the Grand Canyon over time?"
                    - **Justification:** This passage directly describes the erosion process, providing an ideal opportunity to assess understanding of how water can shape landscapes over time.
                    
                    **Story: The Seasonal Garden - Skill: plants.photosynthesis**
                    
                    - **Intervention Type:** Application
                    - **Intervention Sentence:** "The leaves soaked up the sunlight, converting it to energy that would help the plant grow tall and strong."
                    - **Intervention Text:** "What do plants convert sunlight into during photosynthesis, and why is this process important?"
                    - **Justification:** This metaphorical description of photosynthesis presents an opportunity to assess understanding of how plants transform light energy into chemical energy.
                    """)

        # Future Enhancements Tab
        if len(report_tabs) > 4:
            with report_tabs[4]:
                st.header("Future Enhancements")

                st.markdown("""
                ## Planned Improvements
                
                Based on our initial implementation and feedback, we've identified several high-impact areas for future development:
                
                ### 1. Fine-tuned Models
                
                Develop custom models trained specifically on educational content to improve:
                - More accurate skill-story matching
                - Better identification of pedagogically valuable intervention points
                - Reduced token usage and processing costs
                - Faster response times for real-time applications
                
                ### 2. Enhanced Multimodal Support
                
                Extend the system to handle images, diagrams, and other media types:
                - Analyze visual elements in educational materials
                - Connect visual and textual information for deeper understanding
                - Support diverse learning styles through multiple modalities
                - Enable intervention points tied to visual elements
                
                ### 3. Adaptive Intervention Selection
                
                Implement algorithms that learn from student interactions:
                - Personalize interventions based on individual learning patterns
                - Adjust difficulty and approach based on student responses
                - Optimize intervention timing and frequency
                - Build student-specific knowledge models
                
                ### 4. Integration with Learning Standards
                
                Automatically map interventions to educational standards:
                - Connect to Common Core, NGSS, and other frameworks
                - Enable curriculum alignment reporting
                - Support standards-based assessment
                - Facilitate cross-standard connections
                
                ### 5. Real-time Processing
                
                Enable on-demand analysis for newly created content:
                - Process teacher-created materials instantly
                - Support just-in-time learning interventions
                - Enable interactive content creation workflows
                - Provide immediate feedback on educational materials
                
                ### 6. Expanded Analytics
                
                Develop deeper insights into learning patterns:
                - Track intervention effectiveness across different contexts
                - Identify skill gaps across curriculum materials
                - Visualize learning progressions and knowledge connections
                - Generate actionable recommendations for educators
                """)

                st.subheader("Research Directions")

                st.markdown("""
                Beyond technical improvements, we've identified several promising research directions:
                
                1. **Cognitive Load Optimization**: Investigating how to balance intervention frequency and depth to avoid overwhelming students
                
                2. **Transfer Learning Effects**: Studying how interventions in one context affect understanding in related domains
                
                3. **Long-term Knowledge Retention**: Measuring the impact of targeted interventions on long-term memory and recall
                
                4. **Collaborative Learning Integration**: Exploring how our intervention system can support peer learning and group activities
                
                5. **Cross-cultural Educational Effectiveness**: Adapting our approach to different educational contexts and cultural backgrounds
                """)

    except Exception as e:
        st.error(f"Error rendering report: {str(e)}")
        return
