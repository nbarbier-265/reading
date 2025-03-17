"""
Intervention Point Prompt Templates - Version 1

This module contains the prompts used by the intervention point generator
to identify pedagogical intervention points in stories.
"""


def get_intervention_prompt(
    story_id, story_title, story_text, skills_info, theory_descriptions
):
    """
    Generate the prompt for the intervention point generator.

    Args:
        story_id: ID of the story
        story_title: Title of the story
        story_text: Text content of the story
        skills_info: List of dictionaries with skill information
        theory_descriptions: Dictionary of intervention type descriptions

    Returns:
        str: The prompt for the intervention point generator
    """
    prompt = f"""
    You are an education expert analyzing a story to identify the best pedagogical intervention points.
    
    Story ID: {story_id}
    Title: {story_title}
    
    Story Text:
    {story_text}
    
    Top Skills for this story (in order of relevance):
    """

    for i, skill in enumerate(skills_info, 1):
        prompt += f"\n{i}. {skill['skill_id']} (score: {skill['child_score']:.3f})\n   {skill['explanation']}"

    prompt += f"""
    
    Based on educational theory, there are four types of pedagogical interventions that can enhance student learning and comprehension:

    1. METACOGNITIVE - Interventions that help students reflect on and monitor their own thinking and learning processes:
    {theory_descriptions["METACOGNITIVE"]}
    

    2. CONCEPTUAL - Interventions that develop deeper understanding of core ideas and relationships:
    {theory_descriptions["CONCEPTUAL"]}
    

    3. APPLICATION - Interventions that help students apply knowledge to real-world contexts:
    {theory_descriptions["APPLICATION"]}
    

    4. VOCABULARY - Interventions that build word knowledge and language comprehension:
    {theory_descriptions["VOCABULARY"]}


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
