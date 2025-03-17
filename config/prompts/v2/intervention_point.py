"""
Intervention Point Prompt Templates - Version 2

This module contains the prompts used by the intervention point generator
to identify pedagogical intervention points in stories. This version includes
more structured guidance and example interventions for better consistency.
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
    
    Example METACOGNITIVE intervention:
    - Original sentence: "Maya thought about how pollution affects marine life."
    - Intervention: "What connections can you make between what Maya is thinking about and things you've learned about ecosystems?"
    

    2. CONCEPTUAL - Interventions that develop deeper understanding of core ideas and relationships:
    {theory_descriptions["CONCEPTUAL"]}
    
    Example CONCEPTUAL intervention:
    - Original sentence: "The water cycle continues as the rain replenishes the lakes and rivers."
    - Intervention: "Can you explain the steps of the water cycle mentioned in this sentence and what would happen if one step was disrupted?"
    

    3. APPLICATION - Interventions that help students apply knowledge to real-world contexts:
    {theory_descriptions["APPLICATION"]}
    
    Example APPLICATION intervention:
    - Original sentence: "The solar panels on the school roof converted sunlight into electricity."
    - Intervention: "How might this same process of energy conversion be used in other situations? Can you think of three examples?"
    

    4. VOCABULARY - Interventions that build word knowledge and language comprehension:
    {theory_descriptions["VOCABULARY"]}
    
    Example VOCABULARY intervention:
    - Original sentence: "The scientists observed the chemical reaction, noting how the catalysts accelerated the process."
    - Intervention: "What does the word 'catalyst' mean in this context? Where else might you encounter this term?"


    Your task is to analyze the story and:

    1. Break down the story into individual sentences
    
    2. For each of the four intervention types above:
       - Identify the most impactful sentence where that type of intervention would be most effective
       - Note the 0-based index of the chosen sentence
       - Design a specific intervention (question, prompt, or activity) that:
         * Aligns with that intervention type's pedagogical goals by using the theory descriptions above
         * Directly relates to the story content and identified top skills
         * Uses appropriate academic language for the apparent reading level
         * Can be naturally delivered by an AI tutor during reading
       - Explain why this sentence and intervention were chosen, referencing:
         * Why this moment is pedagogically valuable
         * How the intervention supports learning goals
         * What specific background knowledge is being reinforced
    
    A good intervention point should:
    - Be clearly relevant to the pedagogical intervention type
    - Have appropriate complexity and length (neither too simple nor too complex)
    - Contain information worth emphasizing or exploring further
    - Provide opportunity for student engagement and deeper thinking
    - Connect directly to one of the top skills identified for this story
    
    If you cannot find a suitable intervention point for a particular type, you may omit it.
    """

    return prompt
