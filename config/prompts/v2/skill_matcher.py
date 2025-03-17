"""
Skill Matcher Prompt Templates - Version 2

This module contains the prompts used by the skill matcher to score
stories against skills. This version includes more detailed guidance
and calibration examples for more consistent scoring.
"""


def get_skill_score_prompt(
    story_id, title, story_text, skill_id, description="", skill_type=None
):
    """
    Generate the prompt for the skill matcher.

    Args:
        story_id: ID of the story
        title: Title of the story
        story_text: Text content of the story
        skill_id: ID of the skill to score
        description: Optional description of the skill
        skill_type: The type of skill (parent or child)

    Returns:
        str: The prompt for the skill matcher
    """
    if skill_type is None:
        skill_type = "parent category" if "." not in skill_id else "specific skill"

    desc_text = f": {description}" if description else ""

    return f"""
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
    - 0.25 means slightly relevant (the skill is mentioned but not a key focus)
    - 0.5 means moderately relevant (the skill is clearly addressed but not central)
    - 0.75 means very relevant (the skill is a major theme or focus area)
    - 1 means extremely relevant (the story is primarily about this skill)
    
    You can use any value in this continuous range, not just these example points.
    
    When evaluating the story:
    1. Look for explicit mentions of concepts related to the skill
    2. Consider implicit applications of the skill in the story context
    3. Evaluate how central the skill is to understanding the story
    4. Consider the depth of coverage related to the skill
    5. Assess if the story provides teachable moments related to the skill
    
    Provide a brief explanation (3-5 sentences) for your score that includes:
    - Specific evidence from the text supporting your assessment
    - Why you believe the story is or isn't relevant to the skill
    - How the story could be used to teach this skill (if relevant)
    """
