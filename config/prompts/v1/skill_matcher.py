"""
Skill Matcher Prompt Templates - Version 1

This module contains the prompts used by the skill matcher to score
stories against skills.
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
    - 1 means extremely relevant
    
    This should be a continuous number between 0 and 1, not just 0 or 1.
    
    Provide a brief explanation for your score.
    """
