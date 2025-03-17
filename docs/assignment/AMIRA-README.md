# Background Knowledge Skill Tagging Challenge

In this challenge, we invite you to develop a solution that strengthens student learning by connecting educational content with relevant background knowledge skills. This is an open-ended problem where we're interested in seeing your unique approach and creative thinking.

Effective learning from text depends heavily on a student's existing knowledge base. As students engage with stories, certain passages naturally present ideal opportunities to assess or reinforce specific knowledge domains. When students encounter key concepts during reading, having targeted support at crucial moments can significantly improve both their understanding of the text and retention of new information.

For an effective educational platform, we need an intelligent system that can identify both which texts best align with particular skills and precisely where in these texts meaningful learning opportunities exist. Your solution will identify the most valuable places in texts where strategic interventions could reinforce specific educational skills.

The scale of educational content makes this particularly challenging. Human experts could manually identify these connections, but their review capacity is inherently limited and cannot keep pace with the volume of potential content. An effective solution must therefore not only identify the right connections, but also be designed to make optimal use of limited human review resources through smart prioritization and efficient feedback mechanisms.

## Available Inputs

* A set of background knowledge skills (e.g., earth_systems.erosion_weathering)
* A collection of stories (e.g., "Helen realized that everything had a name and...")

These are provided in the `data` folder:
- `data/skills.csv`: Background knowledge skills with IDs and descriptions
- `data/stories.csv`: Stories with IDs, text content, and titles

## Your Task

Given a collection of stories and a set of background knowledge skills, your challenge is to build a system that identifies stories that effectively illustrate specific topics and pinpoints the precise places in these texts where it would be meaningful to either measure or exercise these background knowledge skills.

We encourage you to leverage LLM techniques creatively to develop an effective solution. Consider what makes a specific moment in text pedagogically valuable – it might be a key term that represents a core concept, a sentence that explains a process, or a paragraph that provides a meaningful example. These intervention points can range from individual words to phrases or short passages, depending on where the most meaningful learning opportunity exists. Your solution should identify these rich educational moments where targeted support would have the greatest impact on student understanding.

There is no single "correct" approach to this challenge. We're interested in a diverse range of solutions and perspectives. Feel free to interpret aspects of the problem in ways that highlight your strengths and interests – whether that means focusing on sophisticated NLP techniques, educational theory applications, innovative evaluation methods, or other areas you find compelling.

## Deliverables

While we've outlined some suggested deliverables below, we welcome creative interpretations of these requirements. In the format of a GitHub repository, please provide:

1. **System Design and Methodology**: A brief explanation of:
   - Your overall approach and key design choices
   - Which techniques you employed and why
   - How you determined what makes a moment in text suitable for a learning intervention
   - How your system integrates with human review processes, including how you evaluated effectiveness, strategies for incorporating feedback with limited review capacity, and approaches to quality control that reduce human burden
     *(Note: Implementation of human review integration is not required. A conceptual description of your approach is sufficient for this aspect.)*

2. **Sample Output**: Results for at least 3-5 skills, demonstrating:
   - Which stories were matched to each skill and why
   - The specific locations in those stories where educational interventions would be valuable
   - Justification for why these particular moments are pedagogically meaningful

3. **Pipeline Code**: Implementation of your solution that we can run to replicate your results. Your code should:
   - Process the provided dataset of skills and stories
   - Identify relevant stories for a given skill
   - Pinpoint specific intervention points within those stories
   - Include clear documentation explaining how to run and use your system

4. **Additional Artifacts**: Any visualizations, diagrams, or supplementary materials that help explain your approach or findings. We encourage you to include any additional elements that showcase your unique perspective on this challenge.

This challenge is intentionally open-ended to let you showcase both your technical skills and your understanding of educational contexts. Don't hesitate to extend or modify aspects of the challenge if you see interesting directions to explore. We're interested in seeing how you can help create more targeted, meaningful learning moments for students through intelligent content analysis, and we value novel approaches that might lead to unexpected insights.

Good luck!