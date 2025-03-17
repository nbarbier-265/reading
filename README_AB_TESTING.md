# A/B Testing Framework for Prompts

This document explains the A/B testing framework that has been added to the Amira project, allowing for experimentation with different prompt versions for skill matching and intervention point generation.

## Overview

The A/B testing framework enables you to:

1. **Create multiple prompt versions** for both skill matching and intervention point generation
2. **Run processing with different prompt versions**
3. **Save results separately** for each prompt version in dedicated directories
4. **Compare results** across different prompt versions to evaluate their effectiveness

This approach ensures that when you experiment with different prompts, you don't overwrite previous results, allowing for proper comparison and evaluation.

## Directory Structure

The prompt versions are stored in the `config/prompts` directory with the following structure:

```
config/
  └── prompts/
      ├── v1/
      │   ├── skill_matcher.py
      │   └── intervention_point.py
      └── v2/
          ├── skill_matcher.py
          └── intervention_point.py
```

- `v1/` contains the baseline prompts
- `v2/` contains alternative prompts for comparison

You can add more versions by creating new directories with the same structure.

## Processed Data Storage

Results for each prompt version are stored separately under:

```
data/
  └── processed_data/
      ├── prompt_v1/
      │   ├── skill_match_results.csv
      │   ├── intervention_points.csv
      │   ├── skill_match_feedback.json
      │   └── intervention_feedback.json
      └── prompt_v2/
          ├── skill_match_results.csv
          ├── intervention_points.csv
          ├── skill_match_feedback.json
          └── intervention_feedback.json
```

This separation ensures that results from different prompt versions don't overwrite each other.

## How to Use the A/B Testing Framework

### 1. Creating New Prompt Versions

To create a new prompt version:

1. Create a new directory under `config/prompts/` (e.g., `config/prompts/v3/`)
2. Create two files in this directory:
   - `skill_matcher.py` - Contains the prompt template for skill matching
   - `intervention_point.py` - Contains the prompt template for intervention point generation

Both files should follow the same structure as the existing versions, implementing the required functions with the same parameters.

### 2. Running Processing with Different Versions

In the application UI:

1. Navigate to the "Process Stories" tab
2. In the sidebar, select the prompt version you want to use from the dropdown
3. The application will display which version is being used and where data will be stored
4. Run the processing as usual (skill matching or intervention generation)
5. Results will be saved in the corresponding version-specific directory

### 3. Viewing Results from Different Versions

When viewing results:

1. Select the prompt version from the dropdown in the sidebar
2. The application will load data from the corresponding version-specific directory
3. You can switch between versions to compare results

### 4. Analyzing Performance

The "Skill Analytics" tab provides tools to analyze and compare prompt versions:

1. Navigate to the "Process Stories" tab and select "Skill Analytics"
2. Use the "Version Comparison" tab to compare metrics across different prompt versions
3. The system will display metrics like average scores, match counts, and score distributions

## Command Line Usage

When using the command-line tools, you can specify the prompt version to use:

```bash
# For skill matching
python -m src.skill_matcher --prompt-version v2

# For intervention point generation
python -m src.intervention_point --prompt-version v2
```

If not specified, the system defaults to using `v1`.

## Modifying Prompt Templates

When modifying prompt templates, consider:

1. **Keeping the core functionality unchanged** - The prompt templates should maintain the same general structure to produce compatible results
2. **Testing thoroughly** - Run a small batch of stories first to ensure the new prompt works correctly
3. **Documenting changes** - Add comments in the prompt template files to explain your changes and the rationale

## Best Practices for A/B Testing

1. **Make targeted changes** - Change only one aspect of the prompt at a time to isolate its effect
2. **Test with a representative sample** - Process a diverse set of stories to ensure robust results
3. **Use quantitative and qualitative evaluation** - Look at both metrics and actual examples
4. **Document your findings** - Keep track of what changes worked and which didn't

## Future Enhancements

Potential future enhancements to the A/B testing framework include:

1. Automated statistical analysis of differences between prompt versions
2. Integration with model evaluation metrics
3. Support for testing different LLM models alongside prompt variations
4. Automated prompt optimization based on feedback

## Troubleshooting

If you encounter issues with the A/B testing framework:

1. **Missing prompt version** - Ensure that the prompt version directory exists and contains the required files
2. **Results not showing** - Check that the processed data directory for that version exists and contains the appropriate files
3. **Processing errors** - Check that the prompt templates in your new version follow the required structure

For further assistance, consult the project maintainers. 