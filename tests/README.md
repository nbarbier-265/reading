# AMIRA Project Test Suite

This directory contains a comprehensive test suite for the AMIRA Streamlit application, including unit tests, integration tests, and test fixtures.

## Test Structure

The tests are organized as follows:

- **Unit Tests**: Located in `tests/unit/` - These tests validate individual functions and classes in isolation.
- **Integration Tests**: Located in `tests/integration/` - These tests validate how different components work together.
- **Fixtures**: Located in `tests/fixtures/` - Contains mock data and shared test fixtures.

## Running the Tests

### Prerequisites

Make sure you have installed all the testing dependencies:

```bash
pip install pytest pytest-mock pytest-cov
```

### Running All Tests

To run the complete test suite with a coverage report:

```bash
pytest --cov=app
```

### Running Specific Test Categories

You can run specific categories of tests using the markers defined in `pytest.ini`:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests for a specific module
pytest tests/unit/test_models.py
```

### Generating a Detailed Coverage Report

For a detailed HTML coverage report:

```bash
pytest --cov=app --cov-report=html
```

This will create a directory named `htmlcov`. Open `htmlcov/index.html` in your browser to view the detailed coverage report.

## Test Fixtures

The test fixtures are defined in `conftest.py` and provide common test data and mock objects used across multiple tests:

- **Mock Streamlit**: Prevents Streamlit UI side effects during tests
- **Sample DataFrames**: Pre-populated DataFrames for testing
- **Mock Objects**: Various mock implementations for file handling, API calls, etc.

## Adding New Tests

When adding new tests:

1. Follow the existing structure and naming conventions
2. Use appropriate markers for your tests
3. Utilize the existing fixtures when possible
4. Make sure your tests are deterministic and do not depend on external resources
5. For UI components, always mock the Streamlit API

## Best Practices

- Each test function should focus on testing a single aspect of functionality
- Tests should be independent of each other
- Use descriptive names for test functions that explain what they're testing
- Write assertions that clearly show what you're expecting
- Use appropriate mocking to avoid external dependencies

## Notes

- No external API calls are made during testing
- The tests use mock data from the fixtures directory
- Environment variables are mocked where necessary 