# pic-splitter Development Guide

## Commands
- Install: `uv sync`
- Add: `uv add <package_name>`
- Run: `pic-splitter <image_path>`

## Code Style Guidelines
- **Imports**: Group imports: stdlib, third-party, local (alphabetical within groups)
- **Formatting**: Use Black-compatible formatting (handled by ruff format)
- **Type Hints**: Include type hints for all function parameters and returns
- **Naming**:
  - Functions/variables: snake_case
  - Classes: PascalCase
  - Constants: UPPER_SNAKE_CASE
- **Error Handling**: Use clear error messages and appropriate exceptions
- **Documentation**: Include docstrings for all functions and classes. Use lowercase for docstrings.
- **Version Control**: Use git for version control, with clear concise commit messages
- **Comments**: Use comments to explain complex logic, but avoid obvious comments. Keep them concise. Use lowercase.
- **Logging**: Use loguru for logging, with appropriate log levels (debug, info, warning, error, critical). Any strings in logs should be lowercase.
- **Testing**: Use pytest for unit tests

This tool processes images using computer vision algorithms to segment them into component parts, applying texture and color-based analysis.
