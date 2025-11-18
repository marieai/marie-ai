# Contributing to Marie MCP Server

Thank you for your interest in contributing to Marie MCP Server! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/marie-mcp-server.git
cd marie-mcp-server
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

4. **Set up environment**

```bash
cp .env.example .env
# Edit .env with your credentials
```

## Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking

Format your code before committing:

```bash
black src/
isort src/
mypy src/
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our code style

3. Add or update tests as needed

4. Run tests and linting:
   ```bash
   pytest tests/
   black src/
   isort src/
   mypy src/
   ```

5. Commit with clear messages:
   ```bash
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit prefixes:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `test:` - Test updates
   - `chore:` - Build/tooling changes

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request on GitHub

## Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## Adding New Tools

To add a new MCP tool:

1. Add the tool function to the appropriate file in `src/marie_mcp/tools/`
2. Use the `@mcp.tool()` decorator
3. Add type hints and docstrings
4. Register the tool in `server.py`
5. Update README.md with tool documentation
6. Add tests in `tests/`

Example:

```python
@mcp.tool()
async def my_new_tool(
    param: Annotated[str, Field(description="Parameter description")],
    ctx: Context = None,
) -> str:
    """
    Tool description here.

    Example:
        my_new_tool(param="value")
    """
    # Implementation
```

## Questions?

Feel free to open an issue for questions or discussion.

Thank you for contributing!
