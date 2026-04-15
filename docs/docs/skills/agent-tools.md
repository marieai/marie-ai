---
sidebar_position: 2
---

# Agent Tools

Marie-AI provides three agent-facing tools that allow agents to dynamically discover and load skills during execution. These tools follow the ADK progressive disclosure pattern to minimize token consumption.

## Tool Reference

### discover_skills

**Level 1** — Returns lightweight metadata for skill discovery and routing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tags` | `List[str]` | `None` | Filter skills by tags (any match) |
| `query` | `str` | `None` | Search query against names and descriptions |
| `limit` | `int` | `20` | Maximum number of results |

**Returns:** JSON array of skill metadata objects.

**Example:**

```python
from marie.agent.skills.toolset import discover_skills

# List all skills
result = discover_skills()

# Search by query
result = discover_skills(query="code review")

# Filter by tags
result = discover_skills(tags=["document", "extraction"])
```

**Response:**

```json
[
  {
    "name": "code-review",
    "description": "Review code for best practices and bugs",
    "tags": ["code", "review", "quality"],
    "user_invokable": true,
    "allowed_tools": ["Read", "Bash"]
  }
]
```

### load_skill

**Level 2** — Loads full instructions for a specific skill.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Exact skill name |
| `include_full_content` | `bool` | `False` | Include raw markdown content |

**Returns:** JSON object with parsed skill sections.

**Example:**

```python
from marie.agent.skills.toolset import load_skill

result = load_skill(name="code-review")
```

**Response:**

```json
{
  "name": "code-review",
  "description": "Review code for best practices and bugs",
  "when_to_use": "Use this skill when reviewing pull requests...",
  "instructions": "1. Read the changed files\n2. Check for...",
  "examples": [
    {"input": "Review this PR", "action": "Run code review checklist"}
  ],
  "allowed_tools": ["Read", "Bash"]
}
```

### load_skill_resource

**Level 3** — Loads a specific resource file from a skill.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skill_name` | `str` | *required* | Name of the skill |
| `resource_path` | `str` | *required* | Filename (e.g., `style-guide.md`) |
| `resource_type` | `str` | `"references"` | One of: `references`, `scripts`, `templates` |

**Returns:** File content as string, or error JSON if not found.

**Example:**

```python
from marie.agent.skills.toolset import load_skill_resource

content = load_skill_resource(
    skill_name="code-review",
    resource_path="style-guide.md",
    resource_type="references",
)
```

## Usage Patterns

### Auto-Registration

Skill tools are auto-registered when the toolset module is imported:

```python
from marie.agent.skills import toolset  # Auto-registers tools

agent = BaseAgent(
    function_list=["discover_skills", "load_skill", "load_skill_resource"],
    llm=my_llm,
)
```

### Using SkillToolset Helper

```python
from marie.agent.skills import SkillToolset

agent = BaseAgent(
    function_list=SkillToolset.get_tool_names(),
    llm=my_llm,
)
```

## Runtime Refresh

Workspace skills created at runtime (e.g., via M3 Forge UI) are discoverable without process restart:

```python
from marie.agent.skills.loader import get_skill_loader

loader = get_skill_loader()
count = loader.refresh_workspace_skills("/path/to/workspace")
# Returns number of skills discovered
```

The refresh operation:
1. Clears only workspace skills from the registry (built-in skills are preserved)
2. Re-scans the workspace `.marie/skills/` directory
3. Registers newly discovered skills

## Error Handling

All tools return JSON responses. When a skill or resource is not found, the response includes available alternatives:

```json
{
  "error": "Skill 'nonexistent' not found",
  "available_skills": ["code-review", "document-extraction"],
  "hint": "Use discover_skills() to see all available skills"
}
```
