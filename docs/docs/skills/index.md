---
sidebar_position: 1
---

# Skills

Marie-AI's skill system provides a filesystem-based framework for packaging domain expertise as reusable, discoverable capabilities. Skills follow the [ADK specification](https://agentskills.io) and use a progressive disclosure pattern to minimize token usage.

## Overview

Skills are multi-step capabilities stored as Markdown files with YAML frontmatter. They are organized into three tiers:

| Tier | Content | Token Cost | When Loaded |
|------|---------|------------|-------------|
| **L1 Metadata** | Name, description, tags | ~100 tokens/skill | Always available |
| **L2 Instructions** | Full instructions, examples | Fewer than 5,000 tokens | On demand via `load_skill` |
| **L3 Resources** | Reference docs, scripts, templates | Varies | On demand via `load_skill_resource` |

## Skill Sources

Skills are discovered from two locations:

| Source | Path | Writable | Purpose |
|--------|------|----------|---------|
| Built-in | `config/skills/` | No | Product skills shipped with Marie-AI |
| Workspace | `.marie/skills/` | Yes | User/agent-created skills |

Workspace skills take precedence over built-in skills with the same name.

## Directory Structure

Each skill lives in its own directory:

```
.marie/skills/
  my-skill/
    SKILL.md           # Required: frontmatter + instructions
    evals/
      evals.json       # Optional: evaluation test cases
    references/
      style-guide.md   # Optional: reference documentation
    scripts/
      helper.py        # Optional: executable scripts
    assets/
      template.md      # Optional: templates
```

## SKILL.md Format

```markdown
---
name: my-skill
description: What this skill does
version: 1.0.0
user-invokable: true
argument-hint: <target>
allowed-tools: Read Write Bash
providers:
  - openai
  - claude
tags:
  - code
  - review
---

## When to Use

Use this skill when you need to...

## Instructions

1. First, do this
2. Then do that
3. Finally, verify the result

## Examples

- Input: "Review this PR"
- Action: Run code review checklist
```

## Quick Start

```python
from marie.agent.skills import toolset  # Auto-registers tools

# Include skill tools in an agent
agent = BaseAgent(
    function_list=["discover_skills", "load_skill", "load_skill_resource"],
    llm=my_llm,
)
```

See [Agent Tools](./agent-tools) for the complete API reference.
