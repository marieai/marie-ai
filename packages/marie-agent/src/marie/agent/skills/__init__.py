"""Marie Agent Skills System.

This package provides a Claude Code-like skills architecture following
the agentskills.io specification. Skills are behavioral instructions
that enhance how agents approach tasks, distinct from tools which
execute actions.

Key Concepts:
- **Skills**: Packaged instructions + expert knowledge that modify model reasoning
- **Tools**: Actionable capabilities that execute operations

Example:
    ```python
    from marie.agent.skills import (
        SKILL_REGISTRY,
        initialize_skills,
        get_skill,
        SkillRouter,
    )

    # Initialize skills on startup
    initialize_skills(workspace_path="/path/to/workspace")

    # Get a specific skill
    skill = get_skill("document-extraction")
    print(skill.description)

    # Route a message to a skill
    router = SkillRouter()
    context = await router.route("Extract data from this invoice")
    if context.skill:
        instructions = context.skill.get_instructions()
    ```

Skill Definition (SKILL.md):
    ```yaml
    ---
    name: document-extraction
    description: Extract structured data from documents using OCR and AI
    allowed-tools: extract_document_ocr extract_document_data
    ---

    ## When to Use
    Use this skill when extracting text or data from PDFs, images, or documents.

    ## Instructions
    1. Identify document type
    2. Select extraction template
    3. Call extract_document_ocr for text
    4. Use extract_document_data for structured fields
    ```
"""

from marie.agent.skills.loader import (
    SkillLoader,
    get_skill_loader,
    initialize_skills,
)
from marie.agent.skills.models import (
    Skill,
    SkillContext,
    SkillExample,
    SkillInstructions,
    SkillMetadata,
    SkillResources,
    SkillSource,
)
from marie.agent.skills.parser import (
    frontmatter_to_metadata,
    parse_frontmatter,
    parse_skill,
    parse_skill_file,
    validate_skill_structure,
)
from marie.agent.skills.registry import (
    SKILL_REGISTRY,
    SkillNotFoundError,
    SkillRegistry,
    get_skill,
    list_skills,
    register_skill,
)
from marie.agent.skills.router import (
    SkillRouter,
)
from marie.agent.skills.search import (
    SkillSearchIndex,
)
from marie.agent.skills.toolset import (
    SkillToolset,
    discover_skills,
    load_skill,
    load_skill_resource,
)
from marie.agent.skills.validator import (
    ValidationError,
    ValidationResult,
    validate_frontmatter,
    validate_skill,
    validate_skill_directory,
)

__all__ = [
    # Core classes
    "Skill",
    "SkillMetadata",
    "SkillInstructions",
    "SkillResources",
    "SkillContext",
    "SkillExample",
    "SkillSource",
    # Registry
    "SkillRegistry",
    "SKILL_REGISTRY",
    "SkillNotFoundError",
    "register_skill",
    "get_skill",
    "list_skills",
    # Router
    "SkillRouter",
    # Search
    "SkillSearchIndex",
    # Loader
    "SkillLoader",
    "get_skill_loader",
    "initialize_skills",
    # Toolset (agent-facing tools)
    "SkillToolset",
    "discover_skills",
    "load_skill",
    "load_skill_resource",
    # Parser
    "parse_skill",
    "parse_skill_file",
    "parse_frontmatter",
    "frontmatter_to_metadata",
    "validate_skill_structure",
    # Validator
    "ValidationResult",
    "ValidationError",
    "validate_skill",
    "validate_skill_directory",
    "validate_frontmatter",
]
