"""Planning Agent Example.

This example demonstrates the PlanningAgent which creates and executes
multi-step plans for complex tasks, using tools that perform real operations.

Shows:
- Multi-step planning and execution
- Real file operations, data analysis, and reporting
- Tool orchestration for complex workflows
- Progress tracking and error handling

Available Tools:
    - list_files: List files with glob patterns, recursive option
    - read_file: Read file contents (with size limits)
    - write_file: Write content to files
    - analyze_text: Text analysis (summary, words, patterns)
    - analyze_code: Code metrics (functions, classes, imports)
    - create_csv: Create CSV files from data
    - read_csv: Read and parse CSV files
    - generate_report: Generate formatted reports (markdown/text/html)
    - run_calculation: Math expressions with variables

Usage Examples:
    # File listing and analysis
    python planning_agent.py --task "List all Python files in current directory"
    python planning_agent.py --task "Find all .py files recursively"
    python planning_agent.py --task "List files matching *.txt"

    # Code analysis
    python planning_agent.py --task "Analyze the code structure of agent_simple.py"
    python planning_agent.py --task "Count functions and classes in planning_agent.py"
    python planning_agent.py --task "Find the largest Python file and analyze it"

    # Text analysis
    python planning_agent.py --task "Analyze word frequency in README.md"
    python planning_agent.py --task "Find all email addresses in config files"

    # Report generation
    python planning_agent.py --task "Create a report about Python files here"
    python planning_agent.py --task "Generate a markdown summary of the codebase"

    # Multi-step complex tasks
    python planning_agent.py --task "List Python files, analyze the largest one, and summarize"
    python planning_agent.py --task "Find all TODO comments in Python files"
    python planning_agent.py --task "Create a CSV of all Python files with their line counts"

    # Calculations
    python planning_agent.py --task "Calculate the average size of Python files"

    # Interactive mode
    python planning_agent.py --tui

    # With OpenAI backend
    python planning_agent.py --backend openai --task "Analyze Python files"

    # Debug mode
    python planning_agent.py --task "List files" --debug
"""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from marie.agent import (
    AgentTool,
    PlanAndExecuteAgent,
    ToolMetadata,
    ToolOutput,
    register_tool,
)
from utils import print_debug_response

# Load environment variables from .env file
load_dotenv()


@register_tool("list_files")
def list_files(
    directory: str = ".", pattern: str = "*", recursive: bool = False
) -> str:
    """List files in a directory with optional pattern matching.

    Args:
        directory: Directory to list (default: current directory)
        pattern: Glob pattern to match (e.g., "*.py", "*.txt")
        recursive: If True, search recursively

    Returns:
        JSON string with file listing.
    """
    try:
        path = Path(directory)
        if not path.exists():
            return json.dumps({"error": f"Directory not found: {directory}"})

        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        file_info = []
        for f in files[:100]:  # Limit to 100 files
            if f.is_file():
                stat = f.stat()
                file_info.append(
                    {
                        "path": str(f),
                        "name": f.name,
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        return json.dumps(
            {
                "directory": str(path.absolute()),
                "pattern": pattern,
                "recursive": recursive,
                "total_found": len(files),
                "files": file_info,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("read_file")
def read_file(file_path: str, max_lines: int = 200) -> str:
    """Read contents of a file.

    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read

    Returns:
        JSON string with file contents.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})

        # Security check
        sensitive = [".env", "password", "secret", "credential", ".ssh"]
        if any(s in file_path.lower() for s in sensitive):
            return json.dumps({"error": "Cannot read sensitive files"})

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        total_lines = len(lines)
        content = ''.join(lines[:max_lines])

        return json.dumps(
            {
                "file_path": str(path.absolute()),
                "content": content,
                "total_lines": total_lines,
                "truncated": total_lines > max_lines,
                "size_bytes": path.stat().st_size,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "file_path": file_path})


@register_tool("write_file")
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """Write content to a file.

    Args:
        file_path: Path to write to
        content: Content to write
        append: If True, append instead of overwrite

    Returns:
        JSON string with write status.
    """
    try:
        # Security: only allow relative paths or /tmp
        if file_path.startswith("/") and not file_path.startswith("/tmp"):
            return json.dumps({"error": "Can only write to relative paths or /tmp"})

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if append else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)

        return json.dumps(
            {
                "success": True,
                "file_path": str(path.absolute()),
                "bytes_written": len(content),
                "mode": "append" if append else "write",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("analyze_text")
def analyze_text(text: str, analysis_type: str = "summary") -> str:
    """Analyze text content and extract statistics/insights.

    Args:
        text: Text to analyze
        analysis_type: Type of analysis ("summary", "words", "lines", "patterns")

    Returns:
        JSON string with analysis results.
    """
    lines = text.split('\n')
    words = text.split()

    # Basic stats
    stats = {
        "total_chars": len(text),
        "total_words": len(words),
        "total_lines": len(lines),
        "non_empty_lines": len([l for l in lines if l.strip()]),
        "avg_line_length": round(len(text) / max(len(lines), 1), 2),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
    }

    if analysis_type == "words":
        # Word frequency
        word_counts = Counter(w.lower() for w in words if w.isalpha())
        stats["top_words"] = dict(word_counts.most_common(20))
        stats["unique_words"] = len(word_counts)

    elif analysis_type == "patterns":
        # Find patterns
        stats["emails"] = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        stats["urls"] = re.findall(r'https?://\S+', text)
        stats["numbers"] = re.findall(r'\b\d+(?:\.\d+)?\b', text)[:50]
        stats["dates"] = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)

    return json.dumps({"analysis_type": analysis_type, "statistics": stats})


@register_tool("analyze_code")
def analyze_code(file_path: str) -> str:
    """Analyze a source code file for metrics.

    Args:
        file_path: Path to the code file

    Returns:
        JSON string with code analysis.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')

        ext = Path(file_path).suffix.lower()

        # Basic metrics
        metrics = {
            "file_path": file_path,
            "extension": ext,
            "total_lines": len(lines),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "code_lines": len(
                [l for l in lines if l.strip() and not l.strip().startswith('#')]
            ),
        }

        # Language-specific analysis
        if ext == ".py":
            metrics["imports"] = len(
                re.findall(r'^(?:import|from)\s+', content, re.MULTILINE)
            )
            metrics["functions"] = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
            metrics["classes"] = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
            metrics["comments"] = len(re.findall(r'#.*$', content, re.MULTILINE))
            metrics["docstrings"] = len(
                re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', content)
            )
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            metrics["imports"] = len(re.findall(r'^import\s+', content, re.MULTILINE))
            metrics["functions"] = len(
                re.findall(r'function\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\(', content)
            )
            metrics["classes"] = len(re.findall(r'class\s+\w+', content))
        elif ext in (".java", ".kt"):
            metrics["imports"] = len(re.findall(r'^import\s+', content, re.MULTILINE))
            metrics["classes"] = len(re.findall(r'class\s+\w+', content))
            metrics["methods"] = len(
                re.findall(r'(?:public|private|protected)\s+\w+\s+\w+\s*\(', content)
            )

        return json.dumps(metrics)
    except Exception as e:
        return json.dumps({"error": str(e), "file_path": file_path})


@register_tool("create_csv")
def create_csv(file_path: str, headers: str, rows: str) -> str:
    """Create a CSV file from data.

    Args:
        file_path: Path to save CSV
        headers: Comma-separated header names
        rows: JSON array of row arrays

    Returns:
        JSON string with creation status.
    """
    try:
        if file_path.startswith("/") and not file_path.startswith("/tmp"):
            return json.dumps({"error": "Can only write to relative paths or /tmp"})

        header_list = [h.strip() for h in headers.split(",")]
        row_data = json.loads(rows)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            writer.writerows(row_data)

        return json.dumps(
            {
                "success": True,
                "file_path": str(path.absolute()),
                "headers": header_list,
                "rows_written": len(row_data),
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("read_csv")
def read_csv(file_path: str, max_rows: int = 100) -> str:
    """Read a CSV file and return its contents.

    Args:
        file_path: Path to CSV file
        max_rows: Maximum rows to read

    Returns:
        JSON string with CSV data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(dict(row))

        return json.dumps(
            {
                "file_path": file_path,
                "headers": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "total_rows": len(rows),
                "truncated": len(rows) >= max_rows,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "file_path": file_path})


@register_tool("generate_report")
def generate_report(
    title: str, sections: str, output_path: str = "", format: str = "markdown"
) -> str:
    """Generate a formatted report from sections.

    Args:
        title: Report title
        sections: JSON object with section_name: content pairs
        output_path: Optional path to save report
        format: Output format ("markdown", "text", "html")

    Returns:
        JSON string with generated report.
    """
    try:
        section_data = json.loads(sections)
    except json.JSONDecodeError:
        section_data = {"Content": sections}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if format == "markdown":
        report = f"# {title}\n\n"
        report += f"*Generated: {timestamp}*\n\n"
        for name, content in section_data.items():
            report += f"## {name}\n\n{content}\n\n"
    elif format == "html":
        report = f"<html><head><title>{title}</title></head><body>\n"
        report += f"<h1>{title}</h1>\n<p><em>Generated: {timestamp}</em></p>\n"
        for name, content in section_data.items():
            report += f"<h2>{name}</h2>\n<p>{content}</p>\n"
        report += "</body></html>"
    else:
        report = f"{title}\n{'=' * len(title)}\n\n"
        report += f"Generated: {timestamp}\n\n"
        for name, content in section_data.items():
            report += f"{name}\n{'-' * len(name)}\n{content}\n\n"

    result = {
        "title": title,
        "format": format,
        "content": report,
        "sections": list(section_data.keys()),
        "timestamp": timestamp,
    }

    if output_path:
        try:
            if not output_path.startswith("/") or output_path.startswith("/tmp"):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                result["saved_to"] = output_path
        except Exception as e:
            result["save_error"] = str(e)

    return json.dumps(result)


@register_tool("run_calculation")
def run_calculation(expression: str, variables: str = "{}") -> str:
    """Evaluate a mathematical expression with variables.

    Args:
        expression: Math expression (e.g., "sum(values) / len(values)")
        variables: JSON object with variable names and values

    Returns:
        JSON string with calculation result.
    """
    import math

    try:
        vars_dict = json.loads(variables)
    except json.JSONDecodeError:
        vars_dict = {}

    safe_builtins = {
        "sum": sum,
        "len": len,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "list": list,
        "range": range,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, {**safe_builtins, **vars_dict})
        return json.dumps(
            {
                "expression": expression,
                "variables": vars_dict,
                "result": result,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


def create_planning_agent(
    backend: str = "marie", model: Optional[str] = None
) -> PlanAndExecuteAgent:
    """Create a planning agent with real tools.

    Args:
        backend: LLM backend ("marie" or "openai")
        model: Model name

    Returns:
        Configured PlanAndExecuteAgent instance.
    """
    from utils import create_llm

    llm = create_llm(backend=backend, model=model)

    tools = [
        "list_files",
        "read_file",
        "write_file",
        "analyze_text",
        "analyze_code",
        "create_csv",
        "read_csv",
        "generate_report",
        "run_calculation",
    ]

    return PlanAndExecuteAgent(
        llm=llm,
        function_list=tools,
        name="Planning Agent",
        description="An agent that creates and executes multi-step plans using real tools.",
        system_message="""You are a planning assistant that breaks down complex tasks into steps.

When given a task:
1. Create a numbered PLAN with clear steps
2. Execute each step using available tools
3. Verify results before proceeding
4. Provide a FINAL ANSWER with summary

Available tools:
- **list_files**: List files in a directory (supports patterns like "*.py")
- **read_file**: Read file contents
- **write_file**: Write content to files
- **analyze_text**: Analyze text (summary, words, patterns)
- **analyze_code**: Analyze source code files
- **create_csv**: Create CSV files from data
- **read_csv**: Read CSV files
- **generate_report**: Generate formatted reports (markdown, text, html)
- **run_calculation**: Evaluate math expressions

Format:
PLAN:
1. [Step description]
2. [Step description]
...

STEP 1: [Action]
[Tool calls and results]

FINAL ANSWER:
[Summary of completed task]""",
        max_iterations=15,
    )


def run_task(task: str, backend: str = "marie", debug: bool = False):
    """Run a planning task."""
    print("=" * 60)
    print("PLANNING AGENT")
    print("=" * 60)
    print(f"\nTask: {task}")
    if debug:
        print(f"Backend: {backend}")
        print("Debug: ON")
    print("-" * 60)

    agent = create_planning_agent(backend=backend)
    messages = [{"role": "user", "content": task}]

    iteration = 0
    for responses in agent.run(messages=messages):
        if responses:
            iteration += 1
            if debug:
                for resp in responses:
                    print_debug_response(resp, iteration)

            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(f"\n{content}")
                print()


def run_interactive(backend: str = "marie", debug: bool = False):
    """Run in interactive mode."""
    print("=" * 60)
    print("Planning Agent - Interactive Mode")
    print("=" * 60)
    if debug:
        print("Debug: ON")
    print("Enter complex tasks for the agent to plan and execute.")
    print("Commands: 'quit', 'clear'")
    print()

    agent = create_planning_agent(backend=backend)
    messages = []

    while True:
        try:
            user_input = input("\nTask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            messages = []
            print("Cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        print("\n" + "-" * 40)

        iteration = 0
        response_list = []
        for response_list in agent.run(messages=messages):
            if response_list:
                iteration += 1
                if debug:
                    for resp in response_list:
                        print_debug_response(resp, iteration)

                last = response_list[-1]
                content = (
                    last.get("content", "") if isinstance(last, dict) else last.content
                )
                if content:
                    print(content)

        print("-" * 40)

        if response_list:
            for r in response_list:
                messages.append(r if isinstance(r, dict) else r.model_dump())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning Agent Example")
    parser.add_argument("--task", "-t", type=str, help="Task to execute")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show raw LLM responses"
    )

    args = parser.parse_args()

    if args.tui:
        run_interactive(backend=args.backend, debug=args.debug)
    elif args.task:
        run_task(args.task, backend=args.backend, debug=args.debug)
    else:
        print("Planning Agent - Multi-step Task Execution")
        print("=" * 60)
        print()
        print("AVAILABLE TOOLS:")
        print("  list_files      - List files with glob patterns")
        print("  read_file       - Read file contents")
        print("  write_file      - Write to files")
        print("  analyze_text    - Text analysis (words, patterns)")
        print("  analyze_code    - Code metrics (functions, classes)")
        print("  create_csv      - Create CSV files")
        print("  read_csv        - Read CSV files")
        print("  generate_report - Generate reports (md/txt/html)")
        print("  run_calculation - Math with variables")
        print()
        print("USAGE EXAMPLES:")
        print()
        print("  # File operations")
        print("  python planning_agent.py --task 'List Python files'")
        print("  python planning_agent.py --task 'Find files matching *.txt'")
        print()
        print("  # Code analysis")
        print("  python planning_agent.py --task 'Analyze agent_simple.py'")
        print("  python planning_agent.py --task 'Find the largest Python file'")
        print()
        print("  # Report generation")
        print("  python planning_agent.py --task 'Create a report about files here'")
        print()
        print("  # Multi-step tasks")
        print("  python planning_agent.py --task 'List Python files, analyze largest'")
        print("  python planning_agent.py --task 'Create CSV of file sizes'")
        print()
        print("  # Interactive / OpenAI / Debug")
        print("  python planning_agent.py --tui")
        print("  python planning_agent.py --backend openai --task '...'")
        print("  python planning_agent.py --task '...' --debug")
