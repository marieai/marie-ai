#!/usr/bin/env python3
"""Agent Tools Test Suite.

This script provides usage examples and tests for all agent tools.
Run specific tests or all tests to verify the agent framework is working.

Usage:
    # Run all tests
    python test_agents.py --all

    # Run specific example tests
    python test_agents.py --example agent_simple
    python test_agents.py --example agent_assistant
    python test_agents.py --example planning_agent
    python test_agents.py --example multi_agent_router

    # Run with OpenAI backend
    python test_agents.py --all --backend openai

    # List all available tests
    python test_agents.py --list
"""

import argparse
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# Test cases for each example
TEST_CASES = {
    "agent_simple": {
        "description": "Simple Agent - Basic tools starter template",
        "tools": ["add", "multiply", "get_time", "list_files", "read_file", "counter"],
        "tests": [
            ("add", "Add 5 and 3"),
            ("multiply", "Multiply 7 by 8"),
            ("get_time", "What time is it?"),
            ("list_files", "List Python files in current directory"),
            ("read_file", "Read the first few lines of agent_simple.py"),
            ("counter", "Increment the counter 3 times and tell me the value"),
        ],
    },
    "agent_assistant": {
        "description": "Assistant Agent - Comprehensive tool set",
        "tools": [
            "get_current_time",
            "calculator",
            "run_shell_command",
            "read_file",
            "write_file",
            "web_fetch",
            "system_info",
        ],
        "tests": [
            ("get_current_time", "What time is it in Tokyo?"),
            ("get_current_time", "What time is it in New York?"),
            ("calculator", "Calculate 15% tip on $85"),
            ("calculator", "What is sqrt(144) + 25?"),
            ("run_shell_command", "Run 'pwd' to show current directory"),
            ("run_shell_command", "List files with 'ls -la'"),
            ("read_file", "Read the contents of agent_assistant.py"),
            ("system_info", "What system am I running on?"),
        ],
    },
    "planning_agent": {
        "description": "Planning Agent - Multi-step task execution",
        "tools": [
            "list_files",
            "read_file",
            "write_file",
            "analyze_text",
            "analyze_code",
            "create_csv",
            "read_csv",
            "generate_report",
            "run_calculation",
        ],
        "tests": [
            ("list_files", "List all Python files in current directory"),
            ("analyze_code", "Analyze the code structure of agent_simple.py"),
            ("analyze_text", "Analyze the word frequency in planning_agent.py"),
            ("run_calculation", "Calculate the sum of [1, 2, 3, 4, 5]"),
            (
                "generate_report",
                "Create a markdown report about the current directory",
            ),
            (
                "multi_step",
                "List Python files, analyze the smallest one, and summarize findings",
            ),
        ],
    },
    "multi_agent_router": {
        "description": "Multi-Agent Router - Task delegation to specialists",
        "tools": ["get_time", "calculator", "list_files"],
        "agents": ["time_assistant", "math_assistant", "file_assistant"],
        "tests": [
            ("time_assistant", "What time is it?"),
            ("time_assistant", "What day of the week is it?"),
            ("math_assistant", "Calculate 25 * 4"),
            ("math_assistant", "What is 15% of 200?"),
            ("file_assistant", "List files in current directory"),
            ("file_assistant", "Show me the Python files here"),
            ("routing", "What time is it and how much is 10 + 20?"),
        ],
    },
    "document_agent": {
        "description": "Document Agent - Visual document understanding (requires Marie API)",
        "tools": [
            "document_ocr",
            "document_classifier",
            "table_extractor",
            "ner_extractor",
            "kv_extractor",
            "document_info",
            "check_marie_status",
        ],
        "tests": [
            ("check_marie_status", "Check if Marie API is available"),
            ("document_info", "Get info about a sample document"),
            ("document_ocr", "Extract text from a document image"),
            ("document_classifier", "Classify the document type"),
            ("ner_extractor", "Extract named entities from the document"),
        ],
        "note": "Requires Marie API server running (marie server start)",
    },
}


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_test_case(tool: str, query: str, index: int):
    """Print a test case."""
    print(f"\n  [{index}] Tool: {tool}")
    print(f"      Query: \"{query}\"")


def list_tests():
    """List all available tests."""
    print_header("Available Agent Tests")

    for example, info in TEST_CASES.items():
        print(f"\n{example}")
        print(f"  Description: {info['description']}")
        print(f"  Tools: {', '.join(info['tools'])}")
        if "agents" in info:
            print(f"  Agents: {', '.join(info['agents'])}")
        if "note" in info:
            print(f"  Note: {info['note']}")
        print(f"  Test cases: {len(info['tests'])}")


def run_test(
    example: str,
    backend: str = "marie",
    model: Optional[str] = None,
    test_index: Optional[int] = None,
):
    """Run tests for a specific example.

    Args:
        example: Example name (e.g., "agent_simple")
        backend: LLM backend ("marie" or "openai")
        model: Model name
        test_index: Specific test index to run (runs all if None)
    """
    if example not in TEST_CASES:
        print(f"Unknown example: {example}")
        print(f"Available: {', '.join(TEST_CASES.keys())}")
        return

    info = TEST_CASES[example]
    print_header(f"{example} - {info['description']}")
    print(f"\nBackend: {backend}")
    if model:
        print(f"Model: {model}")
    print(f"Tools: {', '.join(info['tools'])}")

    tests = info["tests"]
    if test_index is not None:
        if 0 <= test_index < len(tests):
            tests = [tests[test_index]]
        else:
            print(f"Invalid test index. Available: 0-{len(tests)-1}")
            return

    # Import and run the appropriate example
    try:
        if example == "agent_simple":
            from agent_simple import create_agent

            agent = create_agent(backend=backend, model=model)
        elif example == "agent_assistant":
            from agent_assistant import create_assistant

            agent = create_assistant(backend=backend, model=model)
        elif example == "planning_agent":
            from planning_agent import create_planning_agent

            agent = create_planning_agent(backend=backend, model=model)
        elif example == "multi_agent_router":
            from multi_agent_router import create_router

            agent = create_router(backend=backend, model=model)
        elif example == "document_agent":
            from document_agent import init_document_agent

            agent = init_document_agent(backend=backend, model=model)
        else:
            print(f"No runner implemented for: {example}")
            return
    except ImportError as e:
        print(f"Failed to import {example}: {e}")
        return
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Run each test
    for i, (tool, query) in enumerate(tests):
        print_test_case(tool, query, i + 1)
        print("      " + "-" * 50)

        try:
            messages = [{"role": "user", "content": query}]
            response_text = ""

            for responses in agent.run(messages=messages):
                if responses:
                    last = responses[-1]
                    content = (
                        last.get("content", "")
                        if isinstance(last, dict)
                        else last.content
                    )
                    if content:
                        response_text = content

            if response_text:
                # Indent response for readability
                for line in response_text.split("\n")[:10]:
                    print(f"      {line}")
                if response_text.count("\n") > 10:
                    print("      ...")
            else:
                print("      (No response)")

        except Exception as e:
            print(f"      ERROR: {e}")

        print()


def run_all_tests(backend: str = "marie", model: Optional[str] = None):
    """Run all test examples."""
    print_header("Running All Agent Tests")

    examples_to_test = [
        "agent_simple",
        "agent_assistant",
        "planning_agent",
        "multi_agent_router",
    ]

    # Skip document_agent unless Marie API is available
    try:
        import json

        from document_agent import check_marie_status

        status = json.loads(check_marie_status())
        if status.get("online"):
            examples_to_test.append("document_agent")
        else:
            print("\nSkipping document_agent (Marie API not available)")
    except Exception:
        print("\nSkipping document_agent (import failed)")

    for example in examples_to_test:
        run_test(example, backend=backend, model=model)


def run_quick_tests(backend: str = "marie", model: Optional[str] = None):
    """Run quick smoke tests (one test per example)."""
    print_header("Quick Smoke Tests")
    print(f"Backend: {backend}")

    quick_tests = [
        ("agent_simple", "Add 5 and 3"),
        ("agent_assistant", "Calculate 15% tip on $85"),
        ("planning_agent", "List Python files in current directory"),
        ("multi_agent_router", "What time is it?"),
    ]

    for example, query in quick_tests:
        print(f"\n{example}: \"{query}\"")
        print("-" * 50)

        try:
            if example == "agent_simple":
                from agent_simple import create_agent

                agent = create_agent(backend=backend, model=model)
            elif example == "agent_assistant":
                from agent_assistant import create_assistant

                agent = create_assistant(backend=backend, model=model)
            elif example == "planning_agent":
                from planning_agent import create_planning_agent

                agent = create_planning_agent(backend=backend, model=model)
            elif example == "multi_agent_router":
                from multi_agent_router import create_router

                agent = create_router(backend=backend, model=model)

            messages = [{"role": "user", "content": query}]
            for responses in agent.run(messages=messages):
                if responses:
                    last = responses[-1]
                    content = (
                        last.get("content", "")
                        if isinstance(last, dict)
                        else last.content
                    )
                    if content:
                        print(content[:200])
                        if len(content) > 200:
                            print("...")
            print("[OK]")

        except Exception as e:
            print(f"[FAILED] {e}")


def show_usage():
    """Show usage examples for all tools."""
    print_header("Agent Tools Usage Examples")

    print(
        """
AGENT_SIMPLE (agent_simple.py)
------------------------------
Basic starter template with simple tools.

  # Math operations
  python agent_simple.py --task "Add 5 and 3"
  python agent_simple.py --task "Multiply 7 by 8"

  # Time
  python agent_simple.py --task "What time is it?"

  # File operations
  python agent_simple.py --task "List Python files"
  python agent_simple.py --task "Read agent_simple.py"

  # Stateful counter
  python agent_simple.py --task "Increment counter twice"


ASSISTANT_BASIC (agent_assistant.py)
------------------------------------
Comprehensive assistant with real-world tools.

  # Time with timezones
  python agent_assistant.py --query "What time is it in Tokyo?"
  python agent_assistant.py --query "What time is it in New York?"

  # Calculator with percentages
  python agent_assistant.py --query "Calculate 15% tip on $85"
  python agent_assistant.py --query "What is sqrt(144) + 25?"

  # Shell commands (safe subset)
  python agent_assistant.py --query "Run pwd command"
  python agent_assistant.py --query "List files with ls"

  # System info
  python agent_assistant.py --query "What system am I on?"

  # With OpenAI backend
  python agent_assistant.py --backend openai --query "Calculate 15% tip on $85"


PLANNING_AGENT (planning_agent.py)
----------------------------------
Multi-step task planning and execution.

  # File analysis
  python planning_agent.py --task "List all Python files and count them"
  python planning_agent.py --task "Analyze the code structure of agent_simple.py"

  # Text analysis
  python planning_agent.py --task "Analyze word frequency in planning_agent.py"

  # Report generation
  python planning_agent.py --task "Create a report about Python files here"

  # Complex multi-step
  python planning_agent.py --task "List Python files, analyze the largest, summarize"


MULTI_AGENT_ROUTER (multi_agent_router.py)
------------------------------------------
Routes tasks to specialized sub-agents.

  # Time queries -> time_assistant
  python multi_agent_router.py --task "What time is it?"
  python multi_agent_router.py --task "What day is today?"

  # Math queries -> math_assistant
  python multi_agent_router.py --task "Calculate 25 * 4"
  python multi_agent_router.py --task "What is 15% of 200?"

  # File queries -> file_assistant
  python multi_agent_router.py --task "List files here"
  python multi_agent_router.py --task "Show Python files"


DOCUMENT_AGENT (document_agent.py)
----------------------------------
Visual document understanding (requires Marie API).

  # Check API status
  python document_agent.py --demo

  # Process a document
  python document_agent.py --document /path/to/doc.png --task "Extract all text"
  python document_agent.py --document /path/to/invoice.pdf --task "Find the total amount"

  # Interactive mode
  python document_agent.py --tui


INTERACTIVE MODE
----------------
All agents support interactive TUI mode:

  python agent_simple.py --tui
  python agent_assistant.py --tui
  python planning_agent.py --tui
  python multi_agent_router.py --tui
  python document_agent.py --tui
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agent Tools Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List all available tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--usage", action="store_true", help="Show usage examples")
    parser.add_argument(
        "--example",
        "-e",
        type=str,
        choices=list(TEST_CASES.keys()),
        help="Run tests for specific example",
    )
    parser.add_argument(
        "--test", "-t", type=int, help="Run specific test index (use with --example)"
    )
    parser.add_argument(
        "--backend",
        "-b",
        default="marie",
        choices=["marie", "openai"],
        help="LLM backend",
    )
    parser.add_argument("--model", "-m", type=str, help="Model name")

    args = parser.parse_args()

    # Change to examples/agents directory for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    if args.list:
        list_tests()
    elif args.usage:
        show_usage()
    elif args.all:
        run_all_tests(backend=args.backend, model=args.model)
    elif args.quick:
        run_quick_tests(backend=args.backend, model=args.model)
    elif args.example:
        run_test(
            args.example,
            backend=args.backend,
            model=args.model,
            test_index=args.test,
        )
    else:
        print("Agent Tools Test Suite")
        print("=" * 60)
        print()
        print("This script tests all agent examples and their tools.")
        print()
        print("QUICK START:")
        print("  python test_agents.py --usage      # Show all usage examples")
        print("  python test_agents.py --list       # List all tests")
        print("  python test_agents.py --quick      # Run quick smoke tests")
        print()
        print("RUN SPECIFIC EXAMPLE:")
        print("  python test_agents.py --example agent_simple")
        print("  python test_agents.py --example agent_assistant --backend openai")
        print()
        print("RUN ALL TESTS:")
        print("  python test_agents.py --all")
        print("  python test_agents.py --all --backend openai")
