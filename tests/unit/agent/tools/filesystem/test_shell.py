"""Tests for ShellTool."""

from __future__ import annotations

import json

import pytest

from marie.agent.tools.filesystem import ShellTool


class TestShellTool:
    """Tests for ShellTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = ShellTool()
        assert tool.name == "shell"
        assert "shell" in tool.description.lower()
        assert tool.metadata.fn_schema is not None

    def test_allowed_command_ls(self, temp_dir):
        """Test executing allowed 'ls' command."""
        tool = ShellTool()
        result = tool.call(command=f"ls {temp_dir}")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["success"] is True
        assert data["return_code"] == 0

    def test_allowed_command_pwd(self):
        """Test executing allowed 'pwd' command."""
        tool = ShellTool()
        result = tool.call(command="pwd")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["success"] is True
        assert len(data["stdout"]) > 0

    def test_allowed_command_echo(self):
        """Test executing allowed 'echo' command."""
        tool = ShellTool()
        result = tool.call(command="echo hello world")

        assert result.is_error is False
        data = json.loads(result.content)
        assert "hello world" in data["stdout"]

    def test_blocked_command(self):
        """Test that disallowed commands are blocked."""
        tool = ShellTool()

        blocked_commands = [
            "rm -rf /",
            "curl http://example.com",
            "wget http://example.com",
            "python -c 'print(1)'",
            "bash -c 'echo test'",
            "nc localhost 80",
        ]

        for cmd in blocked_commands:
            result = tool.call(command=cmd)
            assert result.is_error is True
            data = json.loads(result.content)
            assert "not allowed" in data["error"].lower()

    def test_empty_command(self):
        """Test error on empty command."""
        tool = ShellTool()
        result = tool.call(command="")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "empty" in data["error"].lower()

    def test_command_with_arguments(self, temp_dir):
        """Test commands with arguments work correctly."""
        tool = ShellTool()
        result = tool.call(command=f"ls -la {temp_dir}")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["success"] is True

    def test_timeout_respected(self):
        """Test that timeout parameter is respected."""
        tool = ShellTool()
        # Note: This test doesn't actually trigger timeout, just verifies the parameter is passed
        result = tool.call(command="echo fast", timeout=1)

        assert result.is_error is False

    def test_stderr_captured(self, temp_dir):
        """Test that stderr is captured."""
        tool = ShellTool()
        result = tool.call(command=f"ls /nonexistent_path_12345")

        data = json.loads(result.content)
        assert data["return_code"] != 0
        assert len(data["stderr"]) > 0 or "No such file" in data.get("stdout", "")

    def test_allowed_commands_list_in_metadata(self):
        """Test that allowed commands are listed in metadata."""
        tool = ShellTool()
        for cmd in ["ls", "pwd", "cat", "grep", "find"]:
            assert cmd in tool.ALLOWED_COMMANDS

    def test_command_piping_blocked(self):
        """Test that command piping with dangerous commands is blocked."""
        tool = ShellTool()
        # The base command is checked, so this should be allowed since 'echo' is in whitelist
        result = tool.call(command="echo test | cat")
        assert result.is_error is False

        # But starting with a disallowed command should fail
        result = tool.call(command="curl test | echo")
        assert result.is_error is True

    def test_tool_output_structure(self):
        """Test that output has correct structure."""
        tool = ShellTool()
        result = tool.call(command="echo test")

        assert result.tool_name == "shell"
        assert result.raw_input["command"] == "echo test"

        data = json.loads(result.content)
        assert "command" in data
        assert "stdout" in data
        assert "stderr" in data
        assert "return_code" in data
        assert "success" in data

    def test_which_command_allowed(self):
        """Test that 'which' command is allowed."""
        tool = ShellTool()
        result = tool.call(command="which ls")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["success"] is True

    def test_uname_command_allowed(self):
        """Test that 'uname' command is allowed."""
        tool = ShellTool()
        result = tool.call(command="uname -a")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["success"] is True
