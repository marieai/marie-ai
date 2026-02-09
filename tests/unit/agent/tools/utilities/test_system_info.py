"""Tests for SystemInfoTool."""

from __future__ import annotations

import json
import os
import platform
import sys

import pytest

from marie.agent.tools.utilities import SystemInfoTool


class TestSystemInfoTool:
    """Tests for SystemInfoTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = SystemInfoTool()
        assert tool.name == "system_info"
        assert "system" in tool.description.lower()

    def test_returns_python_version(self):
        """Test that Python version is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        assert result.is_error is False
        data = json.loads(result.content)
        assert "python_version" in data
        assert data["python_version"] == sys.version.split()[0]

    def test_returns_platform_info(self):
        """Test that platform information is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "platform" in data
        assert "system" in data
        assert "machine" in data
        assert data["system"] == platform.system()

    def test_returns_cwd(self):
        """Test that current working directory is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "cwd" in data
        assert data["cwd"] == os.getcwd()

    def test_returns_user(self):
        """Test that user information is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "user" in data
        expected_user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
        assert data["user"] == expected_user

    def test_returns_home_directory(self):
        """Test that home directory is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "home" in data

    def test_returns_env_var_count(self):
        """Test that environment variable count is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "env_var_count" in data
        assert data["env_var_count"] == len(os.environ)

    def test_no_error_on_call(self):
        """Test that the tool doesn't error on normal call."""
        tool = SystemInfoTool()
        result = tool.call()

        assert result.is_error is False
        assert result.tool_name == "system_info"

    def test_raw_output_matches_content(self):
        """Test that raw_output matches parsed content."""
        tool = SystemInfoTool()
        result = tool.call()

        parsed = json.loads(result.content)
        assert result.raw_output == parsed

    def test_content_is_formatted_json(self):
        """Test that content is formatted JSON (indented)."""
        tool = SystemInfoTool()
        result = tool.call()

        # The content should be indented (contain newlines)
        assert "\n" in result.content
        # And should be valid JSON
        data = json.loads(result.content)
        assert isinstance(data, dict)

    def test_python_implementation(self):
        """Test that Python implementation is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "python_implementation" in data
        assert data["python_implementation"] == platform.python_implementation()

    def test_release_info(self):
        """Test that OS release info is returned."""
        tool = SystemInfoTool()
        result = tool.call()

        data = json.loads(result.content)
        assert "release" in data
        assert data["release"] == platform.release()
