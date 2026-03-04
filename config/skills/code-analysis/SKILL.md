---
name: code-analysis
description: >
  Analyze code for quality, security, performance, and best practices.
  Use when reviewing code, finding bugs, or suggesting improvements.
version: "1.0.0"
license: Apache-2.0
compatibility: Works with any LLM provider
allowed-tools: read_file search_code analyze_dependencies
user-invokable: true
argument-hint: "[file path or code block]"
providers:
  - openai
  - claude
  - vllm
tags:
  - code
  - analysis
  - review
  - security
metadata:
  author: marie-ai
  category: development
---

# Code Analysis Skill

Provide expert code analysis including quality review, security assessment, and performance optimization.

## When to Use

Use this skill when:
- User asks for a code review
- User wants to find bugs or issues in code
- User asks about code quality or best practices
- User wants security analysis of code
- User asks for performance optimization suggestions
- User shares code and asks "what's wrong with this?"

Do NOT use this skill when:
- User wants you to write new code from scratch (use coding skill)
- User is asking about general programming concepts without specific code
- User wants documentation written (use documentation skill)

## Instructions

1. **Understand the context**:
   - Identify the programming language
   - Understand the code's purpose
   - Note the framework/libraries being used

2. **Perform multi-dimensional analysis**:

   **Quality Analysis**:
   - Code structure and organization
   - Naming conventions
   - Code duplication (DRY principle)
   - Function/method length and complexity
   - Error handling patterns

   **Security Analysis**:
   - Input validation
   - SQL injection vulnerabilities
   - XSS vulnerabilities
   - Authentication/authorization issues
   - Sensitive data exposure
   - Dependency vulnerabilities

   **Performance Analysis**:
   - Algorithm complexity (Big O)
   - Memory usage patterns
   - Database query efficiency
   - Caching opportunities
   - Unnecessary computations

   **Best Practices**:
   - Language-specific idioms
   - Framework conventions
   - Testing considerations
   - Documentation gaps

3. **Prioritize findings**:
   - Critical: Security vulnerabilities, bugs causing data loss
   - High: Bugs, significant performance issues
   - Medium: Code quality issues, maintainability concerns
   - Low: Style issues, minor improvements

4. **Provide actionable feedback**:
   - Explain WHY something is an issue
   - Show HOW to fix it with code examples
   - Reference relevant documentation or standards

## Examples

**User**: "Review this Python function"
**Action**: Analyze for Pythonic patterns, type hints, error handling, edge cases

**User**: "Is this code secure?"
**Action**: Focus on security analysis - injection, auth, data exposure

**User**: "Why is this slow?"
**Action**: Focus on performance - complexity, queries, loops, caching

**User**: "What's wrong with this?"
**Action**: Full analysis starting with most critical issues

## Output Format

```
## Code Analysis Report

### Summary
[Brief overview of findings]

### Critical Issues
1. **SQL Injection Vulnerability** (line 23)
   - Issue: User input directly interpolated into SQL query
   - Fix: Use parameterized queries
   ```python
   # Before (vulnerable)
   cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

   # After (safe)
   cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
   ```

### High Priority
[List of high priority issues with fixes]

### Recommendations
[Medium/low priority improvements]

### Positive Observations
[What the code does well - maintain morale!]
```

## Language-Specific Guidance

**Python**: Check for type hints, f-strings, context managers, list comprehensions
**JavaScript/TypeScript**: Check for async/await patterns, null checks, type safety
**Java**: Check for null handling, resource management, generics usage
**Go**: Check for error handling, goroutine safety, interface usage
**Rust**: Check for ownership, lifetimes, unsafe blocks

## Security Checklist

Always check for:
- [ ] Input validation on all user inputs
- [ ] Parameterized queries for database access
- [ ] Output encoding for web responses
- [ ] Authentication on protected endpoints
- [ ] Authorization checks before actions
- [ ] Secure password handling
- [ ] Sensitive data not logged
- [ ] Dependencies up to date
