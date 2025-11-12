"""Example: Using Marie MCP with OpenAI Agents SDK."""

from agents import Agent
from agents.mcp import MCPServerStdio


def main():
    """Example OpenAI Agents integration with Marie MCP."""

    # Create Marie MCP server connection
    print("Connecting to Marie MCP server...")
    marie_server = MCPServerStdio(
        command="uvx",
        args=["marie-mcp"],
        env={
            "MARIE_BASE_URL": "http://localhost:5000",
            "MARIE_API_KEY": "your-api-key-here",
            "AWS_ACCESS_KEY_ID": "your-aws-key",
            "AWS_SECRET_ACCESS_KEY": "your-aws-secret",
            "S3_BUCKET": "marie",
        },
    )

    # Create agent with Marie tools
    agent = Agent(
        name="DocumentProcessor",
        instructions="""You are a document processing assistant using Marie AI.
        You can:
        - Extract text from documents using OCR
        - Extract structured data using templates
        - Monitor job status and retrieve results
        - Check system health and capacity

        Always provide clear feedback about job submission and status.""",
        tools=marie_server.get_tools(),
        model="gpt-4",
    )

    # Example 1: OCR extraction
    print("\n" + "=" * 60)
    print("Example 1: OCR Extraction")
    print("=" * 60)

    result = agent.run(
        "Extract text from invoice.pdf. Use ref_id 'inv_12345' and ref_type 'invoice'"
    )
    print(f"Agent response: {result}")

    # Example 2: Template-based extraction
    print("\n" + "=" * 60)
    print("Example 2: Template-Based Data Extraction")
    print("=" * 60)

    result = agent.run(
        "Extract data from medical_form.pdf using template ID 117183. "
        "Use ref_id 'form_001' and ref_type 'medical_form'"
    )
    print(f"Agent response: {result}")

    # Example 3: Job monitoring
    print("\n" + "=" * 60)
    print("Example 3: Monitor Jobs")
    print("=" * 60)

    result = agent.run("Show me all active jobs and their status")
    print(f"Agent response: {result}")

    # Example 4: System monitoring
    print("\n" + "=" * 60)
    print("Example 4: System Health")
    print("=" * 60)

    result = agent.run("Check system health and tell me about available capacity")
    print(f"Agent response: {result}")


if __name__ == "__main__":
    main()
