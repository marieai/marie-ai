"""Example: Using Marie MCP with LangChain."""

import asyncio

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters


async def main():
    """Example LangChain integration with Marie MCP."""

    # Load Marie MCP tools
    print("Loading Marie MCP tools...")
    marie_tools = await load_mcp_tools(
        StdioServerParameters(
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
    )

    print(f"Loaded {len(marie_tools)} tools: {[t.name for t in marie_tools]}")

    # Create LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create prompt template
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # Create agent
    agent = create_react_agent(llm, marie_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=marie_tools, verbose=True, handle_parsing_errors=True
    )

    # Example 1: OCR extraction
    print("\n" + "=" * 60)
    print("Example 1: OCR Extraction")
    print("=" * 60)

    result = await agent_executor.ainvoke(
        {
            "input": "Extract text from the file invoice.pdf with ref_id 'invoice_001' and ref_type 'invoice'"
        }
    )
    print(f"Result: {result['output']}")

    # Example 2: Check job status
    print("\n" + "=" * 60)
    print("Example 2: Check Job Status")
    print("=" * 60)

    result = await agent_executor.ainvoke({"input": "List all completed jobs"})
    print(f"Result: {result['output']}")

    # Example 3: System health
    print("\n" + "=" * 60)
    print("Example 3: System Health Check")
    print("=" * 60)

    result = await agent_executor.ainvoke(
        {"input": "Check the health of the Marie system and get capacity information"}
    )
    print(f"Result: {result['output']}")


if __name__ == "__main__":
    asyncio.run(main())
