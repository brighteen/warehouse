from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import (MCPToolset,
                                                   StdioServerParameters)

load_dotenv('../.env')

# --- Step 1: Import Tools from MCP Server ---
async def get_tools_async():
  print("Attempting to connect to MCP Blender server...")
  tools, exit_stack = await MCPToolset.from_server(
      connection_params=StdioServerParameters(
          command='uvx', # Command to run the server
          args=["blender-mcp"],
      )
  )
  print("MCP Toolset created successfully.")
  return tools, exit_stack

# --- Step 2: Agent Definition ---
async def get_agent_async():
  tools, exit_stack = await get_tools_async()
  print(f"Fetched {len(tools)} tools from MCP server.")
  root_agent = LlmAgent(
      # model=LiteLlm(model='anthropic/claude-3-7-sonnet-20250219'),
      model=LiteLlm(model='gemini/gemini-2.0-flash'),
      name='blender_assistant',
      instruction='Help user interact with the blender using available tools.',
      tools=tools, # Provide the MCP tools to the ADK agent
  )
  return root_agent, exit_stack

# root_agent = get_agent_async()

# 비동기 함수를 실행하기 위한 메인 함수
async def main():
    root_agent, exit_stack = await get_agent_async()
    return root_agent, exit_stack

# FastAPI와 같은 외부 모듈에서 접근할 수 있도록 함수 노출
get_root_agent_async = get_agent_async

# ADK CLI가 root_agent 변수를 찾을 수 있도록 함수 자체를 할당
# 이렇게 하면 비동기 함수를 직접 변수에 할당하는 방식으로 문제를 해결
root_agent = get_agent_async

# 스크립트가 직접 실행될 때만 실행
if __name__ == "__main__":
    import asyncio
    root_agent, exit_stack = asyncio.run(main())