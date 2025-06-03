import asyncio
import json
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import httpx
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class MCPMistralClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        self.mistral_base_url = "https://api.mistral.ai/v1"
        self.http_client = httpx.AsyncClient()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def convert_mcp_tools_to_mistral_format(self, mcp_tools) -> List[Dict[str, Any]]:
        """Convert MCP tools to Mistral function calling format"""
        mistral_tools = []
        
        for tool in mcp_tools:
            mistral_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            mistral_tools.append(mistral_tool)
        
        return mistral_tools

    async def call_mistral_api(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Call Mistral API with messages and optional tools"""
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-large-latest",  # You can change this to other models like "open-mistral-7b"
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = await self.http_client.post(
            f"{self.mistral_base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
        
        return response.json()

    async def process_query(self, query: str) -> str:
        """Process a query using Mistral and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Get available tools from MCP server
        response = await self.session.list_tools()
        available_tools = self.convert_mcp_tools_to_mistral_format(response.tools)

        # Initial Mistral API call
        mistral_response = await self.call_mistral_api(messages, available_tools)
        
        # Process response and handle tool calls
        final_text = []
        
        choice = mistral_response["choices"][0]
        message = choice["message"]
        
        # Add assistant message to conversation history
        messages.append(message)
        
        if message.get("content"):
            final_text.append(message["content"])
        
        # Handle tool calls if present
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                final_text.append(f"[Calling tool {function_name} with args {function_args}]")
                
                try:
                    # Execute tool call via MCP
                    result = await self.session.call_tool(function_name, function_args)
                    
                    # Add tool result to messages
                    tool_result_message = {
                        "role": "tool",
                        "name": function_name,
                        "content": str(result.content),
                        "tool_call_id": tool_call["id"]
                    }
                    messages.append(tool_result_message)
                    
                except Exception as e:
                    # Handle tool execution errors
                    error_message = {
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error executing tool: {str(e)}",
                        "tool_call_id": tool_call["id"]
                    }
                    messages.append(error_message)
            
            # Get final response from Mistral after tool calls
            final_response = await self.call_mistral_api(messages)
            final_choice = final_response["choices"][0]
            
            if final_choice["message"].get("content"):
                final_text.append(final_choice["message"]["content"])

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Mistral Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.http_client.aclose()
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mistral_client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPMistralClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())