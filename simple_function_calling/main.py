from dotenv import load_dotenv
load_dotenv()

from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch

tools = [TavilySearch()]

llm = ChatOpenAI(
    base_url="https://6xk1zcf93er75x-8000.proxy.runpod.net/v1",
    api_key="EMPTY",
    temperature=0.9,
    model="Qwen/Qwen3-14B-FP8"
)
llm_with_tools = llm.bind_tools(tools)

def find_tool_to_use(tools: List[BaseTool], tool_name: str):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

messages = [HumanMessage(content="What is Tom Brady's favorite type of cake?")]

def main():
    print("Hello from simple-function-calling!")
    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or [] #defensive programming
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_to_use(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
        else:
            print(ai_message.content)
            break

if __name__ == "__main__":
    main()
