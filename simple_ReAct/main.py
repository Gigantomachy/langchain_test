from dotenv import load_dotenv
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_core.output_parsers.pydantic import PydanticOutputParser # no need if we are using structured output instead
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
# from schemas import AgentResponse
from schemas import WeatherResponse

tools = [TavilySearch()]
#llm = ChatOpenAI("gpt-5")
llm = ChatOpenAI(
    base_url="https://5qzd4s24beha3n-8000.proxy.runpod.net/v1",
    api_key="EMPTY",
    model="Qwen/Qwen3-4B-Instruct-2507",
    temperature=0.7,
    max_tokens=2048,
)
# structured_llm = llm.with_structured_output(WeatherResponse) -> doesn't work with self hosted Qwen, probably works with ChatGPT

# Note: for what we are trying to do here, it seems that non-thinking models like Qwen3-4B-Instruct-2507 are actually better 
# the ReAct prompt seems to confuse the </think>

# The disadvantage of structured_llm is that it seems less reliable with non-Open AI models. Qwen3 self hosted seemed to
# fail initially but work on subsequent runs.

# However, structured output has an advantage in that we are not sending the format instructions over and over again.
# With Pydantic, we are sending the format instructions even at the beginning when the LLM is deciding which tool to use.	
# With structured_llm, we only activate the structured_llm at the end to parse the final answer.

# react_prompt = hub.pull("hwchase17/react") # https://smith.langchain.com/hub/hwchase17/react
output_parser = PydanticOutputParser(pydantic_object=WeatherResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
#).partial(format_instructions="") -> no need for format_instructions if we use llm.with_structured_output(WeatherResponse)
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm,
    tools,
    #react_prompt
    react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(agent=agent, 
                               tools=tools, 
                               verbose=True,
                               handle_parsing_errors=True,
                               max_iterations=10,
                               return_intermediate_steps=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

# chain = agent_executor | extract_output | structured_llm
chain = agent_executor | extract_output | parse_output

def main():
    print("Hello from section-4!")
    result = chain.invoke(
        input={
            "input": "Search for information about the weather in Ottawa"
        }
    )
    print(result)

if __name__ == "__main__":
    main()
