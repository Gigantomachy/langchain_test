from typing import List
from pydantic import BaseModel, Field

# class Source(BaseModel):
#     """Schema for a source used by the agent"""
    
#     url: str = Field(description="The URL of the source")

# class AgentResponse(BaseModel):
#     """Schema for agent response with answer and sources"""

#     answer: str = Field(description="The agent's response to the query")
#     sources: List[Source] = Field(
#         default_factory=list, description="List of sources used to generate the answer"
#     )

class WeatherResponse(BaseModel):
    answer: str = Field(description="A friendly, conversational answer to the user's question")
    location: str = Field(description="The city name that was queried")
    temperature_c: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="Brief weather condition like 'sunny', 'rainy', 'snowy'")