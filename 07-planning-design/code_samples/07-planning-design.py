## practice from 07-semantic-kernel.py
import os
import json
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import KernelArguments
from pydantic import BaseModel, Field, ValidationError
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client
)

### 核心代码
class SubTask(BaseModel):
    assigned_agent: str = Field(
        description="The specific agent assigned to handle this subtask"
    )
    task_details: str = Field(
        description="Detailed description of what needs to be done for this subtask"
    )

class TravelPlan(BaseModel):
    main_task: str = Field(
        description="The overall travel request from the user"
    )
    subtasks: list[SubTask] = Field(
        description="List of subtasks broken down from the main task, each assigned to a specialized agent"
    )

AGENT_NAME = "TravelAgent"
AGENT_INSTRUCTIONS = """You are a planner agent.
    Your job is to decide which agents to run based on the user's request.
    Below are the available agents specialised in different tasks:
    - FlightBooking: For booking flights and providing flight information
    - HotelBooking: For booking hotels and providing hotel information
    - CarRental: For booking cars and providing car rental information
    - ActivitiesBooking: For booking activities and providing activity information
    - DestinationInfo: For providing information about destinations
    - DefaultAgent: For handling general requests"""

settings = OpenAIChatPromptExecutionSettings(response_format=TravelPlan)

agent = ChatCompletionAgent(
    service=chat_completion_service,
    name=AGENT_NAME,
    instructions=AGENT_INSTRUCTIONS,
    arguments=KernelArguments(settings)
)

async def main():
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Create a travel plan for a family of 4, with 2 kids, from Singapore to Melbourne"
    ]

    for user_input in user_inputs:
        output_lines = []
        output_lines.append(f"# User: {user_input}\n")

        response = await agent.get_response(messages=user_input, thread=thread)
        thread = response.thread

        try:
            travel_plan = TravelPlan.model_validate(json.loads(response.message.content))
            formatted_json = travel_plan.model_dump_json(indent=4)
            output_lines.append("Validated Travel Plan:")
            output_lines.append(formatted_json)
        except ValidationError as e:
            output_lines.append("Validation Error:")
            output_lines.append(str(e))
            output_lines.append("\nRaw Response:")
            output_lines.append(response.content)

        output_lines.append("=" * 80)
        print("\n".join(output_lines))

if __name__ == "__main__":
    asyncio.run(main())