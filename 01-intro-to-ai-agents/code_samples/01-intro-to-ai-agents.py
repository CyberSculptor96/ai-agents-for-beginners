## practice from 01-semantic-kernel.py
import os
from typing import Annotated
from openai import AsyncOpenAI

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

import random
import asyncio

class DestinationsPlugin:
    def __init__(self):
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia"
        ]
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(self) -> Annotated[str, "Returns a random vacation destination."]:
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)
        
        destination = random.choice(available_destinations)

        self.last_destination = destination
        return destination

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/",
)

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o",
    async_client=client
)

from semantic_kernel.agents import ChatCompletionAgent
agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions="You are a helpful AI Agent that can help plan vacations for customers at random destinations",
)

from semantic_kernel.agents import ChatHistoryAgentThread
async def main():
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "plan me a day trip."
    ]

    for user_input in user_inputs:
        print(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(
            messages=user_input, thread=thread,
        ):
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread
        print()
    
    # Clean up the thread
    await thread.delete() if thread else None

import asyncio
if __name__ == "__main__":
    asyncio.run(main())