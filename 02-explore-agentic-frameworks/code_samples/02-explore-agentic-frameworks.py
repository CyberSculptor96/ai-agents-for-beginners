## practice from 02-semantic-kernel.py
import os
import json
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel.contents import StreamingTextContent, FunctionCallContent, FunctionResultContent 
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
        print(">>> 🎯 FUNCTION CALLED")
        return destination

load_dotenv()

from openai import AsyncOpenAI
client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)

from semantic_kernel.agents import ChatCompletionAgent
agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions="You are a helpful AI agent that can help plan vacations for customers at random destinations"
)
print("行为对象:", agent.function_choice_behavior)                # ➜ FunctionChoiceBehavior(...)
print("类型标签:", agent.function_choice_behavior.type_)         # ➜ FunctionChoiceType.AUTO
print("是否启用插件:", agent.function_choice_behavior.enable_kernel_functions)

user_inputs = [
    "Plan me a day trip.",
    "Thx!"
    # "I don't like that destination. Plan me another vacation."
]

# user_inputs = [
#     "Call the function that gives me a random vacation destination.",
#     "Use the plugin to get another destination.",
# ]

# user_inputs = [
#     # 任何一次对话里只要提到“随机目的地”，模型就会尝试调用函数
#     "Give me a random vacation spot. Feel free to call any function you need.",
#     "Thanks. I don't like that one. Please pick another destination using your function.",
# ]

async def spy(msg):                         # ← ① 监听代理内部消息
    for itm in msg.items or []:
        if isinstance(itm, FunctionCallContent):
            print(f"[CALL]  {itm.function_name} {itm.arguments}")
        elif isinstance(itm, FunctionResultContent):
            print(f"[RESULT] {itm.result}")
    pass

from semantic_kernel.agents import ChatHistoryAgentThread
async def main():
    thread: ChatHistoryAgentThread | None = None

    for user_input in user_inputs:
        agent_name = None
        full_response: list[str] = []
        function_calls: list[str] = []
        
        # Buffer to reconstruct streaming function call
        current_function_name = None
        argument_buffer = ""

        async for response in agent.invoke_stream(
            messages=user_input, thread=thread,
            on_intermediate_message=spy,
        ):
            thread = response.thread
            agent_name = response.name
            content_items = list(response.items)
            
            for item in content_items:
                if isinstance(item, FunctionCallContent):
                    if item.function_name:
                        current_function_name = item.function_name
                    
                    if isinstance(item.arguments, str):
                        argument_buffer += item.arguments
                elif isinstance(item, FunctionResultContent):
                    if current_function_name:
                        formatted_args = argument_buffer.strip()
                        try:
                            parsed_args = json.loads(formatted_args)
                            formatted_args = json.dumps(parsed_args)
                        except Exception:
                            pass

                        function_calls.append(f"Calling function: {current_function_name}({formatted_args})")
                        current_function_name = None
                        argument_buffer = ""
                    
                    function_calls.append(f"\nFunction Result:\n\n{item.result}")
                if isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)
        
        print(f"# User: {user_input}\n")
        if function_calls:
            print("\nFunction Calls:\n", "\n".join(function_calls))
        print(f"# {agent_name or 'Assistant'}:", "".join(full_response))
        print("=" * 80)



if __name__ == "__main__":
    asyncio.run(main())
