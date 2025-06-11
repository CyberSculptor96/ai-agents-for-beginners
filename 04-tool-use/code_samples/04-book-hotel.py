## 新版本：使用 Semantic Kernel + OpenAI 官方 API（GitHub Models）重写

from dotenv import load_dotenv
import os
import requests
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import asyncio
import random

# 载入 .env 环境变量（含 OpenAI Key）
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from datetime import datetime, timedelta


### 鲁棒性处理：如果用户输入的自然语言请求格式和调用的Tool要求的输入格式不符合，需要进行转换。
def validate_future_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return date_str  # fallback for non-ISO formats
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if dt < today:
        dt = today + timedelta(days=30)  # 推迟到 30 天后
    return dt.strftime("%Y-%m-%d")


def extract_airport_code(location: str) -> str:
    """提取字符串末尾的三字母机场代码"""
    parts = location.strip().split()
    for part in reversed(parts):
        if len(part) == 3 and part.isalpha():
            return part.upper()
    return location

# Define Booking Plugin
class BookingPlugin:
    """Booking Plugin for customers"""

    @kernel_function(description="booking hotel")
    def booking_hotel(
        self,
        query: Annotated[str, "The name of the city"],
        check_in_date: Annotated[str, "Hotel Check-in Time"],
        check_out_date: Annotated[str, "Hotel Check-out Time"],
    ) -> Annotated[str, "Return the result of booking hotel information"]:
        params = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": validate_future_date(check_in_date),
            "check_out_date": validate_future_date(check_out_date),
            "adults": "1",
            "currency": "GBP",
            "gl": "uk",
            "hl": "en",
            "api_key": os.environ.get("SERPAPI_SEARCH_API_KEY")
        }
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            result = str(response.json().get("properties", "No properties found"))
            return result[:4000]
        else:
            print("Error retrieving hotel information:", response.status_code, response.text)

        print(">>> 🎯 FUNCTION CALLED")
        return "Failed to fetch hotel info"

    @kernel_function(description="booking flight")
    def booking_flight(
        self,
        origin: Annotated[str, "The name of Departure"],
        destination: Annotated[str, "The name of Destination"],
        outbound_date: Annotated[str, "The date of outbound"],
        return_date: Annotated[str, "The date of Return_date"],
    ) -> Annotated[str, "Return the result of booking flight information"]:
        params = {
            "engine": "google_flights",
            "departure_id": extract_airport_code(origin),
            "arrival_id": extract_airport_code(destination),
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "GBP",
            "hl": "en",
            "api_key": os.environ.get("SERPAPI_SEARCH_API_KEY")
        }
        go_response = requests.get("https://serpapi.com/search", params=params)
        if go_response.status_code == 200:
            result = str(go_response.json())
            return result[:4000]
        else:
            print("Error retrieving flight information:", go_response.status_code, go_response.text)

        print(">>> 🎯 FUNCTION CALLED")
        return "Failed to fetch flight info"

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
async def main():
    # ✅ 替代 Azure Agent：使用 OpenAI 官方 + Semantic Kernel Chat Agent
    client = AsyncOpenAI(
        api_key=os.environ.get("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com/"
    )

    chat_service = OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",  # 或 "gpt-3.5-turbo"
        async_client=client,
    )

    agent = ChatCompletionAgent(
        service=chat_service,
        plugins=[BookingPlugin()],
        name="BookingAgent",
        instructions="""
        You are a booking agent. Help users book flights or hotels.
        
        - Use plugin functions to look up info.
        - Return result in markdown tables.
        """
    )

    thread: ChatHistoryAgentThread | None = None
    # user_inputs = [
    #     "Help me book flight tickets and hotel for the following trip: London Heathrow LHR Feb 20th 2025 to New York JFK returning Feb 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York. Please provide costs."
    # ]
    user_inputs = [
        "Help me book flight tickets and hotel for the following trip: LHR June 20th 2025 to JFK returning June 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York. Please provide costs."
    ]

    for user_input in user_inputs:
        print(f"\n# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(messages=user_input, thread=thread):
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread

    await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main())
