import os
import chromadb
import json
from typing import Annotated, TYPE_CHECKING
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent
import asyncio
if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
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

## 获取向量知识库的检索上下文的插件
class PromptPlugin:
    def __init__(self, collection: "Collection"):
        self.collection = collection

    @kernel_function(
        name="build_augmented_prompt",
        description="Build an augmented prompt using retrieval context."
    )
    def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
        return (
            f"Retrieved Context:\n{retrieval_context}\n\n"
            f"User Query: {query}\n\n"
            "Based ONLY on the above context, please provide your answer."
        )

    @kernel_function(name="retrieve_context", description="Retrieve conetxt from the database.")
    def get_retrieval_context(self, query: str) -> str:
        results = self.collection.query(
            query_texts=[query],
            include=["documents", "metadatas"],
            n_results=2
        )

        context_entries = []
        if results and results.get('documents') and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                context_entries.append(f"Documnet: {doc}\nMetadata: {metadata}")

        return "\n\n".join(context_entries) if context_entries else "No retrieval context found." 


## 获取地区天气信息的插件
class WeatherInfoPlugin:
    def __init__(self):
        ## dict
        self.destination_temperatures = {
            "maldives": "82°F (28°C)",
            "swiss alps": "45°F (7°C)",
            "african safaris": "75°F (24°C)"
        }
    
    @kernel_function(description="Get the average temperature for a specific travel destination.")
    def get_destination_temperature(self, destination: str) -> Annotated[str, "Returns the average temperature for the destination."]:
        normalized_destination = destination.lower()

        if normalized_destination in self.destination_temperatures:
            return f"The average temperature in {destination} is {self.destination_temperatures[normalized_destination]}."
        else:
            return f"Sorry, I don't have the temperature information for {destination}. The available destinations are: Maldives, Swiss Alps, and African safaris."
        

class DestinationsPlugin:
    DESTINATIONS = {
        "maldives": {
            "name": "The Maldives",
            "description": "An archipelago of 26 atolls in the Indian Ocean, kwown for prestine beaches and overwater bungalows.",
            "best_time": "November to April (dry season)",
            "activities": ["Snorkeling", "Diving", "Island hopping", "Spa retreats", "Underwater dining"],
            "avg_cost": "$400-1200 per night for luxury resorts"
        },
        "swiss alps": {
            "name": "The Swiss Alps",
            "description": "Mountain range spanning across Switzerland with picturesque villages and world-class ski resorts.",
            "best_time": "December to March for skiing, June to September for hiking",
            "activities": ["Skiing", "Snowboarding", "Hiking", "Mountain biking", "Paragliding"],
            "avg_cost": "$250-500 per night for alpine accommodations"
        },
        "safari": {
            "name": "African Safari",
            "description": "Wildlife viewing experiences across various African countries including Kenya, Tanzania, and South Africa.",
            "best_time": "June to October (dry season) for optimal wildlife viewing",
            "activities": ["Game drives", "Walking safaris", "Hot air balloon rides", "Cultural village visits"],
            "avg_cost": "$400-800 per person per day for luxury safari packages"
        },
        "bali": {
            "name": "Bali, Indonesia",
            "description": "Island paradise known for lush rice terraces, beautiful temples, and vibrant culture.",
            "best_time": "April to October (dry season)",
            "activities": ["Surfing", "Temple visits", "Rice terrace trekking", "Yoga retreats", "Beach relaxation"],
            "avg_cost": "$100-500 per night depending on accommodation type"
        },
        "santorini": {
            "name": "Santorini, Greece",
            "description": "Stunning volcanic island with white-washed buildings and blue domes overlooking the Aegean Sea.",
            "best_time": "Late April to early November",
            "activities": ["Sunset watching in Oia", "Wine tasting", "Boat tours", "Beach hopping", "Ancient ruins exploration"],
            "avg_cost": "$200-600 per night for caldera view accommodations"
        }
    }

    @kernel_function(
        name="get_destination_info",
        description="Provides detailed information about specific travel destinations."
    )
    def get_destination_info(self, query: str) -> str:
        query_lower = query.lower()
        matching_destinations = []

        for key, details in DestinationsPlugin.DESTINATIONS.items():
            if key in query_lower or details['name'].lower() in query_lower:
                matching_destinations.append(details)
        
        if not matching_destinations:
            return (f"User Query: {query}\n\n"
                    f"I couldn't find specific destination information in our database. "
                    f"Please use the general retrieval system for this query.")

        destination_info = "\n\n".join([
            f"Destination: {dest['name']}\n"
            f"Description: {dest['description']}\n"
            f"Best time to visit: {dest['best_time']}\n"
            f"Popular activities: {", ".join(dest['activities'])}\n"
            f"Average cost: {dest['avg_cost']}" for dest in matching_destinations
        ])

        return (f"Destination information:\n{destination_info}\n\n"
                f"User Query: {query}\n\n"
                "Based on the above destination details, provide a helpful response "
                "that addresses the user's query about this location."
                )
        

## 定义向量数据库
collection = chromadb.PersistentClient(path="./chroma_db").create_collection(
    name="travel_documents",
    metadata={"description": "travel_service"},
    get_or_create=True
)

documents = [
    "Contoso Travel offers luxury vacation packages to exotic destinations worldwide.",
    "Our premium travel services include personalized itinerary planning and 24/7 concierge support.",
    "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage.",
    "Popular destinations include the Maldives, Swiss Alps, and African safaris.",
    "Contoso Travel provides exclusive access to boutique hotels and private guided tours.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": "training", "type": "explanation"} for _ in documents]
)

agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[DestinationsPlugin(), WeatherInfoPlugin(), PromptPlugin(collection)],
    name="TravelAgent",
    instructions="Answer the travel query using the provided context. If context is provided, do not say 'I have no context for the information.'"
) 

async def main():
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Can you explain Contoso's travel insurance coverage?",
        "What is the average temperature of the Maldives?",
        "What is a good cold destination offered by Contoso and what is it average temperature?",
    ]

    for user_input in user_inputs:
        agent_name = None
        full_response: list[str] = []
        function_calls: list[str] = []

        # Buffer to reconstruct streaming function call
        current_function_name = None
        argument_buffer = ""

        async for response in agent.invoke_stream(
            messages=user_input, thread=thread
        ):
            thread = response.thread
            agent_name = response.name

            content_items = response.items

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
                elif isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)
            
        ## print
        print(f"# User: {user_input}\n")
        if function_calls:
            print(f"Function calls: {function_calls}")
        print(f"# {agent_name or 'assistant'}:", "".join(full_response))
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())