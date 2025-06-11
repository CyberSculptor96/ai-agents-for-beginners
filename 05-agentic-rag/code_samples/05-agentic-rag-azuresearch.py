## practice from 05-semantic-kernel-azuresearch.py
## Topic: agent with RAG
import os
from typing import Annotated
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import StreamingTextContent
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

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchFieldDataType
from azure.core.credentials import AzureKeyCredential

class SearchPlugin:
    def __init__(self, search_clinet: SearchClient):
        self.search_client = search_clinet
    
    @kernel_function(
        name="build_augmented_prompt",
        description="Build an augmented prompt using retrieval context or function results.",
    )
    def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
        return (
            f"Retrieved Context:\n{retrieval_context}\n\n"
            f"User Query: {query}\n\n"
            "First review the retrieved context, if this does not answer the query, try calling an available plugin functions that might give you an answer. If no context is avaible, say no."
        )

    @kernel_function(
        name="retrieve_documents",
        description="Retrieve documents from the Azure Search service."
    )
    def get_retrieval_context(self, query: str) -> str:
        results = self.search_client.search(query)
        context_strings = []
        for result in results:
            context_strings.append(f"Documents: {result['content']}")
        return "\n\n".join(context_strings) if context_strings else "No results found"

    
class WeatherInfoPlugin:
    def __init__(self):
        ## dict
        self.destination_temperatures = {
            "maldives": "82°F (28°C)",
            "swiss alps": "45°F (7°C)",
            "african safaris": "75°F (24°C)",
        }
    
    @kernel_function(description="Get the average temperature for a specific travel destination.")
    def get_destination_temperatures(self, destination: str) -> Annotated[str, "Returns the average temperature for the destination."]:
        normalized_destination = destination.lower()

        if normalized_destination in self.destination_temperatures:
            return f"The average temperature in {destination} is {self.destination_temperatures[normalized_destination]}"
        else:
            return f"Sorry, I don't have temperature information for {destination}. Available destinations are: Maldives, Swiss Alps, African Safaris."


search_service_endpoint = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
search_api_key = os.environ.get("AZURE_SEARCH_API_KEY")
index_name = "travel-documents"


## 职责分离，两个客户端分别负责内容查询和建表
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=AzureKeyCredential(search_api_key),
)

## 数据库基础知识：设置不同的字段（列名），指定数据类型，添加主键约束
## index对应的是数据库中的表，fields对应的是表中的列
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String)
]

index = SearchIndex(name=index_name, fields=fields)

## 建表
try:
    existing_index = index_client.get_index(index_name)
    print(f"Index '{index_name}' already exists, using the existing index.")
except Exception as e:
    print(f"Creating new index '{index_name}'...")
    index_client.create_index(index)

## 数据库的内容：文档（RAG）
documents = [
    {"id": "1", "content": "Contoso Travel offers luxury vacation packages to exotic destinations worldwide."},
    {"id": "2", "content": "Our premium travel services include personalized itinerary planning and 24/7 concierge support."},
    {"id": "3", "content": "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage."},
    {"id": "4", "content": "Popular destinations include the Maldives, Swiss Alps, and African safaris."},
    {"id": "5", "content": "Contoso Travel provides exclusive access to boutique hotels and private guided tours."}
]

# 将文档内容添加到索引（表）
search_client.upload_documents(documents)

agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[SearchPlugin(search_clinet=search_client), WeatherInfoPlugin()],
    name="TravelAgent",
    instructions="Answer travel queries using the provided tools and context. If context is provided, do noy say 'I have no context for that.'"
)

from semantic_kernel.agents import ChatHistoryAgentThread

async def main():
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Can you explain Contoso's travel insurance coverage?",
        "What is the average temperature of the Maldives?",
        "What is a good cold destination offered by Contoso and what is it average temperature?"
    ]

    for user_input in user_inputs:
        agent_name = None
        full_response: list[str] = []

        async for response in agent.invoke_stream(
            messages=user_input, thread=thread
        ):
            thread = response.thread
            agent_name = response.name
            
            content_items = list(response.items)
            for item in content_items:
                if isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)
        
        print(f"# User: {user_input}\n")
        print(f"# {agent_name or 'assistant'}: ", "".join(full_response))
        print("=" * 80)



if __name__ == "__main__":
    asyncio.run(main())