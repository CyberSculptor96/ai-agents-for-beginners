{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Semantic Kernel \n",
    "\n",
    "In this code sample, you will use the [Semantic Kernel](https://aka.ms/ai-agents-beginners/semantic-kernel) AI Framework to create a basic agent. \n",
    "\n",
    "The goal of this sample is to show you the steps that we will later use in the additional code samples when implementing the different agentic patterns. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import the Needed Python Packages "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os \n",
    "from typing import Annotated\n",
    "from openai import AsyncOpenAI\n",
    "\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread\n",
    "from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion\n",
    "from semantic_kernel.functions import kernel_function"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating the Client\n",
    "\n",
    "In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. \n",
    "\n",
    "The `ai_model_id` is defined as `gpt-4o-mini`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. \n",
    "\n",
    "For us to use the `Azure Inference SDK` that is used for the `base_url` for GitHub Models, we will use the `OpenAIChatCompletion` connector within Semantic Kernel. There are also other [available connectors](https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion) to use Semantic Kernel for other model providers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random   \n",
    "\n",
    "# Define a sample plugin for the sample\n",
    "\n",
    "class DestinationsPlugin:\n",
    "    \"\"\"A List of Random Destinations for a vacation.\"\"\"\n",
    "\n",
    "    def __init__(self):\n",
    "        # List of vacation destinations\n",
    "        self.destinations = [\n",
    "            \"Barcelona, Spain\",\n",
    "            \"Paris, France\",\n",
    "            \"Berlin, Germany\",\n",
    "            \"Tokyo, Japan\",\n",
    "            \"Sydney, Australia\",\n",
    "            \"New York, USA\",\n",
    "            \"Cairo, Egypt\",\n",
    "            \"Cape Town, South Africa\",\n",
    "            \"Rio de Janeiro, Brazil\",\n",
    "            \"Bali, Indonesia\"\n",
    "        ]\n",
    "        # Track last destination to avoid repeats\n",
    "        self.last_destination = None\n",
    "\n",
    "    @kernel_function(description=\"Provides a random vacation destination.\")\n",
    "    def get_random_destination(self) -> Annotated[str, \"Returns a random vacation destination.\"]:\n",
    "        # Get available destinations (excluding last one if possible)\n",
    "        available_destinations = self.destinations.copy()\n",
    "        if self.last_destination and len(available_destinations) > 1:\n",
    "            available_destinations.remove(self.last_destination)\n",
    "\n",
    "        # Select a random destination\n",
    "        destination = random.choice(available_destinations)\n",
    "\n",
    "        # Update the last destination\n",
    "        self.last_destination = destination\n",
    "\n",
    "        return destination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "load_dotenv()\n",
    "client = AsyncOpenAI(\n",
    "    api_key=os.environ.get(\"GITHUB_TOKEN\"), \n",
    "    base_url=\"https://models.inference.ai.azure.com/\",\n",
    ")\n",
    "\n",
    "# Create an AI Service that will be used by the `ChatCompletionAgent`\n",
    "chat_completion_service = OpenAIChatCompletion(\n",
    "    ai_model_id=\"gpt-4o-mini\",\n",
    "    async_client=client,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating the Agent \n",
    "\n",
    "Below we create the Agent called `TravelAgent`.\n",
    "\n",
    "For this example, we are using very simple instructions. You can change these instructions to see how the agent responds differently. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "agent = ChatCompletionAgent(\n",
    "    service=chat_completion_service, \n",
    "    plugins=[DestinationsPlugin()],\n",
    "    name=\"TravelAgent\",\n",
    "    instructions=\"You are a helpful AI Agent that can help plan vacations for customers at random destinations\",\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Running the Agent\n",
    "\n",
    "Now we can run the Agent by defining a thread of type `ChatHistoryAgentThread`.  Any required system messages are provided to the agent's invoke_stream `messages` keyword argument.\n",
    "\n",
    "After these are defined, we create a `user_inputs` that will be what the user is sending to the agent. In this case, we have set this message to `Plan me a sunny vacation`. \n",
    "\n",
    "Feel free to change this message to see how the agent responds differently. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "# User: Plan me a day trip.\n",
      "\n"
     ]
    },
    {
     "ename": "ServiceResponseException",
     "evalue": "(\"<class 'semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.OpenAIChatCompletion'> service failed to complete the prompt\", AuthenticationError(\"Error code: 401 - {'error': {'code': 'unauthorized', 'message': 'The `models` permission is required to access this endpoint', 'details': 'The `models` permission is required to access this endpoint'}}\"))",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mAuthenticationError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/connectors/ai/open_ai/services/open_ai_handler.py:87\u001b[39m, in \u001b[36mOpenAIHandler._send_completion_request\u001b[39m\u001b[34m(self, settings)\u001b[39m\n\u001b[32m     86\u001b[39m         settings_dict.pop(\u001b[33m\"\u001b[39m\u001b[33mparallel_tool_calls\u001b[39m\u001b[33m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m)\n\u001b[32m---> \u001b[39m\u001b[32m87\u001b[39m     response = \u001b[38;5;28;01mawait\u001b[39;00m \u001b[38;5;28mself\u001b[39m.client.chat.completions.create(**settings_dict)\n\u001b[32m     88\u001b[39m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py:2028\u001b[39m, in \u001b[36mAsyncCompletions.create\u001b[39m\u001b[34m(self, messages, model, audio, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, modalities, n, parallel_tool_calls, prediction, presence_penalty, reasoning_effort, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, web_search_options, extra_headers, extra_query, extra_body, timeout)\u001b[39m\n\u001b[32m   2027\u001b[39m validate_response_format(response_format)\n\u001b[32m-> \u001b[39m\u001b[32m2028\u001b[39m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mawait\u001b[39;00m \u001b[38;5;28mself\u001b[39m._post(\n\u001b[32m   2029\u001b[39m     \u001b[33m\"\u001b[39m\u001b[33m/chat/completions\u001b[39m\u001b[33m\"\u001b[39m,\n\u001b[32m   2030\u001b[39m     body=\u001b[38;5;28;01mawait\u001b[39;00m async_maybe_transform(\n\u001b[32m   2031\u001b[39m         {\n\u001b[32m   2032\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mmessages\u001b[39m\u001b[33m\"\u001b[39m: messages,\n\u001b[32m   2033\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mmodel\u001b[39m\u001b[33m\"\u001b[39m: model,\n\u001b[32m   2034\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33maudio\u001b[39m\u001b[33m\"\u001b[39m: audio,\n\u001b[32m   2035\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mfrequency_penalty\u001b[39m\u001b[33m\"\u001b[39m: frequency_penalty,\n\u001b[32m   2036\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mfunction_call\u001b[39m\u001b[33m\"\u001b[39m: function_call,\n\u001b[32m   2037\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mfunctions\u001b[39m\u001b[33m\"\u001b[39m: functions,\n\u001b[32m   2038\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mlogit_bias\u001b[39m\u001b[33m\"\u001b[39m: logit_bias,\n\u001b[32m   2039\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mlogprobs\u001b[39m\u001b[33m\"\u001b[39m: logprobs,\n\u001b[32m   2040\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mmax_completion_tokens\u001b[39m\u001b[33m\"\u001b[39m: max_completion_tokens,\n\u001b[32m   2041\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mmax_tokens\u001b[39m\u001b[33m\"\u001b[39m: max_tokens,\n\u001b[32m   2042\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mmetadata\u001b[39m\u001b[33m\"\u001b[39m: metadata,\n\u001b[32m   2043\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mmodalities\u001b[39m\u001b[33m\"\u001b[39m: modalities,\n\u001b[32m   2044\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mn\u001b[39m\u001b[33m\"\u001b[39m: n,\n\u001b[32m   2045\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mparallel_tool_calls\u001b[39m\u001b[33m\"\u001b[39m: parallel_tool_calls,\n\u001b[32m   2046\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mprediction\u001b[39m\u001b[33m\"\u001b[39m: prediction,\n\u001b[32m   2047\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mpresence_penalty\u001b[39m\u001b[33m\"\u001b[39m: presence_penalty,\n\u001b[32m   2048\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mreasoning_effort\u001b[39m\u001b[33m\"\u001b[39m: reasoning_effort,\n\u001b[32m   2049\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mresponse_format\u001b[39m\u001b[33m\"\u001b[39m: response_format,\n\u001b[32m   2050\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mseed\u001b[39m\u001b[33m\"\u001b[39m: seed,\n\u001b[32m   2051\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mservice_tier\u001b[39m\u001b[33m\"\u001b[39m: service_tier,\n\u001b[32m   2052\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mstop\u001b[39m\u001b[33m\"\u001b[39m: stop,\n\u001b[32m   2053\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mstore\u001b[39m\u001b[33m\"\u001b[39m: store,\n\u001b[32m   2054\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mstream\u001b[39m\u001b[33m\"\u001b[39m: stream,\n\u001b[32m   2055\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mstream_options\u001b[39m\u001b[33m\"\u001b[39m: stream_options,\n\u001b[32m   2056\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mtemperature\u001b[39m\u001b[33m\"\u001b[39m: temperature,\n\u001b[32m   2057\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mtool_choice\u001b[39m\u001b[33m\"\u001b[39m: tool_choice,\n\u001b[32m   2058\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mtools\u001b[39m\u001b[33m\"\u001b[39m: tools,\n\u001b[32m   2059\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mtop_logprobs\u001b[39m\u001b[33m\"\u001b[39m: top_logprobs,\n\u001b[32m   2060\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mtop_p\u001b[39m\u001b[33m\"\u001b[39m: top_p,\n\u001b[32m   2061\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33muser\u001b[39m\u001b[33m\"\u001b[39m: user,\n\u001b[32m   2062\u001b[39m             \u001b[33m\"\u001b[39m\u001b[33mweb_search_options\u001b[39m\u001b[33m\"\u001b[39m: web_search_options,\n\u001b[32m   2063\u001b[39m         },\n\u001b[32m   2064\u001b[39m         completion_create_params.CompletionCreateParamsStreaming\n\u001b[32m   2065\u001b[39m         \u001b[38;5;28;01mif\u001b[39;00m stream\n\u001b[32m   2066\u001b[39m         \u001b[38;5;28;01melse\u001b[39;00m completion_create_params.CompletionCreateParamsNonStreaming,\n\u001b[32m   2067\u001b[39m     ),\n\u001b[32m   2068\u001b[39m     options=make_request_options(\n\u001b[32m   2069\u001b[39m         extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout\n\u001b[32m   2070\u001b[39m     ),\n\u001b[32m   2071\u001b[39m     cast_to=ChatCompletion,\n\u001b[32m   2072\u001b[39m     stream=stream \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28;01mFalse\u001b[39;00m,\n\u001b[32m   2073\u001b[39m     stream_cls=AsyncStream[ChatCompletionChunk],\n\u001b[32m   2074\u001b[39m )\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/openai/_base_client.py:1748\u001b[39m, in \u001b[36mAsyncAPIClient.post\u001b[39m\u001b[34m(self, path, cast_to, body, files, options, stream, stream_cls)\u001b[39m\n\u001b[32m   1745\u001b[39m opts = FinalRequestOptions.construct(\n\u001b[32m   1746\u001b[39m     method=\u001b[33m\"\u001b[39m\u001b[33mpost\u001b[39m\u001b[33m\"\u001b[39m, url=path, json_data=body, files=\u001b[38;5;28;01mawait\u001b[39;00m async_to_httpx_files(files), **options\n\u001b[32m   1747\u001b[39m )\n\u001b[32m-> \u001b[39m\u001b[32m1748\u001b[39m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mawait\u001b[39;00m \u001b[38;5;28mself\u001b[39m.request(cast_to, opts, stream=stream, stream_cls=stream_cls)\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/openai/_base_client.py:1555\u001b[39m, in \u001b[36mAsyncAPIClient.request\u001b[39m\u001b[34m(self, cast_to, options, stream, stream_cls)\u001b[39m\n\u001b[32m   1554\u001b[39m     log.debug(\u001b[33m\"\u001b[39m\u001b[33mRe-raising status error\u001b[39m\u001b[33m\"\u001b[39m)\n\u001b[32m-> \u001b[39m\u001b[32m1555\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;28mself\u001b[39m._make_status_error_from_response(err.response) \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[32m   1557\u001b[39m \u001b[38;5;28;01mbreak\u001b[39;00m\n",
      "\u001b[31mAuthenticationError\u001b[39m: Error code: 401 - {'error': {'code': 'unauthorized', 'message': 'The `models` permission is required to access this endpoint', 'details': 'The `models` permission is required to access this endpoint'}}",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[31mServiceResponseException\u001b[39m                  Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[6]\u001b[39m\u001b[32m, line 28\u001b[39m\n\u001b[32m     25\u001b[39m     \u001b[38;5;66;03m# Clean up the thread\u001b[39;00m\n\u001b[32m     26\u001b[39m     \u001b[38;5;28;01mawait\u001b[39;00m thread.delete() \u001b[38;5;28;01mif\u001b[39;00m thread \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[32m---> \u001b[39m\u001b[32m28\u001b[39m \u001b[38;5;28;01mawait\u001b[39;00m main()\n",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[6]\u001b[39m\u001b[32m, line 14\u001b[39m, in \u001b[36mmain\u001b[39m\u001b[34m()\u001b[39m\n\u001b[32m     12\u001b[39m \u001b[38;5;28mprint\u001b[39m(\u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33m# User: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00muser_input\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[33m\"\u001b[39m)\n\u001b[32m     13\u001b[39m first_chunk = \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[32m---> \u001b[39m\u001b[32m14\u001b[39m \u001b[38;5;28;01masync\u001b[39;00m \u001b[38;5;28;01mfor\u001b[39;00m response \u001b[38;5;129;01min\u001b[39;00m agent.invoke_stream(\n\u001b[32m     15\u001b[39m     messages=user_input, thread=thread,\n\u001b[32m     16\u001b[39m ):\n\u001b[32m     17\u001b[39m     \u001b[38;5;66;03m# 5. Print the response\u001b[39;00m\n\u001b[32m     18\u001b[39m     \u001b[38;5;28;01mif\u001b[39;00m first_chunk:\n\u001b[32m     19\u001b[39m         \u001b[38;5;28mprint\u001b[39m(\u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33m# \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mresponse.name\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m: \u001b[39m\u001b[33m\"\u001b[39m, end=\u001b[33m\"\u001b[39m\u001b[33m\"\u001b[39m, flush=\u001b[38;5;28;01mTrue\u001b[39;00m)\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/utils/telemetry/agent_diagnostics/decorators.py:39\u001b[39m, in \u001b[36mtrace_agent_invocation.<locals>.wrapper_decorator\u001b[39m\u001b[34m(*args, **kwargs)\u001b[39m\n\u001b[32m     36\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m agent.description:\n\u001b[32m     37\u001b[39m     span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)\n\u001b[32m---> \u001b[39m\u001b[32m39\u001b[39m \u001b[38;5;28;01masync\u001b[39;00m \u001b[38;5;28;01mfor\u001b[39;00m response \u001b[38;5;129;01min\u001b[39;00m invoke_func(*args, **kwargs):\n\u001b[32m     40\u001b[39m     \u001b[38;5;28;01myield\u001b[39;00m response\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/agents/chat_completion/chat_completion_agent.py:447\u001b[39m, in \u001b[36mChatCompletionAgent.invoke_stream\u001b[39m\u001b[34m(self, messages, thread, on_intermediate_message, arguments, kernel, **kwargs)\u001b[39m\n\u001b[32m    444\u001b[39m response_builder: \u001b[38;5;28mlist\u001b[39m[\u001b[38;5;28mstr\u001b[39m] = []\n\u001b[32m    445\u001b[39m start_idx = \u001b[38;5;28mlen\u001b[39m(agent_chat_history)\n\u001b[32m--> \u001b[39m\u001b[32m447\u001b[39m \u001b[38;5;28;01masync\u001b[39;00m \u001b[38;5;28;01mfor\u001b[39;00m response_list \u001b[38;5;129;01min\u001b[39;00m responses:\n\u001b[32m    448\u001b[39m     \u001b[38;5;28;01mfor\u001b[39;00m response \u001b[38;5;129;01min\u001b[39;00m response_list:\n\u001b[32m    449\u001b[39m         role = response.role\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/connectors/ai/chat_completion_client_base.py:261\u001b[39m, in \u001b[36mChatCompletionClientBase.get_streaming_chat_message_contents\u001b[39m\u001b[34m(self, chat_history, settings, **kwargs)\u001b[39m\n\u001b[32m    259\u001b[39m all_messages: \u001b[38;5;28mlist\u001b[39m[\u001b[33m\"\u001b[39m\u001b[33mStreamingChatMessageContent\u001b[39m\u001b[33m\"\u001b[39m] = []\n\u001b[32m    260\u001b[39m function_call_returned = \u001b[38;5;28;01mFalse\u001b[39;00m\n\u001b[32m--> \u001b[39m\u001b[32m261\u001b[39m \u001b[38;5;28;01masync\u001b[39;00m \u001b[38;5;28;01mfor\u001b[39;00m messages \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m._inner_get_streaming_chat_message_contents(\n\u001b[32m    262\u001b[39m     chat_history, settings, request_index\n\u001b[32m    263\u001b[39m ):\n\u001b[32m    264\u001b[39m     \u001b[38;5;28;01mfor\u001b[39;00m msg \u001b[38;5;129;01min\u001b[39;00m messages:\n\u001b[32m    265\u001b[39m         \u001b[38;5;28;01mif\u001b[39;00m msg \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/utils/telemetry/model_diagnostics/decorators.py:165\u001b[39m, in \u001b[36mtrace_streaming_chat_completion.<locals>.inner_trace_streaming_chat_completion.<locals>.wrapper_decorator\u001b[39m\u001b[34m(*args, **kwargs)\u001b[39m\n\u001b[32m    159\u001b[39m \u001b[38;5;129m@functools\u001b[39m.wraps(completion_func)\n\u001b[32m    160\u001b[39m \u001b[38;5;28;01masync\u001b[39;00m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34mwrapper_decorator\u001b[39m(\n\u001b[32m    161\u001b[39m     *args: Any, **kwargs: Any\n\u001b[32m    162\u001b[39m ) -> AsyncGenerator[\u001b[38;5;28mlist\u001b[39m[\u001b[33m\"\u001b[39m\u001b[33mStreamingChatMessageContent\u001b[39m\u001b[33m\"\u001b[39m], Any]:\n\u001b[32m    163\u001b[39m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m are_model_diagnostics_enabled():\n\u001b[32m    164\u001b[39m         \u001b[38;5;66;03m# If model diagnostics are not enabled, just return the completion\u001b[39;00m\n\u001b[32m--> \u001b[39m\u001b[32m165\u001b[39m         \u001b[38;5;28;01masync\u001b[39;00m \u001b[38;5;28;01mfor\u001b[39;00m streaming_chat_message_contents \u001b[38;5;129;01min\u001b[39;00m completion_func(*args, **kwargs):\n\u001b[32m    166\u001b[39m             \u001b[38;5;28;01myield\u001b[39;00m streaming_chat_message_contents\n\u001b[32m    167\u001b[39m         \u001b[38;5;28;01mreturn\u001b[39;00m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/connectors/ai/open_ai/services/open_ai_chat_completion_base.py:110\u001b[39m, in \u001b[36mOpenAIChatCompletionBase._inner_get_streaming_chat_message_contents\u001b[39m\u001b[34m(self, chat_history, settings, function_invoke_attempt)\u001b[39m\n\u001b[32m    107\u001b[39m settings.messages = \u001b[38;5;28mself\u001b[39m._prepare_chat_history_for_request(chat_history)\n\u001b[32m    108\u001b[39m settings.ai_model_id = settings.ai_model_id \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m.ai_model_id\n\u001b[32m--> \u001b[39m\u001b[32m110\u001b[39m response = \u001b[38;5;28;01mawait\u001b[39;00m \u001b[38;5;28mself\u001b[39m._send_request(settings)\n\u001b[32m    111\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(response, AsyncStream):\n\u001b[32m    112\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m ServiceInvalidResponseError(\u001b[33m\"\u001b[39m\u001b[33mExpected an AsyncStream[ChatCompletionChunk] response.\u001b[39m\u001b[33m\"\u001b[39m)\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/connectors/ai/open_ai/services/open_ai_handler.py:59\u001b[39m, in \u001b[36mOpenAIHandler._send_request\u001b[39m\u001b[34m(self, settings)\u001b[39m\n\u001b[32m     57\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m.ai_model_type == OpenAIModelTypes.TEXT \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m.ai_model_type == OpenAIModelTypes.CHAT:\n\u001b[32m     58\u001b[39m     \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(settings, OpenAIPromptExecutionSettings)  \u001b[38;5;66;03m# nosec\u001b[39;00m\n\u001b[32m---> \u001b[39m\u001b[32m59\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mawait\u001b[39;00m \u001b[38;5;28mself\u001b[39m._send_completion_request(settings)\n\u001b[32m     60\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m.ai_model_type == OpenAIModelTypes.EMBEDDING:\n\u001b[32m     61\u001b[39m     \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(settings, OpenAIEmbeddingPromptExecutionSettings)  \u001b[38;5;66;03m# nosec\u001b[39;00m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/semantic_kernel/connectors/ai/open_ai/services/open_ai_handler.py:104\u001b[39m, in \u001b[36mOpenAIHandler._send_completion_request\u001b[39m\u001b[34m(self, settings)\u001b[39m\n\u001b[32m     99\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m ServiceResponseException(\n\u001b[32m    100\u001b[39m         \u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mtype\u001b[39m(\u001b[38;5;28mself\u001b[39m)\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m service failed to complete the prompt\u001b[39m\u001b[33m\"\u001b[39m,\n\u001b[32m    101\u001b[39m         ex,\n\u001b[32m    102\u001b[39m     ) \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mex\u001b[39;00m\n\u001b[32m    103\u001b[39m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m ex:\n\u001b[32m--> \u001b[39m\u001b[32m104\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m ServiceResponseException(\n\u001b[32m    105\u001b[39m         \u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mtype\u001b[39m(\u001b[38;5;28mself\u001b[39m)\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m service failed to complete the prompt\u001b[39m\u001b[33m\"\u001b[39m,\n\u001b[32m    106\u001b[39m         ex,\n\u001b[32m    107\u001b[39m     ) \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mex\u001b[39;00m\n",
      "\u001b[31mServiceResponseException\u001b[39m: (\"<class 'semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.OpenAIChatCompletion'> service failed to complete the prompt\", AuthenticationError(\"Error code: 401 - {'error': {'code': 'unauthorized', 'message': 'The `models` permission is required to access this endpoint', 'details': 'The `models` permission is required to access this endpoint'}}\"))"
     ]
    }
   ],
   "source": [
    "async def main():\n",
    "    # Create a new thread for the agent\n",
    "    # If no thread is provided, a new thread will be\n",
    "    # created and returned with the initial response\n",
    "    thread: ChatHistoryAgentThread | None = None\n",
    "\n",
    "    user_inputs = [\n",
    "        \"Plan me a day trip.\",\n",
    "    ]\n",
    "\n",
    "    for user_input in user_inputs:\n",
    "        print(f\"# User: {user_input}\\n\")\n",
    "        first_chunk = True\n",
    "        async for response in agent.invoke_stream(\n",
    "            messages=user_input, thread=thread,\n",
    "        ):\n",
    "            # 5. Print the response\n",
    "            if first_chunk:\n",
    "                print(f\"# {response.name}: \", end=\"\", flush=True)\n",
    "                first_chunk = False\n",
    "            print(f\"{response}\", end=\"\", flush=True)\n",
    "            thread = response.thread\n",
    "        print()\n",
    "\n",
    "    # Clean up the thread\n",
    "    await thread.delete() if thread else None\n",
    "\n",
    "await main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AZURE_OPENAI_API_KEY = FWBYKUuZaCkodA9RzI1aXKG2j0fzFNvDIUsEajKamLEiiFACcbX5JQQJ99BFACYeBjFXJ3w3AAAAACOGO8ji\n",
      "AZURE_OPENAI_ENDPOINT = https://agentlearning3408569709.openai.azure.com/\n",
      "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = ...\n",
      "AZURE_OPENAI_API_VERSION = ...\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "vars_to_check = [\n",
    "    \"AZURE_OPENAI_API_KEY\",\n",
    "    \"AZURE_OPENAI_ENDPOINT\",\n",
    "    \"AZURE_OPENAI_CHAT_DEPLOYMENT_NAME\",\n",
    "    \"AZURE_OPENAI_API_VERSION\"\n",
    "]\n",
    "\n",
    "for var in vars_to_check:\n",
    "    print(f\"{var} = {os.getenv(var)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "APIRemovedInV1",
     "evalue": "\n\nYou tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.\n\nYou can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. \n\nAlternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`\n\nA detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742\n",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mAPIRemovedInV1\u001b[39m                            Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[7]\u001b[39m\u001b[32m, line 11\u001b[39m\n\u001b[32m      7\u001b[39m openai.api_version = os.getenv(\u001b[33m\"\u001b[39m\u001b[33mAZURE_OPENAI_API_VERSION\u001b[39m\u001b[33m\"\u001b[39m)  \u001b[38;5;66;03m# e.g. 2024-03-01-preview\u001b[39;00m\n\u001b[32m      9\u001b[39m deployment = os.getenv(\u001b[33m\"\u001b[39m\u001b[33mAZURE_OPENAI_CHAT_DEPLOYMENT_NAME\u001b[39m\u001b[33m\"\u001b[39m)\n\u001b[32m---> \u001b[39m\u001b[32m11\u001b[39m response = \u001b[43mopenai\u001b[49m\u001b[43m.\u001b[49m\u001b[43mChatCompletion\u001b[49m\u001b[43m.\u001b[49m\u001b[43mcreate\u001b[49m\u001b[43m(\u001b[49m\n\u001b[32m     12\u001b[39m \u001b[43m    \u001b[49m\u001b[43mengine\u001b[49m\u001b[43m=\u001b[49m\u001b[43mdeployment\u001b[49m\u001b[43m,\u001b[49m\n\u001b[32m     13\u001b[39m \u001b[43m    \u001b[49m\u001b[43mmessages\u001b[49m\u001b[43m=\u001b[49m\u001b[43m[\u001b[49m\n\u001b[32m     14\u001b[39m \u001b[43m        \u001b[49m\u001b[43m{\u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mrole\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m:\u001b[49m\u001b[43m \u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43muser\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mcontent\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m:\u001b[49m\u001b[43m \u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mHello, who are you?\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m}\u001b[49m\n\u001b[32m     15\u001b[39m \u001b[43m    \u001b[49m\u001b[43m]\u001b[49m\n\u001b[32m     16\u001b[39m \u001b[43m)\u001b[49m\n\u001b[32m     18\u001b[39m \u001b[38;5;28mprint\u001b[39m(response[\u001b[33m'\u001b[39m\u001b[33mchoices\u001b[39m\u001b[33m'\u001b[39m][\u001b[32m0\u001b[39m][\u001b[33m'\u001b[39m\u001b[33mmessage\u001b[39m\u001b[33m'\u001b[39m][\u001b[33m'\u001b[39m\u001b[33mcontent\u001b[39m\u001b[33m'\u001b[39m])\n",
      "\u001b[36mFile \u001b[39m\u001b[32m/data/nas/yexin/workspace/huanghj/project/agent/ai-agents-for-beginners/.conda/lib/python3.12/site-packages/openai/lib/_old_api.py:39\u001b[39m, in \u001b[36mAPIRemovedInV1Proxy.__call__\u001b[39m\u001b[34m(self, *_args, **_kwargs)\u001b[39m\n\u001b[32m     38\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m, *_args: Any, **_kwargs: Any) -> Any:\n\u001b[32m---> \u001b[39m\u001b[32m39\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m APIRemovedInV1(symbol=\u001b[38;5;28mself\u001b[39m._symbol)\n",
      "\u001b[31mAPIRemovedInV1\u001b[39m: \n\nYou tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.\n\nYou can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. \n\nAlternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`\n\nA detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742\n"
     ]
    }
   ],
   "source": [
    "import openai\n",
    "import os\n",
    "\n",
    "openai.api_type = \"azure\"\n",
    "openai.api_key = os.getenv(\"AZURE_OPENAI_API_KEY\")\n",
    "openai.api_base = os.getenv(\"AZURE_OPENAI_ENDPOINT\")\n",
    "openai.api_version = os.getenv(\"AZURE_OPENAI_API_VERSION\")  # e.g. 2024-03-01-preview\n",
    "\n",
    "deployment = os.getenv(\"AZURE_OPENAI_CHAT_DEPLOYMENT_NAME\")\n",
    "\n",
    "response = openai.ChatCompletion.create(\n",
    "    engine=deployment,\n",
    "    messages=[\n",
    "        {\"role\": \"user\", \"content\": \"Hello, who are you?\"}\n",
    "    ]\n",
    ")\n",
    "\n",
    "print(response['choices'][0]['message']['content'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
