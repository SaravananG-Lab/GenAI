import os
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from IPython.display import display, HTML
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.models import Gemini
import asyncio

api_Key_name="GOOGLE_API_KEY"
api_key_value=os.environ.get(api_Key_name)

if not api_key_value:
    raise ValueError(f"Environment variable '{api_Key_name}' is not set.")
print(f"API Key retrieved successfully: {api_key_value[:4]}****")  # Print only the first 4 characters for security

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

google_search_tool =GoogleSearchTool()

root_agent= Agent(
    name="helpful_agent",
    model = Gemini(model="gemini-2.5-flash-lite", retry_option=retry_config),
    static_instruction = "You are a helpful assistant. Use Google search for current information or if unsure.",
    tools=[google_search_tool]
)

runner = InMemoryRunner(agent=root_agent)

def extract_text(response):
    results = []

    if isinstance(response, list):
        for turn in response:
            if hasattr(turn, "content"):
                for c in turn.content:
                    # Some responses return tuple: (role, ResponseText)
                    if isinstance(c, tuple) and len(c) == 2:
                        c = c[1]

                    if hasattr(c, "text") and c.text:
                        results.append(c.text)

    return "\n".join(results)

async def run_agent():
    response = await runner.run_debug(
        "What is the ADK in Google GenAI? Provide a brief summary."
    )
    
    text = extract_text(response)
    display(HTML(text))

asyncio.run(run_agent())