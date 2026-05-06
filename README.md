# CAD Parsing x Rasterization x DSPY sub-agents 

# Replacing custom agents with openai-agents-sdk

```python
import asyncio
import os
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, function_tool

from dotenv import load_dotenv # Run: pip install python-dotenv
from agents import set_tracing_disabled

set_tracing_disabled(True)

load_dotenv() # This loads the variables from .env into os.environ

client = AsyncOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Mock for demo
    return f"Sunny and 22°C in {city}."

async def main():
    agent = Agent(
        name="Weather Qwen",
        instructions="You are a helpful assistant powered by Qwen. Use tools when needed.",
        tools=[get_weather],
        model=OpenAIChatCompletionsModel(
            model=os.getenv("VISION_MODEL"), 
            openai_client=client
        )
    )

    # Update this part:
    result = await Runner.run(
        agent,
        "What's the weather in New York?",
        # run_config=RunConfig(disable_tracing=True) # Explicitly turn off tracing
    )
    print(result.final_output)

asyncio.run(main())
```
