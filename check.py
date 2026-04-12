import asyncio
import os
from google import genai
from dotenv import load_dotenv  # Add this

# This line reads the .env file and makes the keys available to os.getenv
load_dotenv() 

async def check_models():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print("Fetching models...")
    # Notice we AWAIT the list call before iterating
    models = await client.aio.models.list() 
    
    for model in models:
        if "embed" in model.name:
            print(f"Available: {model.name}")

if __name__ == "__main__":
    asyncio.run(check_models())