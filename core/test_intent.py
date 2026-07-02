import asyncio
import os

from dotenv import load_dotenv

from core.intent_recognizer import IntentRecognizer

load_dotenv()


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    r = IntentRecognizer(api_key=api_key) if api_key else IntentRecognizer()
    messages = [
        "I want to return something I bought last week",
        "I keep failing to log in, what's going on?",
        "Your support is way too slow, I want to file a complaint",
    ]
    for m in messages:
        res = await r.recognize(m)
        print(f"{m:55} -> {res.intent.value:10} conf={res.confidence:.2f}  {res.reasoning}")


asyncio.run(main())