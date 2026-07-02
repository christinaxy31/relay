import asyncio, os
from dotenv import load_dotenv
from core.intent_recognizer import IntentRecognizer

load_dotenv()

async def main():
    r = IntentRecognizer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    for m in ["I want a refund", "the app keeps crashing", "hello"]:
        res = await r.recognize(m)
        print(f"{m:30} -> {res.intent.value:10} conf={res.confidence:.2f}")
        print(f"     reasoning: {res.reasoning}")
        for name, v in res.votes.items():          # llm / emb / pat 各自
            print(f"     {name}: {v['intent']:10} conf={v['conf']:.2f}")
        print()

    await r.recognize("I want a refund")            # cache hit
    print("cache:", r.cache_hits, "hits /", r.cache_misses, "misses")

asyncio.run(main())