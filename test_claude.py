import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
resp = client.messages.create(
    model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
)
print(resp.content[0].text)