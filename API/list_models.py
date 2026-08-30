"""
Lists all models currently available on your GonkaRouter account.
Run this to get exact, valid model ID strings before hardcoding one.
"""

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["sk-RcsRzbUbHNLw51arJYveY5GtviqjTJRT7LWM2CPxnthasGOT"],
    base_url="https://api.gonkarouter.io/v1",
)

models = client.models.list()

print("Available models:\n")
for m in models.data:
    print(f"  {m.id}")
