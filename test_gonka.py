from openai import OpenAI
 
client = OpenAI(
    api_key="sk-RcsRzbUbHNLw51arJYveY5GtviqjTJRT7LWM2CPxnthasGOT",                          
    base_url="https://api.gonkarouter.io/v1",       
)
 
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",     
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)
 
print(response.choices[0].message.content)