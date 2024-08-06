from openai import OpenAI
import os
from functools import lru_cache
from retry import retry


@retry(tries=3)
def chat_with_model(prompt, model, max_tokens=4000, temperature=0):
    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content


@lru_cache(maxsize=10000)
@retry(tries=3)
def embed(text):
    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-large", input=[text])
    return response.data[0].embedding

@lru_cache(maxsize=10000)
@retry(tries=3)
def embed_octo(text):
    client = OpenAI(
        api_key=os.getenv("OCTO_API_KEY"),
        base_url="https://text.octoai.run/v1")

    response = client.embeddings.create(
        model="thenlper/gte-large", input=[text])
    return response.data[0].embedding
