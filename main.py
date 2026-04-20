import fastapi_poe as fp
from openai import AsyncOpenAI, RateLimitError
import httpx
import asyncio
import os
import json
import sys

class MistralBot(fp.PoeBot):
    def __init__(self):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=os.environ["MISTRAL_API_KEY"],
            base_url="https://api.mistral.ai/v1",
        )

    async def web_search(self, query: str) -> str:
        async with httpx.AsyncClient() as http:
            res = await http.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": os.environ["TAVILY_API_KEY"],
                    "query": query,
                    "max_results": 3,
                    "search_depth": "basic"
                },
                timeout=10
            )
            results = res.json().get("results", [])
            if not results:
                return "No results found."
            return "\n\n".join([
                f"Source: {r['url']}\n{r['title']}\n{r['content']}"
                for r in results
            ])

    async def get_response(self, request: fp.QueryRequest):
        messages = []

        messages.append({
            "role": "system",
            "content": """You are a helpful assistant with access to web search.
If the user asks about current events, recent news, prices, weather, or anything that requires up-to-date information, respond ONLY with this exact format:
SEARCH: <your search query>
Otherwise, answer directly without searching."""
        })

        for msg in request.query:
            role = "assistant" if msg.role == "bot" else msg.role

            # Handle attachment (gambar)
            if msg.attachments:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for attachment in msg.attachments:
                    if attachment.content_type and attachment.content_type.startswith("image/"):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": attachment.url}
                        })
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": msg.content})

        print(f"Messages count: {len(messages)}", file=sys.stderr)
        print(f"Payload size: {len(json.dumps(messages))} bytes", file=sys.stderr)

        # Panggil Mistral pertama kali
        first_response = ""
        MAX_RETRIES = 3

        for attempt in range(MAX_RETRIES):
            try:
                stream = await self.client.chat.completions.create(
                    model="mistral-small-2603",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        first_response += delta
                break
            except RateLimitError as e:
                print(f"Rate limit (attempt {attempt + 1}): {e}", file=sys.stderr)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(5)
                else:
                    yield fp.PartialResponse(text="❌ Server overloaded, coba lagi nanti.")
                    return

        print(f"First response: {first_response[:100]}", file=sys.stderr)

        # Cek apakah model minta search
        if first_response.strip().startswith("SEARCH:"):
            query = first_response.strip().replace("SEARCH:", "").strip()
            yield fp.PartialResponse(text=f"🔍 Searching: *{query}*\n\n")

            search_results = await self.web_search(query)

            messages.append({"role": "assistant", "content": first_response})
            messages.append({
                "role": "user",
                "content": f"Here are the search results:\n\n{search_results}\n\nNow answer the original question based on these results."
            })

            for attempt in range(MAX_RETRIES):
                try:
                    stream2 = await self.client.chat.completions.create(
                        model="mistral-small-2603",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024,
                        stream=True,
                    )
                    async for chunk in stream2:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield fp.PartialResponse(text=delta)
                    return
                except RateLimitError as e:
                    print(f"Rate limit search call (attempt {attempt + 1}): {e}", file=sys.stderr)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(5)
                    else:
                        yield fp.PartialResponse(text="❌ Server overloaded, coba lagi nanti.")
        else:
            yield fp.PartialResponse(text=first_response)

app = fp.make_app(MistralBot(), access_key=os.environ["POE_ACCESS_KEY"])
