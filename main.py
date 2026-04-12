import fastapi_poe as fp
from openai import AsyncOpenAI
import httpx
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
            role = msg.role
            if role == "bot":
                role = "assistant"
            messages.append({"role": role, "content": msg.content})

        # Batasi history
        MAX_MESSAGES = 10
        if len(messages) > MAX_MESSAGES + 1:
            messages = [messages[0]] + messages[-MAX_MESSAGES:]

        print(f"Messages count: {len(messages)}", file=sys.stderr)
        print(f"Payload size: {len(json.dumps(messages))} bytes", file=sys.stderr)

        # Panggil Mistral pertama kali
        first_response = ""
        stream = await self.client.chat.completions.create(
            model="ministral-8b-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                first_response += delta

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

            # Panggil Mistral kedua kali dengan hasil search
            stream2 = await self.client.chat.completions.create(
                model="ministral-8b-latest",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream2:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield fp.PartialResponse(text=delta)
        else:
            yield fp.PartialResponse(text=first_response)

app = fp.make_app(MistralBot(), access_key=os.environ["POE_ACCESS_KEY"])
