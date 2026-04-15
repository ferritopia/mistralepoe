
# Mistral POE Bot with Web Search

This project is a FastAPI‑POE bot that uses Mistral's chat completion API as the backend model and optionally calls Tavily web search for up‑to‑date information when the model requests it.  

## Features

- Uses **Mistral** chat completions (`mistral-small-2603`) via the OpenAI‑compatible API.  
- Integrates with **Poe** using `fastapi_poe` to expose the bot as a Poe-compatible endpoint.  
- Adds **Tavily** web search when the model responds with a `SEARCH: <query>` directive.  
- Limits chat history length to keep payload size manageable (default: last 10 messages + system prompt).  
- Streams responses back to Poe for a responsive chat experience.  

## How it works

1. Every Poe request is converted into an OpenAI‑style `messages` list with a system prompt that instructs the model to either answer directly or respond with `SEARCH: <query>` when it needs fresh web data.  
2. The bot calls Mistral's chat completion API in streaming mode and concatenates the first response.  
3. If the first response starts with `SEARCH:`, the bot:
   - Extracts the query string.  
   - Calls Tavily's search API (`https://api.tavily.com/search`) with a small set of concise results.  
   - Injects the search results back into the conversation as a new user message and calls Mistral again.  
   - Streams the second response to the user.  
4. If there is no `SEARCH:` directive, the bot simply streams the initial Mistral response.  

## Requirements

- The following Python packages:
  - `fastapi_poe`  
  - `openai` (new `AsyncOpenAI` client)  
  - `httpx`  

Install dependencies (example):

```bash
pip install fastapi_poe openai httpx
```

You may also want to use a virtual environment (`venv`, `poetry`, etc.) depending on your deployment setup.

## Environment variables

The bot is configured entirely via environment variables.  

- `MISTRAL_API_KEY` – API key for Mistral (`https://api.mistral.ai/v1`).  
- `TAVILY_API_KEY` – API key for Tavily search.  
- `POE_ACCESS_KEY` – Access key used by Poe to authenticate to your FastAPI app.  

Example `.env` (do not commit this file):

```env
MISTRAL_API_KEY=your_mistral_key_here
TAVILY_API_KEY=your_tavily_key_here
POE_ACCESS_KEY=your_poe_access_key_here
```

## Running the app

`main.py` exposes a FastAPI application via `fastapi_poe.make_app`.  
You can run it with any ASGI server, for example `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then configure Poe to connect to your bot using the URL of this server and the `POE_ACCESS_KEY` you set above.

### Development tips

- Adjust `MAX_MESSAGES` in `main.py` if you want longer or shorter history per request.  
- You can change the default model (`mistral-small-2603`) and generation parameters (`temperature`, `max_tokens`) directly in the `AsyncOpenAI.chat.completions.create` call.  
- Add logging or observability around the `SEARCH:` branch to debug search behaviour or track API usage.  

## Project structure

Current minimal layout:

```text
mistralepoe/
├─ main.py   # Poe bot implementation using Mistral + Tavily search
```

`main.py` defines the `MistralBot` class and the `app` object used by your ASGI server.  
