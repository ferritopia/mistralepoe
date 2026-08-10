# Mistral POE Bot with Web Search

This project is a FastAPI-POE bot that uses Mistral's chat completion API as the backend model and optionally calls Tavily web search for up-to-date information when the model requests it.

## Features

- Uses **Mistral** chat completions via the OpenAI-compatible API (`mistral-large-latest` for the first response, `mistral-large-2512` for the post-search answer).
- Integrates with **Poe** using `fastapi_poe` to expose the bot as a Poe-compatible endpoint.
- Adds **Tavily** web search when the model responds with a `SEARCH: <query>` directive.
- Supports **image attachments** — images sent by the user are forwarded to Mistral as `image_url` content.
- Renders a collapsible **source card** (HTML) below the answer whenever a web search was performed.
- Includes **retry logic** (up to 3 attempts with a 5s backoff) to handle Mistral rate limits.
- Streams responses back to Poe for a responsive chat experience.
- Exposes a `/health` endpoint returning `{"status": "ok"}`.

## How it works

1. Every Poe request is converted into an OpenAI-style `messages` list with a system prompt (including the current Asia/Jakarta date/time) that instructs the model to either answer directly or respond with `SEARCH: <query>` when it needs fresh web data.
2. The bot calls Mistral's chat completion API in streaming mode and concatenates the first response.
3. If the first response starts with `SEARCH:`, the bot:
   - Extracts the query string (only the first line is used).
   - Calls Tavily's search API (`https://api.tavily.com/search`) for up to 5 concise results.
   - Injects the search results back into the conversation as a new user message and calls Mistral again.
   - Streams the second response to the user, followed by a source card listing the results.
4. If there is no `SEARCH:` directive, the bot simply streams the initial Mistral response.

## Requirements

- Python packages:
  - `fastapi_poe`
  - `openai` (new `AsyncOpenAI` client)
  - `httpx`

Install dependencies (example):

```
pip install fastapi_poe openai httpx
```

You may also want to use a virtual environment (`venv`, `poetry`, etc.) depending on your deployment setup.

## Environment variables

The bot is configured entirely via environment variables.

- `MISTRAL_API_KEY` – API key for Mistral (`https://api.mistral.ai/v1`).
- `TAVILY_API_KEY` – API key for Tavily search.
- `POE_ACCESS_KEY` – Access key used by Poe to authenticate to your FastAPI app.

Example `.env` (do not commit this file):

```
MISTRAL_API_KEY=your_mistral_key_here
TAVILY_API_KEY=your_tavily_key_here
POE_ACCESS_KEY=your_poe_access_key_here
```

## Running the app

`main.py` exposes a FastAPI application via `fastapi_poe.make_app`.
You can run it with any ASGI server, for example `uvicorn`:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then configure Poe to connect to your bot using the URL of this server and the `POE_ACCESS_KEY` you set above.

### Development tips

- Change the models (`mistral-large-latest` / `mistral-large-2512`) and generation parameters (`temperature`, `max_tokens`) directly in the `AsyncOpenAI.chat.completions.create` calls.
- Adjust `MAX_RETRIES` in `main.py` to change how many times the bot retries on rate limits.
- Add logging or observability around the `SEARCH:` branch to debug search behaviour or track API usage.

## Project structure

Current minimal layout:

```
mistralepoe/
├─ Procfile
├─ README.md
├─ requirements.txt
└─ main.py   # Poe bot implementation using Mistral + Tavily search
```

`main.py` defines the `MistralBot` class and the `app` object used by your ASGI server.
