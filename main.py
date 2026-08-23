import fastapi_poe as fp
from openai import AsyncOpenAI, RateLimitError
import httpx
import asyncio
import os
import json
import sys
import html
from datetime import datetime
from zoneinfo import ZoneInfo

# Isi dengan URL GIF kamu (mis. hasil upload ke poecdn). Kosongkan untuk fallback teks.
SEARCHING_GIF_URL = "https://app.ferri.my.id/galery/catwalk-mini.gif"


class MistralBot(fp.PoeBot):
    def __init__(self):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=os.environ["MISTRAL_API_KEY"],
            base_url="https://api.mistral.ai/v1",
        )

    def _clean(self, text: str) -> str:
        # Escape HTML lalu hilangkan newline (baris kosong = fatal untuk blok <html> di Poe)
        return html.escape(text).replace("\n", " ").replace("\r", " ")

    async def web_search_raw(self, query: str) -> list:
        async with httpx.AsyncClient() as http:
            res = await http.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": os.environ["TAVILY_API_KEY"],
                    "query": query,
                    "max_results": 10,
                    "search_depth": "advanced"
                },
                timeout=10
            )
            print(f"Tavily status: {res.status_code}", file=sys.stderr)
            data = res.json()
            return data.get("results", [])

    def build_source_block(self, query: str, results: list) -> str:
        if results:
            items = "".join(
                f"<li style=\"margin-bottom:8px;\">"
                f"<a href=\"{self._clean(r['url'])}\" style=\"color:#7c9cff;text-decoration:none;font-weight:600;\">{self._clean(r['title'])}</a>"
                f"<div style=\"opacity:0.75;font-size:13px;line-height:1.4;margin-top:2px;\">{self._clean(r['content'][:300])}</div>"
                f"</li>"
                for r in results
            )
        else:
            items = "<li style=\"opacity:0.7;\">No results found.</li>"

        # \n\n di DEPAN: pemisah blok Markdown supaya kartu tidak nempel ke akhir jawaban.
        return (
            "\n\n"
            "<html>"
            "<details style=\"background:#1e1e24;border:1px solid #33333d;border-radius:12px;padding:10px 14px;margin:6px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;\">"
            "<summary style=\"cursor:pointer;list-style:none;font-weight:600;color:#c7c7d1;display:flex;align-items:center;gap:8px;\">"
            "<span style=\"opacity:0.8;\">🔗</span>"
            "<span>Sources</span>"
            "<span style=\"opacity:0.5;font-weight:400;font-size:13px;margin-left:auto;\">click to expand</span>"
            "</summary>"
            "<div style=\"margin-top:10px;padding-top:10px;border-top:1px solid #33333d;color:#b8b8c2;font-size:14px;\">"
            f"<div style=\"opacity:0.7;margin-bottom:8px;\">Dicari: <i>{self._clean(query)}</i></div>"
            f"<ul style=\"margin:0;padding-left:18px;\">{items}</ul>"
            "</div>"
            "</details>"
            "</html>"
        )

    async def get_response(self, request: fp.QueryRequest):
        messages = []

        now = datetime.now(ZoneInfo("Asia/Jakarta"))
        current_time = now.strftime("%A, %d %B %Y, %H:%M %Z")

        messages.append({
            "role": "system",
            "content": f"""You are a helpful, time sensitive assistant. The current date and time is: {current_time}.
If the user asks about current events, recent news, prices, weather, or anything that IS NOT IN YOUR TRAINING DATA, you must search first.
When you need to search, reply with ONLY a single line in this exact format and NOTHING ELSE:
SEARCH: <your search query>
Do not add any explanation, answer, or extra text on that turn. Otherwise, answer directly without searching."""
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

        # Hit pertama: memutuskan search atau jawab langsung.
        # Kita stream sambil "mengintip": tahan token sampai yakin ini SEARCH atau bukan.
        # - Kalau ternyata jawab langsung -> gelontorkan token yang tertahan lalu stream sisanya (tanpa hit kedua).
        # - Kalau ternyata SEARCH -> jangan tampilkan apa-apa, lanjut ke alur search.
        first_response = ""
        is_search = None          # None = belum tahu, True/False = sudah diputuskan
        buffer = ""               # penampung token awal sebelum keputusan
        MARKER = "SEARCH:"
        MAX_RETRIES = 3

        for attempt in range(MAX_RETRIES):
            try:
                stream = await self.client.chat.completions.create(
                    model="ministral-8b-2512",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue
                    first_response += delta

                    if is_search is None:
                        buffer += delta
                        stripped = buffer.lstrip()
                        # Belum cukup karakter untuk memutuskan & masih mungkin jadi "SEARCH:" -> tunggu.
                        if len(stripped) < len(MARKER) and MARKER.startswith(stripped):
                            continue
                        if stripped.startswith(MARKER):
                            is_search = True
                            # jangan yield apa-apa; query diambil dari first_response nanti
                        else:
                            is_search = False
                            yield fp.PartialResponse(text=buffer)  # jawab langsung: keluarkan yang tertahan
                    elif is_search is False:
                        yield fp.PartialResponse(text=delta)       # jawab langsung: stream sisanya
                    # kalau is_search is True: diam, tampung di first_response saja
                break
            except RateLimitError as e:
                print(f"Rate limit (attempt {attempt + 1}): {e}", file=sys.stderr)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(5)
                else:
                    yield fp.PartialResponse(text="❌ Server overloaded, coba lagi nanti.")
                    return

        print(f"First response: {first_response[:100]}", file=sys.stderr)

        # Kalau bukan search, jawaban sudah di-stream di atas. Selesai.
        if is_search is not True:
            return

        # ---- Alur SEARCH ----
        # Ambil HANYA baris pertama sebagai query (buang sisa jawaban jika model bocor)
        first_line = first_response.strip().split("\n")[0]
        query = first_line.replace("SEARCH:", "").strip()

                # Indikator "sedang mencari".
        # Pakai post_message_attachment supaya Poe yang host GIF-nya (anti-blokir).
        if SEARCHING_GIF_URL:
            try:
                attachment = await self.post_message_attachment(
                    message_id=request.message_id,
                    download_url=SEARCHING_GIF_URL,
                    is_inline=True,
                )
                yield fp.PartialResponse(text=f"![searching]{attachment.inline_ref}\n\n")
            except Exception as e:
                # Kalau upload gagal (URL mati/diblok saat download), fallback ke teks.
                print(f"GIF attach failed: {e}", file=sys.stderr)
                yield fp.PartialResponse(text=f"🔎 Looking for: *{query}*\n\n")
        else:
            yield fp.PartialResponse(text=f"🔎 Looking for: *{query}*\n\n")

        # Rebuild flat results string to feed back into Mistral
        if results:
            search_results = "\n\n".join(
                f"Source: {r['url']}\n{r['title']}\n{r['content']}"
                for r in results
            )
        else:
            search_results = "No results found."

        messages.append({"role": "assistant", "content": first_response})
        messages.append({
            "role": "user",
            "content": f"Here are the search results:\n\n{search_results}\n\nNow answer the original question based on these results."
        })

        for attempt in range(MAX_RETRIES):
            try:
                stream2 = await self.client.chat.completions.create(
                    model="ministral-8b-2512",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=True,
                )
                async for chunk in stream2:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield fp.PartialResponse(text=delta)

                # Jawaban selesai -> tampilkan blok sumber di BAWAH jawaban
                yield fp.PartialResponse(text=self.build_source_block(query, results))
                return
            except RateLimitError as e:
                print(f"Rate limit search call (attempt {attempt + 1}): {e}", file=sys.stderr)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(5)
                else:
                    yield fp.PartialResponse(text="❌ Server overloaded, coba lagi nanti.")


app = fp.make_app(MistralBot(), access_key=os.environ["POE_ACCESS_KEY"])

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}
