import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def web_search_allowed(query: str, domain: str, max_results: int = 5):
    q = f"site:{domain} {query}"
    out = []

    with DDGS() as ddgs:
        for r in ddgs.text(q, max_results=max_results):
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": (r.get("href") or r.get("url") or "").strip(),
                "snippet": (r.get("body") or "").strip(),
            })

    return [x for x in out if x["url"]]


def fetch_page_text(url: str, timeout: int = 10, max_chars: int = 6000) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:max_chars]


def pick_short_quotes(text: str, max_quotes: int = 2, max_len: int = 220):
    quotes = []
    for line in text.splitlines():
        if len(line) < 40:
            continue
        quotes.append(line[:max_len].strip())
        if len(quotes) >= max_quotes:
            break
    return quotes
