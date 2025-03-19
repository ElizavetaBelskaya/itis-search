import os
import asyncio

import aiofiles
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag

# Начальная страница
START_URL = "https://kpop.fandom.com/wiki/BTS"
INDEX_FILE = "index.txt"
OUTPUT_FOLDER = "downloaded_pages"
MAX_PAGES = 200
CONCURRENT_REQUESTS = 10

# Создаем папку для скачанных страниц
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Храним посещенные страницы
visited_urls = set()
to_visit = set([START_URL])
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)


async def fetch(url: str, session: ClientSession):
    """Асинхронная загрузка страницы."""
    try:
        async with semaphore:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.text()
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
        return None


def clean_html(text: str) -> str:
    """Удаляет ненужные теги (JS, стили, метаданные) из HTML."""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "link", "meta"]):
        tag.decompose()
    return str(soup)


def extract_links(base_url: str, html: str) -> set:
    """Извлекает ссылки на страницы, убирает фрагменты (#section), параметры (?query=), исключает редиректы и файлы."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    excluded_extensions = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf",
                           ".doc", ".docx", ".xls", ".xlsx", ".mp4", ".mp3", ".zip"}

    for a_tag in soup.find_all("a", href=True):
        full_link = urljoin(base_url, a_tag["href"])  # Преобразуем относительные ссылки в абсолютные
        clean_link, _ = urldefrag(full_link)  # Удаляем fragment
        parsed = urlparse(clean_link)

        # Убираем параметры запроса (?query=123)
        clean_link = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Проверяем, не является ли ссылка файлом
        if (any(clean_link.lower().endswith(ext) for ext in excluded_extensions) or "file" in clean_link.lower()):
            continue

        # Фильтруем только ссылки на страницы в "kpop.fandom.com/wiki"
        if (clean_link not in visited_urls and parsed.scheme in {"http", "https"} and
                "kpop.fandom.com/wiki" in clean_link and "redirect" not in clean_link and
                "category" not in clean_link.lower()):
            links.add(clean_link)

    return links


async def save_page(index: int, url: str, html: str):
    """Сохраняет HTML-страницу и обновляет индексный файл."""
    file_name = f"page_{index}.txt"
    file_path = os.path.join(OUTPUT_FOLDER, file_name)
    async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
        await file.write(html)

    async with aiofiles.open(INDEX_FILE, "a", encoding="utf-8") as index_file:
        await index_file.write(f"{index}: {url}\n")

    print(f"[✓] Страница {index} сохранена: {url}")


async def crawl():
    """Асинхронный обход страниц."""
    index = 1
    async with aiohttp.ClientSession() as session:
        while to_visit and index <= MAX_PAGES:
            url = to_visit.pop()
            if url in visited_urls:
                continue

            visited_urls.add(url)
            html = await fetch(url, session)

            if html:
                cleaned_html = clean_html(html)
                await save_page(index, url, cleaned_html)
                index += 1

                new_links = extract_links(url, html)
                to_visit.update(new_links)

        print(f"✅ Завершено: скачано {index - 1} страниц.")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(crawl())
