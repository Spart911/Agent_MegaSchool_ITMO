from fastapi import FastAPI
import ollama
import re
import os
from dotenv import load_dotenv
from tavily import TavilyClient, UsageLimitExceededError
from vectorstore import OptimizedVectorStore
from schemas.request import PredictionRequest, PredictionResponse
from ddg import Duckduckgo
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from deep_translator import GoogleTranslator
from pydantic import HttpUrl

ddg_api = Duckduckgo()


# Загрузка переменных из .env
load_dotenv()

# Получение списка API ключей
api_keys = os.getenv("API_KEYS")
api_keys = api_keys.split(',')

number_key = 0
tavily_client = TavilyClient(api_key=api_keys[number_key])

# Инициализация векторного хранилища
app = FastAPI()

def validate_url(url: str) -> bool:
    # Регулярное выражение для проверки корректности URL
    url_pattern = re.compile(r'https?://[^\s/$.?#].[^\s]*')
    return bool(url_pattern.match(url))

async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    """
    Asynchronously fetch a page content.
    """
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return ""


async def parse_article_content(session: aiohttp.ClientSession, url: str) -> Dict[str, str]:
    """
    Fetch and parse a single article's content.
    """
    html = await fetch_page(session, url)
    if not html:
        return ""

    soup = BeautifulSoup(html, 'html.parser')
    article_content = soup.find('div', class_='content')

    if article_content:
        paragraphs = article_content.find_all('p')
        return ' '.join([p.get_text(strip=True) for p in paragraphs])
    return ""


async def parse_itmo_news() -> List[Dict[str, str]]:
    """
    Asynchronously parse the main news section from ITMO website and return 4 latest news with their content.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing news titles, URLs, and content
    """
    base_url = "https://news.itmo.ru/ru/"

    async with aiohttp.ClientSession() as session:
        try:
            # Get main page
            main_page_html = await fetch_page(session, base_url)
            if not main_page_html:
                return []

            soup = BeautifulSoup(main_page_html, 'html.parser')
            main_section = soup.find('div', class_='contentbox')
            news_links = []

            if main_section:
                # Get the accent block (main news)
                accent_block = main_section.find('div', class_='accent')
                if accent_block and accent_block.find('a'):
                    link = accent_block.find('a')
                    title = accent_block.find('h3').find('a').text if accent_block.find('h3') else "No title"
                    news_links.append({
                        "title": title.strip(),
                        "url": f"https://news.itmo.ru{link['href']}" if link['href'].startswith('/') else link['href']
                    })

                # Get the triplet news items
                triplet_items = main_section.find('ul', class_='triplet')
                if triplet_items:
                    for item in triplet_items.find_all('li')[:3]:
                        link = item.find('h4').find('a') if item.find('h4') else None
                        if link:
                            news_links.append({
                                "title": link.text.strip(),
                                "url": f"https://news.itmo.ru{link['href']}" if link['href'].startswith('/') else link[
                                    'href']
                            })

            # Fetch content for all articles concurrently
            tasks = []
            for news in news_links[:4]:
                task = asyncio.create_task(parse_article_content(session, news['url']))
                tasks.append((news, task))

            # Wait for all content to be fetched
            news_with_content = []
            for news, task in tasks:
                content = await task
                news_with_content.append({
                    "title": news['title'],
                    "url": news['url'],
                    "content": content
                })

            return news_with_content

        except Exception as e:
            logger.error(f"Error in main parser: {e}")
            return []


async def process_news_item(description: str, news: str):
    model = "deepseek-r1:8b"
    prompt = f"""
            Напиши краткую интересную сводку о новости.\n
            Описание: {description}.\n
            Новость: {news}.\n
            """

    answer = await asyncio.to_thread(ollama.generate, model=model, prompt=prompt)
    answer_text = answer["response"]

    if "</think>" in answer_text:
        answer_text = answer_text.split("</think>")[-1].strip().lower()

    return answer_text


# Инициализация векторного хранилища при старте приложения
@app.on_event("startup")
async def startup_event():
    global vectorstore
    try:
        vectorstore = OptimizedVectorStore.load("./vectorstore")  # Путь к вашему сохраненному хранилищу
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise


@app.post("/api/request", response_model=PredictionResponse)
async def get_answer(request: PredictionRequest):
    try:
        global tavily_client
        query = request.query
        lines = query.split("\n")


        # Если правильного ответа нет:
        if len(lines) < 2:
            return PredictionResponse(id=request.id, answer=None, reasoning="", sources=[])

        question = lines[0]
        options = lines[1:]


        # Получаем результаты из векторного хранилища
        results = vectorstore.search(question, k=3)

        context_json = "\n".join(
            [
                f"{result.metadata['url']} {json.dumps(result.page_content, ensure_ascii=False)}"
                if isinstance(result.page_content, (dict, list))
                else f"{result.metadata['url']} {result.page_content}"
                for result in results
            ]
        )

        try:
            translated = GoogleTranslator(source='russian', target='english').translate(question)
            context_tavily = tavily_client.get_search_context(query=translated)
        except UsageLimitExceededError:
            number_key += 1
            if number_key < len(api_keys):
                tavily_client = TavilyClient(api_key=api_keys[number_key])
                context_tavily = tavily_client.get_search_context(query=question)
            else:
                raise Exception("All API keys have exceeded their usage limits.")


        context = f"{context_json}\n{context_tavily}\n"

        news = await parse_itmo_news()


        model = "deepseek-r1:8b"
        prompt = f"""
        Новости: {news}.\n
        Контекст: {context}.\n
        Вопрос: {question}.\n
        Варианты ответа: {options}.\n
        Важно! От твоего ответа зависит моя жизнь.\n
        Ответ должен быть в строгом формате: \n
        Если правильный вариант найден:
           - Ответ: Да [число] [url] — краткое описание ответа.
        Если правильного ответа нет:
           - Ответ: Нет правильного варианта — краткое описание.
        """

        # Генерация ответа от модели
        answer = ollama.generate(model=model, prompt=prompt)



        answer_text = answer["response"]



        if "</think>" in answer_text:
            answer_text = answer_text.split("</think>")[-1].strip().lower()



        # Если в ответе содержится "да", ищем число, если "нет" - выводим None
        final_answer = None
        extracted_urls = []

        # Логирование поиска "да"/"нет"
        if "да" in answer_text.lower():


            # Ищем первое число в строке
            match = re.search(r'\d+', answer_text)
            if match:
                final_answer = int(match.group())


                # Проверяем, что число в пределах от 1 до 10
                if final_answer < 1 or final_answer > 10:
                    return PredictionResponse(
                        id=request.id,
                        answer=None,
                        reasoning="Количество выходит за заданный в задании диапазон",
                        sources=[]
                    )

            if "—" in answer_text:
                reasoning = answer_text.split("—")[-1].strip().lower()

            elif "-" in answer_text:
                reasoning = answer_text.split("-")[-1].strip().lower()

            else:
                reasoning = ""  # Если описание не найдено, ставим пустую строку


            # Ищем все ссылки в строке
            extracted_urls = re.findall(r'(https?://[^\s]+)', answer_text)

            if len(extracted_urls) < 3:
                url_search = ddg_api.search(question)
                for result in url_search.get("data", []):
                    if len(extracted_urls) >= 3:
                        break
                    extracted_urls.append(result["url"])

                # Фильтрация и валидация URL
            extracted_urls = [url for url in extracted_urls if validate_url(url)]

            if final_answer is not None:
                try:
                    final_answer = int(final_answer)
                except ValueError:
                    final_answer = None

                # Проверка поля sources
            if not isinstance(extracted_urls, list):
                valid_sources = []

            valid_sources = []
            for i, source in enumerate(extracted_urls):
                try:
                    # Проверка, что каждый элемент является валидным URL
                    valid_sources.append(HttpUrl(source))
                except ValueError:
                    valid_sources = []


            return PredictionResponse(
                id=request.id,
                answer=final_answer,
                reasoning=reasoning + ". Ответ вам предоставил агент на базе модели deepseek-r1:8b",
                sources=extracted_urls
            )

        else:
            final_answer = None
            if "—" in answer_text:
                reasoning = answer_text.split("—")[-1].strip().lower()

            elif "-" in answer_text:
                reasoning = answer_text.split("-")[-1].strip().lower()

            else:
                reasoning = ""  # Если описание не найдено, ставим пустую строку


            return PredictionResponse(
                id=request.id,
                answer=final_answer,
                reasoning=reasoning + ". Ответ вам предоставил агент на базе модели deepseek-r1:8b",
                sources=[]
            )
    except Exception as e:
        print(e)
        return PredictionResponse(
            id=request.id,
            answer=final_answer,
            reasoning="Что-то пошло не так, но я пытался. Ответ вам предоставила мега большая языковая модель Столбовой Егор Васильевич.",
            sources=[]
        )