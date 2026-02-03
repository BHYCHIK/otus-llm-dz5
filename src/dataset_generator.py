import json
import os
import shutil

import requests
from bs4 import BeautifulSoup

from concurrent.futures import ThreadPoolExecutor, as_completed

# сколько одновременно качать PDF (подберите под сеть/диск)
PDF_WORKERS = 15
# сколько одновременно парсить категорий
CAT_WORKERS = 8

def download_pdf(link, path, session):
    try:
        pdf_response = session.get(link, stream=True)
        pdf_response.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(pdf_response.raw, f)
    except Exception as e:
        print(f"Error downloading PDF: {e}")

def download_by_meta(meta, category, session):
    os.makedirs('data/' + category, exist_ok=True)
    with open('data/' + category + '/meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as executor:
        futures = [
            executor.submit(download_pdf, m['link'], 'data/' + category + '/' + m['id'] + '.pdf', session)
            for m in meta
        ]
        for fut in futures:
            fut.result()

def get_category_page(link, category, session):
    resp = session.get(link)
    if resp.status_code != 200:
        raise Exception(f"Invalid status code {resp.status_code}")
    soup = BeautifulSoup(resp.content, "html.parser")
    articles = soup.find_all("div", class_="meta")
    metadata = []
    for article in articles:
        title = article.find("div", class_="list-title").get_text()[6:].strip()
        authors = article.find("div", class_="list-authors").get_text().split(',')
        authors = [authors.strip() for authors in authors]
        metadata.append({"title": title, "authors": authors})
    pdf_links = soup.find_all("a", attrs={'title': 'Download PDF'})
    i = 0
    for pdf_link in pdf_links:
        link = 'https://arxiv.org' + pdf_link["href"]
        metadata[i]["id"] = link.split("/")[-1]
        metadata[i]["link"] = link
        metadata[i]["category"] = category
        i += 1
    download_by_meta(metadata, category, session)

def main():
    with requests.Session() as session:
        links_to_parse = []

        resp = session.get("https://arxiv.org/")
        if resp.status_code != 200:
            raise Exception(f"Invalid status code {resp.status_code}")
        soup = BeautifulSoup(resp.content, "html.parser")
        links = soup.find_all("a")
        for link in links:
            if link.has_attr("href") and link["href"].endswith("/new"):
                href = 'https://arxiv.org' + link["href"]
                cat = link.get("aria-labelledby").split(" ", 1)[1]
                links_to_parse.append({'href': href, 'cat': cat})

        with ThreadPoolExecutor(max_workers=CAT_WORKERS) as executor:
            futures = [
                executor.submit(get_category_page, link_to_parse['href'], link_to_parse['cat'], session)
                for link_to_parse in links_to_parse
            ]
            for fut in futures:
                fut.result()



if __name__ == "__main__":
    main()