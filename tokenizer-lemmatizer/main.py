import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

INPUT_DIR = '../crawler/downloaded_pages'
TOKENS_DIR = 'tokens'
LEMMAS_DIR = 'lemmas'

# Создаем выходные директории
os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(LEMMAS_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
token_pattern = re.compile(r'^[a-zA-Z]{2,}$')


def process_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = set()

    for token in tokens:
        if token_pattern.fullmatch(token) and token not in stop_words:
            sub_tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)
            for sub_token in sub_tokens:
                if len(sub_token) >= 3:
                    filtered_tokens.add(sub_token)

    lemma_map = defaultdict(set)
    for token in filtered_tokens:
        lemma = lemmatizer.lemmatize(token)
        lemma_map[lemma].add(token)

    return filtered_tokens, lemma_map


def process_page(file_path, page_num):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)

    tokens, lemmas = process_text(text)

    tokens_file = os.path.join(TOKENS_DIR, f'tokens_{page_num}.txt')
    with open(tokens_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(tokens)))

    lemmas_file = os.path.join(LEMMAS_DIR, f'lemmas_{page_num}.txt')
    with open(lemmas_file, 'w', encoding='utf-8') as f:
        for lemma, token_list in sorted(lemmas.items()):
            f.write(f"{lemma} {' '.join(sorted(token_list))}\n")

    return len(tokens), len(lemmas)


def main():
    print("Начало обработки страниц...")

    total_pages = 0
    total_tokens = 0
    total_lemmas = 0

    for i in range(1, 201):
        filename = f'page_{i}.txt'
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            tokens_count, lemmas_count = process_page(file_path, i)
            total_pages += 1
            total_tokens += tokens_count
            total_lemmas += lemmas_count
            print(f"Обработана страница {i}: {tokens_count} токенов, {lemmas_count} лемм")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {str(e)}")

    print("\nИтоговая статистика:")
    print(f"Обработано страниц: {total_pages}")
    print(f"Всего уникальных токенов: {total_tokens}")
    print(f"Всего уникальных лемм: {total_lemmas}")
    print(f"\nРезультаты сохранены в:")
    print(f"- Токены: {TOKENS_DIR}/")
    print(f"- Леммы: {LEMMAS_DIR}/")


if __name__ == "__main__":
    main()
