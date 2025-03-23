import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Папка с HTML документами
html_dir = '../crawler/downloaded_pages'

# Множество для хранения уникальных токенов
tokens_set = set()

# Регулярное выражение для токенов, содержащих и буквы, и цифры
pattern_alphanum = re.compile(r'^(?=.*[a-zA-Z])(?=.*\d).+$')

# Загрузка списка стоп-слов (предлоги, союзы и т.д.) на английском языке
stop_words = set(stopwords.words('english'))


for filename in os.listdir(html_dir):
    filepath = os.path.join(html_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    # Токенизация текста
    tokens = word_tokenize(text)
    for token in tokens:
        token_lower = token.lower()
        # Фильтрация: токен должен состоять только из букв,
        # не быть стоп-словом, числом, и не содержать и букв, и цифр одновременно
        if re.fullmatch(r'[a-zA-Z]+', token_lower):
            if token_lower not in stop_words:
                tokens_set.add(token_lower)


with open('tokens.txt', 'w', encoding='utf-8') as f:
    for token in sorted(tokens_set):
        f.write(token + '\n')

lemmatizer = WordNetLemmatizer()
lemma_groups = {}

for token in tokens_set:
    lemma = lemmatizer.lemmatize(token)
    if lemma not in lemma_groups:
        lemma_groups[lemma] = []
    lemma_groups[lemma].append(token)


with open('lemma_tokens.txt', 'w', encoding='utf-8') as f:
    for lemma, token_list in sorted(lemma_groups.items()):
        line = lemma + ' ' + ' '.join(sorted(token_list))
        f.write(line + '\n')
