### Основы информационного поиска - 4с2s
Елизавета Бельская, 11-101

Альмира Мингазова, 11-103


### 🛠️ Установка

1️⃣ Клонируйте репозиторий:
```bash
git clone https://github.com/ElizavetaBelskaya/itis-search.git
```

2️⃣ Установите зависимости:
```bash
pip install -r requirements.txt
```

### Задание 1 - краулер
Этот проект — асинхронный веб-краулер, который загружает HTML-страницы с **KPOP Fandom** (сайта-wiki) и сохраняет их локально.
Скрипт написан на Python с использованием **aiohttp** и **BeautifulSoup**.

- Асинхронная загрузка страниц для повышения скорости работы  
- Очистка HTML от ненужных тегов (`script`, `style`, `meta`)  
- Сохранение страниц локально в `downloaded_pages/`  
- Логирование загруженных страниц в `index.txt`  
- Ограничение количества запросов для предотвращения блокировки  
- Извлечение и обход ссылок внутри KPOP Fandom wiki с исключением из обхода ссылок на файлы

### 🔧 Использование

Перейдите в папку crawler и запустите `crawler.py`:
```bash
python crawler.py
```

### ⚙️ Настройки
Вы можете изменить параметры в коде:
- `START_URL` — начальная страница
- `MAX_PAGES` — максимальное количество загружаемых страниц
- `CONCURRENT_REQUESTS` — максимальное количество параллельных запросов


### Задание 2 - токенизация и лемматизация

Инструмент для обработки HTML-документов на английском языке. Скрипт выполняет следующие задачи:
- Обходит все HTML-файлы, расположенные в папке `crawler/downloaded_pages`
- Извлекает из каждой страницы текст с помощью библиотеки BeautifulSoup
- Токенизирует текст, выделяя отдельные слова
- Фильтрует токены: исключает дубликаты, стоп-слова (союзы, предлоги), числа, а также "мусор" (например, токены, содержащие одновременно буквы и цифры)
- Лемматизирует токены с использованием WordNetLemmatizer
- Для каждой страницы сохраняет токены в файл `tokens_N.txt` в папке tokens
- Группирует токены по леммам и сохраняет результат в файл `lemmas_N.txt` в папке lemmas (на каждой строке: `<лемма> <токен1> <токен2> ... <токенN>`)

### 🔧 Использование

Перейдите в папку tokenizer-lemmatizer и запустите `main.py`:
```bash
python main.py
```

### Задание 3 - инвертированный индекс
*build_index.py*

Строит инвертированный индекс из файлов lemmas_<id>.txt и сохраняет его в inverted_index.tsv.

*search_by_index.py*

Загружает TSV-файл и предоставляет консольный интерфейс для булевого поиска с операторами (AND, OR, NOT) и скобками.

*inverted_index.tsv*

Файл с инвертированным индексом

### 🔧 Использование

Перейдите в папку index и запустите `build_index.py` для формирования файла с инвертированным индексом:
```bash
python index.py
```

Для поиска по индексу запустить файл `search_by_index.py`
```bash
python search_by_index.py
```

### Задание 4 - TF-IDF
1. Читает токены и леммы из соответствующих директорий для каждого документа (tokens_N.txt, lemmas_N.txt).
2. Считает TF, DF, TF-IDF: 
- для токенов — как частоту и количество документов, в которых они встречаются;  
- для лемм — агрегируя частоты всех токенов, относящихся к лемме.

```
TF (term frequency) — частота термина в документе:
TF(term, doc) = count(term in doc) / total_terms_in_doc
DF (document frequency) — в скольких документах встречается термин.
IDF (inverse document frequency):
IDF(term) = log(N / DF(term))
TF-IDF:
TF-IDF(term, doc) = TF(term, doc) * IDF(term)
```
3. Сохраняет результаты в два набора файлов:
- tfidf_tokens_N.txt — TF-IDF по токенам
- tfidf_lemmas_N.txt — TF-IDF по леммам


### 🔧 Использование

Перейдите в папку tf-idf и запустите `main.py` для формирования файлов:
```bash
python main.py
```

### Задание 5 - поисковая система
Векторная поисковая система на основе TF-IDF и косинусного сходства для документов.
Косинусное сходство — это метрика, которая измеряет угол между двумя векторами в многомерном пространстве.

1. Загружает лемматизированные документы и инвертированный индекс.
2. Строит TF-IDF векторы для каждого документа по леммам.
3. Обрабатывает запрос: разбивает его на токены, фильтрует по алфавиту.
4. Вычисляет TF-IDF вектор запроса.
5. Считает косинусное сходство между запросом и каждым документом.
6. Отображает топ-N документов с наивысшей релевантностью.

Для каждого результата отображаются: номер документа, релевантность (косинусное сходство), сниппет (до 100 слов вокруг термина запроса)

### 🔧 Использование

Перейдите в папку vector-search и запустите `search.py` для формирования файла с инвертированным индексом:
```bash
python search.py
```