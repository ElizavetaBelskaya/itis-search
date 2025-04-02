import os
import math
from collections import defaultdict, Counter

PAGES_DIR = '../crawler/downloaded_pages'
TOKENS_DIR = '../tokenizer-lemmatizer/tokens'
LEMMAS_DIR = '../tokenizer-lemmatizer/lemmas'
OUTPUT_TOKEN_DIR = 'tfidf_tokens'
OUTPUT_LEMMA_DIR = 'tfidf_lemmas'

os.makedirs(OUTPUT_TOKEN_DIR, exist_ok=True)
os.makedirs(OUTPUT_LEMMA_DIR, exist_ok=True)

file_indices = sorted([
    int(f.split('_')[-1].split('.')[0])
    for f in os.listdir(PAGES_DIR) if f.startswith('page_')
])
N = len(file_indices)

token_dfs = defaultdict(int)
lemma_dfs = defaultdict(int)
all_token_counts = {}         # токены по документам
all_lemma_map = {}            # лемма -> токены по документам

for idx in file_indices:
    with open(os.path.join(TOKENS_DIR, f'tokens_{idx}.txt'), 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
        counter = Counter(tokens)
        all_token_counts[idx] = counter
        unique_tokens = set(counter.keys())
        for token in unique_tokens:
            token_dfs[token] += 1

# Чтение всех лемм и формирование отображения: лемма -> токены (и df)
lemma_token_map_all_docs = {}
lemma_token_presence = defaultdict(set)

for idx in file_indices:
    lemma_map = defaultdict(list)
    with open(os.path.join(LEMMAS_DIR, f'lemmas_{idx}.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lemma, tokens = parts[0], parts[1:]
            lemma_map[lemma].extend(tokens)
            if any(t in all_token_counts[idx] for t in tokens):
                lemma_token_presence[lemma].add(idx)
    all_lemma_map[idx] = lemma_map

# Подсчет tf-idf для токенов и сохранение
for idx in file_indices:
    token_tf_idf_lines = []
    total_terms = sum(all_token_counts[idx].values())
    for token, count in all_token_counts[idx].items():
        tf = count / total_terms
        idf = math.log(N / token_dfs[token])
        tfidf = tf * idf
        token_tf_idf_lines.append(f"{token} {idf:.6f} {tfidf:.6f}\n")

    with open(os.path.join(OUTPUT_TOKEN_DIR, f'tfidf_tokens_{idx}.txt'), 'w', encoding='utf-8') as f:
        f.writelines(token_tf_idf_lines)

# Подсчет tf-idf для лемм и сохранение
for idx in file_indices:
    lemma_tf_idf_lines = []
    token_counts = all_token_counts[idx]
    total_terms = sum(token_counts.values())

    for lemma, tokens in all_lemma_map[idx].items():
        count = sum(token_counts[t] for t in tokens if t in token_counts)
        if count == 0:
            continue
        tf = count / total_terms
        idf = math.log(N / len(lemma_token_presence[lemma]))
        tfidf = tf * idf
        lemma_tf_idf_lines.append(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

    with open(os.path.join(OUTPUT_LEMMA_DIR, f'tfidf_lemmas_{idx}.txt'), 'w', encoding='utf-8') as f:
        f.writelines(lemma_tf_idf_lines)
