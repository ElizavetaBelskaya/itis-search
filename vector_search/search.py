import os
import math
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from nltk import word_tokenize

class VectorSearchEngine:

    def __init__(self, pages_dir, tfidf_dir):
        self.pages_dir = pages_dir
        self.tfidf_dir = tfidf_dir
        self.N = len([f for f in os.listdir(tfidf_dir) if f.startswith("tfidf_tokens_")])
        self.doc_vectors, self.idf = self.load_tfidf_vectors()


    def load_tfidf_vectors(self):
        doc_vectors = {}
        idf_accumulator = defaultdict(list)

        for filename in os.listdir(self.tfidf_dir):
            if not filename.startswith("tfidf_lemmas_"):
                continue
            doc_id = int(filename.split("_")[-1].split(".")[0])
            tfidf = {}
            with open(os.path.join(self.tfidf_dir, filename), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    term, idf_val, tfidf_val = parts[0], float(parts[1]), float(parts[2])
                    tfidf[term] = tfidf_val
                    idf_accumulator[term].append(idf_val)
            doc_vectors[doc_id] = tfidf

        # агрегируем IDF — можно взять среднее (если нужно), но берём первое значение
        idf = {term: vals[0] for term, vals in idf_accumulator.items()}
        return doc_vectors, idf


    def query_to_vector(self, query):
        tokens = word_tokenize(query.lower())
        terms = [t for t in tokens if t.isalnum()]
        present_terms = [term for term in terms if term in self.idf]
        tf = Counter(present_terms)
        total = sum(tf.values())
        return {term: (freq / total) * self.idf.get(term, 0) for term, freq in tf.items() if term in self.idf}

    def cosine_similarity(self, vec1, vec2):
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        dot = sum(vec1[t] * vec2[t] for t in common)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def get_snippet(self, doc_id, query_terms, max_words=100):
        try:
            path = os.path.join(self.pages_dir, f"page_{doc_id}.txt")
            with open(path, 'r', encoding='utf-8') as f:
                html = f.read()
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            words = text.split()

            best_score = 0
            best_snippet = words[:max_words] if words else []

            for i in range(0, len(words) - max_words + 1):
                window = words[i:i + max_words]
                window_score = sum(1 for w in window if any(q in w.lower() for q in query_terms))
                if window_score > best_score:
                    best_score = window_score
                    best_snippet = window

            snippet = ' '.join(best_snippet)
            return snippet
        except Exception:
            return ""

    def search(self, query, top_n=10):
        query_vector = self.query_to_vector(query)
        if not query_vector:
            return []

        scores = []
        for doc_id, vec in self.doc_vectors.items():
            if not any(term in vec for term in query_vector):
                continue
            score = self.cosine_similarity(query_vector, vec)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        query_terms = [t.lower() for t in word_tokenize(query) if t.isalnum()]

        results = []
        for doc_id, score in scores[:top_n]:
            snippet = self.get_snippet(doc_id, query_terms)
            results.append({
                "doc_id": doc_id,
                "score": score,
                "snippet": snippet
            })
        return results

    def run_console(self):
        print("\nПоисковая система запущена. Введите 'exit' для выхода.\n")
        while True:
            query = input("Введите запрос: ").strip()
            if query.lower() == 'exit':
                break
            results = self.search(query)
            if not results:
                print("\nНичего не найдено.\n")
                continue
            print(f"\nТоп {len(results)} результатов:")
            for i, res in enumerate(results, 1):
                print(f"{i}. Документ {res['doc_id']} (релевантность: {res['score']:.4f})")
                print(f"   {res['snippet'][:300]}...\n")


if __name__ == '__main__':
    search_engine = VectorSearchEngine(
        pages_dir="../crawler/downloaded_pages",
        tfidf_dir="../tfidf_lemmas"
    )
    search_engine.run_console()
