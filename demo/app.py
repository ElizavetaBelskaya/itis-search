import os

from flask import Flask, render_template, request, Response, abort
from vector_search.search import VectorSearchEngine
import re

app = Flask(__name__)

search_engine = VectorSearchEngine(
    index_file="../index/inverted_index.tsv",
    lemmas_dir="../tokenizer-lemmatizer/lemmas",
    pages_dir="../crawler/downloaded_pages",
    tfidf_dir="../tf-idf/tfidf_lemmas"
)


def highlight(text, query_terms):
    pattern = re.compile(r'(' + '|'.join(map(re.escape, query_terms)) + r')', re.IGNORECASE)
    return pattern.sub(r'<mark>\1</mark>', text)


@app.route("/", methods=["GET"])
def index():
    return render_template("results.html")



@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("index.html", error="Введите запрос")

    results = search_engine.search(query, top_n=10)
    query_terms = query.lower().split()

    for result in results:
        result["snippet"] = highlight(result["snippet"], query_terms)

    return render_template("results.html", query=query, results=results)


@app.route('/pages/<path:filename>')
def serve_page(filename):
    file_path = os.path.join('pages', filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content, mimetype='text/html')
    else:
        abort(404)


if __name__ == "__main__":
    app.run(debug=True)