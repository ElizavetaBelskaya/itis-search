import os
import glob


def build_index(lemmas_folder):
    inverted_index = {}
    for filepath in glob.glob(os.path.join(lemmas_folder, "lemmas_*.txt")):
        filename = os.path.basename(filepath)
        try:
            file_id = int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Пропущен файл с некорректным именем: {filename}")
            continue

        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                lemma = parts[0]
                inverted_index.setdefault(lemma, set()).add(file_id)
    return inverted_index


def save_index_tsv(inverted_index, index_filename="inverted_index.tsv"):
    """
    Сохраняет инвертированный индекс и сопоставление lemma_id -> filename в TSV-файлы.

    Файл inverted_index.tsv:
        lemma    lemma_id1 lemma_id2 lemma_id3 ...
    """
    with open(index_filename, "w", encoding="utf-8") as index_file:
        index_file.write("term\tfile_ids\n")
        for term in sorted(inverted_index.keys()):
            file_ids = sorted(inverted_index[term])
            file_ids_str = " ".join(map(str, file_ids))
            index_file.write(f"{term}\t{file_ids_str}\n")


def main():
    lemmas_folder = "../tokenizer-lemmatizer/lemmas"
    inverted_index = build_index(lemmas_folder)
    save_index_tsv(inverted_index)
    print("Индекс успешно построен и сохранён в формате TSV.")


if __name__ == "__main__":
    main()
