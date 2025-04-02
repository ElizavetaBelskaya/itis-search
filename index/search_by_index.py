import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def load_inverted_index_tsv(tsv_file):
    index = {}
    with open(tsv_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                term, files_str = line.split("\t")
            except ValueError:
                continue
            file_ids = set(map(int, files_str.split()))
            index[term] = file_ids
    return index

def make_supported_query(query):
    """
    Разбивает запрос на токены.
    Поддерживаются: скобки, операторы (AND, OR, NOT) и отдельные термины.
    Операторы приводятся к верхнему регистру, термины – к леммам в нижнем регистре.
    """
    token_pattern = r'\(|\)|AND|OR|NOT|[^\s\(\)]+'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)

    processed = []
    for token in tokens:
        if token.upper() in {"AND", "OR", "NOT"}:
            processed.append(token.upper())
        else:
            lemma = lemmatizer.lemmatize(token.lower())
            processed.append(lemma)
    return processed



def convert_to_postfix(tokens):
    """
    Преобразует список токенов в обратную польскую нотацию (ОПН)
    с помощью алгоритма сортировочной станции.
    Приоритет операторов: NOT > AND > OR.
    """
    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    output = []
    op_stack = []

    for token in tokens:
        if token == "(":
            op_stack.append(token)
        elif token == ")":
            while op_stack and op_stack[-1] != "(":
                output.append(op_stack.pop())
            if op_stack:  # удаляем "("
                op_stack.pop()
        elif token in precedence:
            while (op_stack and op_stack[-1] in precedence and
                   precedence[op_stack[-1]] >= precedence[token]):
                output.append(op_stack.pop())
            op_stack.append(token)
        else:
            # термины
            output.append(token)

    while op_stack:
        output.append(op_stack.pop())

    return output


def evaluate_postfix(postfix_tokens, inverted_index, all_file_ids):
    stack = []
    for token in postfix_tokens:
        if token == "AND":
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
        elif token == "OR":
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        elif token == "NOT":
            a = stack.pop()
            stack.append(all_file_ids - a)
        else:
            stack.append(inverted_index.get(token, set()))
    return stack.pop() if stack else set()


def boolean_search(query, inverted_index):
    """
    Выполняет булев поиск по строковому запросу
    """
    tokens = make_supported_query(query)
    postfix = convert_to_postfix(tokens)
    all_file_ids = set()
    for doc_ids in inverted_index.values():
        all_file_ids.update(doc_ids)

    result = evaluate_postfix(postfix, inverted_index, all_file_ids)
    return sorted(result)


def main():
    inverted_index = load_inverted_index_tsv("inverted_index.tsv")
    print("Введите запрос")
    print("Введите 'exit' для выхода.")

    while True:
        query = input("Ввод запроса: ").strip()
        if query.lower() == "exit":
            break

        try:
            result_file_ids = boolean_search(query, inverted_index)
            print(f"\nНайдено документов: {len(result_file_ids)}")
            for file_id in result_file_ids:
                print(f"{file_id}")
        except Exception as e:
            print("Ошибка при выполнении запроса:", e)
        print()


if __name__ == "__main__":
    main()
