import re
import os
import json


def finalize(chunks, filename, chunk_type, class_name, chunk_name, chunk_content, start_line):
    if chunk_content.strip():
        chunks.append({
            "file": filename,
            "type": chunk_type or "unknown",
            "class": class_name if chunk_type == "method" else None,
            "method": chunk_name if chunk_type in ["method", "function"] else None,
            "name": chunk_name,
            "content": chunk_content.strip(),
            "start_line": start_line
        })


def get_indentation_level(line):
    return len(line) - len(line.lstrip())


get_indentation_level("class MyClass:")  # 0
print(get_indentation_level("    def add(self):"))  # 4
print(get_indentation_level("        return 42"))  # 8


def extract_name_from_line(line, pattern_type):
    if pattern_type == "class":
        # class MyClass(BaseClass) -> "MyClass"
        match = re.search(r'class\s+(\w+)', line)
    elif pattern_type == "def":
        match = re.search(r'def\s+(\w+)', line)
    elif pattern_type == "variable":
        match = re.search(r'^([A-Za-z_]\w*)\s*=', line.strip())
    else:
        return "unknown"

    return match.group(1) if match else "unknown"


print(extract_name_from_line("    def add(self):", "def"))
print(extract_name_from_line("age = 42", "variable"))  # 8
print(extract_name_from_line("class MyClass:", "class"))


def is_method_definition(line, class_context=None):

    stripped = line.strip()
    if not stripped.startswith('def '):
        return False

    indentation = get_indentation_level(line)
    return indentation > 0 and class_context is not None


print(is_method_definition("def add():", None))
print(is_method_definition("  def method(self):", "MyClass"))


def chunking(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")
    chunks = []
    filename = os.path.basename(file_path)

    i = 0
    current_chunk = ''
    current_chunk_type = None
    current_chunk_name = None
    current_class_context = None
    chunk_start_line = 0
    chunk_indentation_level = 0

    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        indentation = get_indentation_level(line)

        # pomin puste liinie (ale nie w srodku funkcji/metody)
        if not stripped_line and current_chunk_type not in ["class", "method", "function"]:
            i += 1
            continue

        # modul docstring na poczatku
        if i == 0 and (stripped_line.startswith('"""') or stripped_line.startswith("'''")):
            finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                     chunk_start_line)

            current_chunk_type = "module_docstring"
            current_chunk_name = "module_docstring"
            chunk_start_line = i

            # Obsluga wieloliniowego docstringa
            quote_type = '"""' if stripped_line.startswith('"""') else "'''"
            current_chunk = line + '\n'

            if not (stripped_line.endswith(quote_type) and len(stripped_line) > 3):
                i += 1
                while i < len(lines) and quote_type not in lines[i]:
                    current_chunk += lines[i] + '\n'
                    i += 1
                if i < len(lines):
                    current_chunk += lines[i] + '\n'

        # importy
        elif re.match(r'^(import|from)\s+', stripped_line):
            if current_chunk_type != "imports":
                finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                         chunk_start_line)
                current_chunk_type = "imports"
                current_chunk_name = "imports"
                chunk_start_line = i
                current_chunk = ''

            current_chunk += line + '\n'

            while (stripped_line.endswith('\\') or ('(' in stripped_line and ')' not in current_chunk)) and i + 1 < len(
                    lines):
                i += 1
                current_chunk += lines[i] + '\n'
                stripped_line = lines[i].strip()

        elif re.match(r'^class\s+\w+', stripped_line):
            finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                     chunk_start_line)
            current_chunk_type = "class"
            current_chunk_name = extract_name_from_line(stripped_line, "class")
            current_class_context = current_chunk_name
            class_indentation_level = indentation
            chunk_start_line = i
            current_chunk = line + '\n'

        # Definicja metody (wcieta w klasie)
        elif is_method_definition(line, current_class_context):
            finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                     chunk_start_line)
            current_chunk_type = "method"
            current_chunk_name = extract_name_from_line(stripped_line, "def")
            chunk_start_line = i
            current_chunk = line + '\n'


        elif re.match(r'^def\s+\w+', stripped_line):
            finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                     chunk_start_line)
            current_chunk_type = "function"
            current_chunk_name = extract_name_from_line(stripped_line, "def")
            current_class_context = None
            chunk_start_line = i
            current_chunk = line + '\n'

        # Stale modulu (UPPERCASE)
        elif re.match(r'^[A-Z][A-Z0-9_]*\s*=', stripped_line):
            if current_chunk_type != "constants":
                finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                         chunk_start_line)
                current_chunk_type = "constants"
                current_chunk_name = "module_constants"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'

        # Zmienne modulu
        elif re.match(r'^[a-z_]\w*\s*=', stripped_line) and not current_class_context:
            if current_chunk_type != "module_variables":
                finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                         chunk_start_line)
                current_chunk_type = "module_variables"
                current_chunk_name = "module_variables"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'

        # Komentarze
        elif stripped_line.startswith('#'):
            if current_chunk_type != "comments":
                finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
                         chunk_start_line)
                current_chunk_type = "comments"
                current_chunk_name = "comments"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'

        # Dekorator
        elif stripped_line.startswith('@'):
            current_chunk += line + '\n'

        else:
            # sprawdzam czy opuszczam klase (powrot do poziomu 0 wciecia)
            if (current_class_context and indentation <= class_indentation_level and stripped_line
                and not stripped_line.startswith('@', '#')) and current_chunk_type not in ["method", "function"]:
                current_class_context = None

            # dodaj linie do chunka jesli chunk jest aktywny
            if current_chunk_type:
                current_chunk += line + '\n'

        i += 1

    finalize(chunks, filename, current_chunk_type, current_class_context, current_chunk_name, current_chunk,
             chunk_start_line)
    chunks.append({
        "file": filename,
        "type": "full_module",
        "class": None,
        "method": None,
        "name": "full_module",
        "content": text,
        "start_line": 0
    })
    return chunks


def get_all_chunks(folder_path, output_file):
    all_chunks = []
    for root, dirs, files in os.walk(folder_path):
        print(f"Folder path {folder_path}")
        for file in files:
            print(f"Filename: {file}")
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    chunks = chunking(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8-sig") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    return all_chunks


if __name__ == "__main__":
    test_file = "projects/rich/rich/console.py"
    if os.path.exists(test_file):
        chunks = chunking(test_file)
        print(f"Total chunks: {len(chunks)}")

        types = {}
        for chunk in chunks:
            chunk_type = chunk['type']
            types[chunk_type] = types.get(chunk_type, 0) + 1

        print("Chunk types:")
        for chunk_type, count in types.items():
            print(f" {chunk_type}: {count}")

        method_chunks = [c for c in chunks if c['type'] == 'method']
        print(f"\nFirst 5 methods found:")
        for i, chunk in enumerate(method_chunks[:5]):
            print(f"  {i + 1}. Class: {chunk['class']}, Method: {chunk['method']}")

    folder_path = "../projects/rich"
    output_path = "../data/project_chunks_fixed.json"
    chunks = get_all_chunks(folder_path, output_path)
