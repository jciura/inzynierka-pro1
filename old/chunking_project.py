import os
import json
import re


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


def indentation(line):
    return len(line) - len(line.lstrip())


def extract_name(line, key):
    match = re.search(rf'{key}\s+(\w+)', line)
    if match:
        return match.group(1)
    else:
        return "unknown"


def chunking(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")
    chunks = []
    filename = os.path.basename(file_path)
    patterns = {
        'module_docstring': r'^""".*?"""$|^\'\'\'.*?\'\'\'$',
        'import_section': r'^(import|from)\s+.*$',
        'class_definition': r'^class\s+\w+.*:$',
        'function_definition': r'^def\s+\w+.*:$',
        'method_definition': r'^\s+def\s+\w+.*:$',
        'constant_definition': r'^[A-Z][A-Z0-9_]*\s*=.*$',
        'variable_definition': r'^[a-z][a-zA-Z0-9_]*\s*=.*$',
        'comment_block': r'^#.*$',
        'decorator': r'^@\w+.*$',
    }
    current_chunk = ''
    current_chunk_type = None
    current_chunk_name = None
    current_class = None
    chunk_start_line = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        if i == 0 and (stripped_line.startswith('"""') or stripped_line.startswith("'''")):
            finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
            current_chunk_type = "module_docstring"
            current_chunk_name = "module_docstring"
            chunk_start_line = i
            current_chunk = line + '\n'
            
            quote_type = '"""' if stripped_line.startswith('"""') else "'''"
            if not (stripped_line.endswith(quote_type) and len(stripped_line) > 3):
                i += 1
                while i < len(lines):
                    current_chunk += lines[i] + '\n'
                    if quote_type in lines[i]:
                        break
                    i += 1
        elif re.match(patterns['import_section'], stripped_line):
            if current_chunk_type != "imports":
                finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
                current_chunk_type = "imports"
                current_chunk_name = "imports"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'
            if stripped_line.endswith('\\') or ('(' in stripped_line and ')' not in stripped_line):
                i += 1
                while i < len(lines):
                    current_chunk += lines[i] + '\n'
                    if ')' in lines[i] or not lines[i].strip().endswith('\\'):
                        break
                    i += 1
        elif re.match(patterns['class_definition'], stripped_line):
            finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
            current_chunk_type = "class"
            current_chunk_name = extract_name(stripped_line, "class")
            current_class = current_chunk_name
            chunk_start_line = i
            current_chunk = line + '\n'
        elif re.match(patterns['method_definition'], stripped_line) and current_class:
            finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
            current_chunk_type = "method"
            current_chunk_name = extract_name(stripped_line, "def")
            chunk_start_line = i
            current_chunk = f"class {current_class}:\n" + line + '\n'
        elif re.match(patterns['function_definition'], stripped_line):
            finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
            current_chunk_type = "function"
            current_chunk_name = extract_name(stripped_line, "def")
            current_class = None
            chunk_start_line = i
            current_chunk = line + '\n'
        elif re.match(patterns['constant_definition'], stripped_line):
            if current_chunk_type != "constants":
                finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
                current_chunk_type = "constants"
                current_chunk_name = "module_constants"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'
        elif re.match(patterns['variable_definition'], stripped_line) and not current_class:
            if current_chunk_type != "module_variables":
                finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
                current_chunk_type = "module_variables"
                current_chunk_name = "module_variables"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'
        elif re.match(patterns['decorator'], stripped_line):
            current_chunk += line + '\n'
        elif re.match(patterns['comment_block'], stripped_line):
            if current_chunk_type != "comments":
                finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
                current_chunk_type = "comments"
                current_chunk_name = "comments"
                chunk_start_line = i
                current_chunk = ''
            current_chunk += line + '\n'
        else:
            if stripped_line or current_chunk_type in ["class", "method", "function"]:
                current_chunk += line + '\n'
            if current_class and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and indentation(lines[i + 1]) == 0:
                    if any(next_line.startswith(k) for k in ('class ', 'def ', 'import ', 'from ')):
                        current_class = None

        i += 1
    finalize(chunks, filename, current_chunk_type, current_class, current_chunk_name, current_chunk, chunk_start_line)
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
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    chunks = chunking(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    return all_chunks


#
# if __name__ == "__main__":
#
#     folder_path = "../projects/rich"
#     output_path = "../data/project_chunks.json"
#     chunks = get_all_chunks(folder_path, output_path)
