import json
import os


def end_of_function(lines, start_idx):
    line = lines[start_idx]
    ind = 0
    for c in line:
        if c == ' ':
            ind += 1
        else:
            break
    i = start_idx + 1
    while i < len(lines):
        l = lines[i]
        if l.strip() == "":
            i += 1
            continue
        curr_indent = 0
        for c in l:
            if c == ' ':
                curr_indent += 1
            else:
                break
        if curr_indent <= ind:
            return i
        i += 1
    return len(lines)


def get_chunks(file):
    with open(file, "r") as f:
        text = f.read()
    lines = text.split("\n")
    chunks = []
    current_class = None
    filename = os.path.basename(file)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("class "):
            class_name = ""
            j = line.find("class ") + 6
            while j < len(line) and (line[j].isalnum() or line[j] == "_"):
                class_name += line[j]
                j += 1
            current_class = class_name
            i += 1
            continue
        if line.strip().startswith("def "):
            fname = ""
            j = line.find("def ") + 4
            while j < len(line) and line[j] != "(":
                fname += line[j]
                j += 1
            fname = fname.strip()
            ind = 0
            for c in line:
                if c == ' ':
                    ind += 1
                else:
                    break
            end = end_of_function(lines, i)
            code = ""
            for k in range(i, end):
                code += lines[k] + "\n"
            code = code.rstrip() + "\n"
            if ind > 0 and current_class is not None:
                chunk = {
                    "file": filename,
                    "type": "method",
                    "class": current_class,
                    "method": fname,
                    "content": "class " + current_class + ":\n" + code
                }
            else:
                current_class = None
                chunk = {
                    "file": filename,
                    "type": "function",
                    "class": None,
                    "method": fname,
                    "content": code
                }
            chunks.append(chunk)
            i = end
            continue
        i += 1
    return chunks


def get_all_chunks(folder_path, output):
    files = os.listdir(folder_path)
    all_chunks = []
    for file in files:
        if file.endswith(".py"):
            chunks = get_chunks(os.path.join(folder_path, file))
            for c in chunks:
                all_chunks.append(c)
    with open(output, "w") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    return output


