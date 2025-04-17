1. https://ollama.com
2. https://drive.google.com/drive/folders/1x55SRmR5dxLMieFkUaBhBjHyDv0sI128?usp=sharing - folder z modelem
3. Folder models skopiować do /.ollama
4. w terminalu: ollama run codellama:7b-instruct - uruchomienie modelu
5. Przykład zapytania w http://127.0.0.1:8000/docs:
   1. "question": "Ile to 2+2?",
   2. "context": ""
6. Przykład zapytania o kod
   1. file_path: examply.py
   2. question: Co robi metoda add?
   3. { "answer": "Metoda `add` jest funkcją, która dodaje dwa argumenty i zwraca ich sumę."}
