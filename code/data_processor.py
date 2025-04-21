import csv
import json


# Klasa do przetwarzania różnych typów danych
class DataProcessor:
    def __init__(self):
        self.data = []

    #  Wczytuje dane z pliku CSV
    def load_csv(self, file_path: str, delimiter: str = ',') -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            self.data = [row for row in reader]

    # Wczytuje dane z pliku JSON
    def load_json(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    # Zapisuje dane do pliku JSON
    def save_json(self, file_path: str, indent: int = 4) -> None:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, indent=indent)



    # Sortuje dane według klucza
    def sort_data(self, key: str, reverse: bool = False) -> None:
        self.data.sort(key=lambda x: x.get(key, None), reverse=reverse)


