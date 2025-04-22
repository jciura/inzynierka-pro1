
# Klasa do przetwarzania stringów
class String:
    # Odwraca stringa
    @staticmethod
    def reverse_string(text: str) -> str:
        return text[::-1]

    # Sprawdza czy tekst jest palindromem
    @staticmethod
    def is_palindrome(s: str) -> bool:
        return s == ''.join(reversed(s))

    # Liczy słowa w tekście
    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split())

    # Zamienia pierwszą literę każdego słowa na wielką
    @staticmethod
    def capitalize_words(text: str) -> str:
        return text.title()

    # Liczy całkowitą liczbę słów
    @staticmethod
    def get_total_words(words: list) -> int:
        return len(words)

    # Liczy liczbę unikalnych słów
    @staticmethod
    def get_unique_words(words: list) -> int:
        return len(set(words))

    # Znajduje najdłuższe słowo
    @staticmethod
    def get_longest_word(words: list) -> str:
        if not words:
            return ''
        return max(words, key=len)

    # Znajduje najkrótsze słowo
    @staticmethod
    def get_shortest_word(words: list) -> str:
        if not words:
            return ''
        return min(words, key=len)



