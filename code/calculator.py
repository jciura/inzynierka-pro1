import math

# Klasa kalkulatora
class Calculator:
    def __init__(self):
        self.memory = 0
        self.history = []
    # dodawanie
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    # odejmowanie
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    # Mnożenie
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    # Dzielenie
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Nie można dzielić przez zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result


    # Potęgowanie
    def power(self, base, exponent):
        result = base ** exponent
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result


    #Pierwiastkowanie
    def sqrt(self, n):
        if n < 0:
            raise ValueError("Nie można obliczyć pierwiastka z liczby ujemnej")
        result = math.sqrt(n)
        self.history.append(f"√{n} = {result}")
        return result



    #Logarytm
    def logarithm(self, n, base=math.e):
        if n <= 0:
            raise ValueError("Logarytm tylko dla liczb dodatnich")
        result = math.log(n, base)
        self.history.append(f"log_{base}({n}) = {result}")
        return result


    # Obliczanie silni
    def factorial(self, n):
        if n < 0:
            raise ValueError("Silnia tylko dla liczb >= 0")
        result = math.factorial(n)
        self.history.append(f"{n}! = {result}")
        return result

    # Zapisuje wartość w pamięci
    def memory_store(self, value):
        self.memory = value

    # Odczytuje wartość z pamięci
    def memory_recall(self):
        return self.memory


    # Czyszczenie pamięci
    def memory_clear(self):
        self.memory = 0

    # Zwraca historie operacji
    def get_history(self):
        return self.history.copy()

    # Czyszczenie historii
    def clear_history(self):
        self.history.clear()
