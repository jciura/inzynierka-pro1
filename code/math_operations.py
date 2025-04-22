import math
from typing import List, Union, Tuple


#Klasa zawierająca różne operacje matematyczne.
class MathOperations:



    # Generuje liczby pierwsze do limitu
    @staticmethod
    def prime_numbers(limit: int) -> List[int]:
        if limit < 2:
            return []
        primes = []
        for num in range(2, limit + 1):
            is_prime = True
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)

        return primes

    # Oblicza n-ty wyraz ciągu Fibonacciego
    @staticmethod
    def fibonacci(n: int) -> int:
        if n <= 0:
            raise ValueError("n musi być większe od 0")
        if n == 1 or n == 2:
            return 1
        a, b = 1, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b

    # Oblicza największy wspólny dzielnik
    @staticmethod
    def gcd(a: int, b: int) -> int:
        return math.gcd(a, b)

    # Oblicza najmniejszą wspólną wielokrotność
    @staticmethod
    def lcm(a: int, b: int) -> int:
        return abs(a * b) // MathOperations.gcd(a, b)

    # Oblicza pierwiastki równania kwadratowego ax^2 + bx + c = 0.
    @staticmethod
    def quadratic_roots(a: float, b: float, c: float) -> Tuple[Union[float, complex], Union[float, complex]]:
        if a == 0:
            raise ValueError("To nie jest równanie kwadratowe")
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            sqrt_delta = math.sqrt(-delta) * 1j
        else:
            sqrt_delta = math.sqrt(delta)
        x1 = (-b + sqrt_delta) / (2 * a)
        x2 = (-b - sqrt_delta) / (2 * a)
        return x1, x2



    # Sprawdza czy liczba jest doskonała.
    @staticmethod
    def is_perfect_number(n: int) -> bool:
        if n <= 1:
            return False
        divisors_sum = 1
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divisors_sum += i
                if i != n // i:
                    divisors_sum += n // i
        return divisors_sum == n