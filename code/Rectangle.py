# Klasa Prostokąta
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    # Zwraca pole powierzchni prostokąta
    def area(self):
        return self.width * self.height

    # Zwraca obwód prostokąta
    def perimeter(self):
        return 2 * (self.width + self.height)

    # Sprawdza, czy prostokąt jest kwadratem
    def is_square(self):
        return self.width == self.height

    # Zmienia wymiary prostokąta
    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height

    # Skaluje wymiary prostokąta
    def scale(self, factor):
        self.width *= factor
        self.height *= factor

    # Zwraca długość przekątnej prostokąta
    def diagonal(self):
        return (self.width**2 + self.height**2) ** 0.5

    def __str__(self):
        return f"Prostokąt {self.width} x {self.height}, pole: {self.area()}, obwód: {self.perimeter()}"
