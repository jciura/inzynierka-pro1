# Klasa do konta bankowego
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transactions = []

    #Wpłacanie środków na konto
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Kwota musi być dodatnia")
        self.balance += amount
        self.transactions.append(f"Wpłata: {amount} zł")

    # Wypłacanie środków z konta
    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Brak środków")
        self.balance -= amount
        self.transactions.append(f"Wypłata: {amount} zł")

    #Przelew na inne konto
    def transfer(self, amount, other_account):
        self.withdraw(amount)
        other_account.deposit(amount)
        self.transactions.append(f"Przelew do {other_account.owner}: {amount} zł")

    # Zwraca bieżący stan konta
    def get_balance(self):
        return self.balance

    # Zwraca historię transakcji
    def get_transaction_history(self):
        return self.transactions.copy()

    def __str__(self):
        return f"Konto: {self.owner}, saldo: {self.balance} zł"
