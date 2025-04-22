from datetime import datetime, timedelta, date
import calendar


# Klasa związana z datą i czasem
class DateTime:
    # Sprawdza czy rok jest przestępny
    @staticmethod
    def is_leap_year(year: int) -> bool:
        return calendar.isleap(year)

    # Oblicza liczbę dni między dwiema datami
    @staticmethod
    def days_between(date1: date, date2: date) -> int:
        return abs((date2 - date1).days)


    # Zwraca nazwę dnia tygodnia
    @staticmethod
    def get_weekday(target_date: date) -> str:
        days = ['Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela']
        return days[target_date.weekday()]

    # Formatuje datę i czas
    @staticmethod
    def format_datetime(dt: datetime, format_string: str = '%Y-%m-%d %H:%M:%S') -> str:
        return dt.strftime(format_string)

    # Zwraca kwartał roku
    @staticmethod
    def get_quarter(date_obj: date) -> int:
        return (date_obj.month - 1) // 3 + 1

    # Zwraca liczbę dni w miesiącu
    @staticmethod
    def get_days_in_month(year: int, month: int) -> int:
        return calendar.monthrange(year, month)[1]


    # Sprawdza czy data wypada w weekend
    @staticmethod
    def is_weekend(date_obj: date) -> bool:
        return date_obj.weekday() >= 5

    # Oblicza czas pozostały do wydarzenia
    @staticmethod
    def time_to_event(event_date: datetime) -> str:
        now = datetime.now()
        if event_date < now:
            return "Wydarzenie już się odbyło"

        diff = event_date - now
        total_seconds = int(diff.total_seconds())

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        parts = []
        if days > 0:
            parts.append(f"{days} dni")
        if hours > 0:
            parts.append(f"{hours} godzin")
        if minutes > 0 or not parts:
            parts.append(f"{minutes} minut")

        return ", ".join(parts)