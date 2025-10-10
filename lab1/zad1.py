from datetime import date
import math


def a():
    name = input("Podaj imie")
    year_of_birth = input("Podaj rok urodzenia ")
    month_of_birth = input("Podaj miesiąc urodzenia ")
    day_of_birth = input("Podaj dzień urodzenia ")
    birthdate = date(int(year_of_birth), int(month_of_birth), int(day_of_birth))
    num_of_days = calculate_days(birthdate)

    twoja_fizyczna_fala = fizyczna_fala(num_of_days)
    twoja_emocjonalna_fala = emocjonalna_fala(num_of_days)
    twoja_intelektualna_fala = intelektualna_fala(num_of_days)

    srednia_fala = (
        twoja_emocjonalna_fala + twoja_emocjonalna_fala + twoja_intelektualna_fala
    ) / 3

    print(f"Witaj {name}, dzisiaj jest twój {num_of_days} dzień życia")
    print(f"Twoja fizyczna_fala {twoja_fizyczna_fala}")
    print(f"Twoja emocjonalna_fala {twoja_emocjonalna_fala}")
    print(f"Twoja intelektualna_fala {twoja_intelektualna_fala}")

    if srednia_fala < -0.5:
        print("może kiedyś bedzie lepiej")
        if moze_jutro_bedzie_lepiej(num_of_days):
            print("Jutro bedzie lepiej")
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAA")

    elif srednia_fala > 0.5:
        print("Jest git")
    else:
        print("Jest mid")


def calculate_days(date):
    today = date.today()
    return (today - date).days


def moze_jutro_bedzie_lepiej(days):
    next_day = days + 1
    srednia_fala = (
        fizyczna_fala(next_day)
        + emocjonalna_fala(next_day)
        + intelektualna_fala(next_day) / 3
    )
    if srednia_fala > 0.5:
        return True
    return False


def fizyczna_fala(days):
    return math.sin(((2 * math.pi) / 23) * days)


def emocjonalna_fala(days):
    return math.sin(((2 * math.pi) / 28) * days)


def intelektualna_fala(days):
    return math.sin(((2 * math.pi) / 33) * days)


a()


# czas poświęcony +- 40 min
