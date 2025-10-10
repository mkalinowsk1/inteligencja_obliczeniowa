from datetime import date
import math


def a():
    name = input("Podaj imię: ")
    year_of_birth = int(input("Podaj rok urodzenia: "))
    month_of_birth = int(input("Podaj miesiąc urodzenia: "))
    day_of_birth = int(input("Podaj dzień urodzenia: "))

    birthdate = date(year_of_birth, month_of_birth, day_of_birth)
    num_of_days = calculate_days(birthdate)

    twoja_fizyczna_fala = fizyczna_fala(num_of_days)
    twoja_emocjonalna_fala = emocjonalna_fala(num_of_days)
    twoja_intelektualna_fala = intelektualna_fala(num_of_days)

    print(f"\nWitaj {name}, dzisiaj jest twój {num_of_days} dzień życia.")
    print(f"Twoja fala fizyczna: {twoja_fizyczna_fala:.2f}")
    print(f"Twoja fala emocjonalna: {twoja_emocjonalna_fala:.2f}")
    print(f"Twoja fala intelektualna: {twoja_intelektualna_fala:.2f}")

    # Liczymy średnią wszystkich fal
    srednia_fala = (
        twoja_fizyczna_fala + twoja_emocjonalna_fala + twoja_intelektualna_fala
    ) / 3

    if srednia_fala < -0.5:
        print("Może kiedyś będzie lepiej...")
        if moze_jutro_bedzie_lepiej(num_of_days):
            print("Jutro będzie lepiej!")
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAA :(")

    elif srednia_fala > 0.5:
        print("Jest git! :)")


def calculate_days(birthdate):
    today = date.today()
    return (today - birthdate).days


def moze_jutro_bedzie_lepiej(days):
    next_day = days + 1
    srednia_jutro = (
        fizyczna_fala(next_day)
        + emocjonalna_fala(next_day)
        + intelektualna_fala(next_day)
    ) / 3
    return srednia_jutro > 0.5


def fizyczna_fala(days):
    return math.sin(((2 * math.pi) / 23) * days)


def emocjonalna_fala(days):
    return math.sin(((2 * math.pi) / 28) * days)


def intelektualna_fala(days):
    return math.sin(((2 * math.pi) / 33) * days)


a()
