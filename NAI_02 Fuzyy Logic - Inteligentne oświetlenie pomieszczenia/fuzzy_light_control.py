"""
===================================================================
System Inteligentnego Oświetlenia - LOGIKA ROZMYTA
===================================================================

OPIS PROBLEMU:
--------------
System automatycznego sterowania oświetleniem wewnętrznym, który
dynamicznie dostosowuje jasność żarówki na podstawie dwóch parametrów:
1. Pora dnia (godzina 0-24)
2. Poziom światła naturalnego (0-100%)

Cel: Minimalizacja zużycia energii przy zapewnieniu odpowiedniego
     poziomu oświetlenia w pomieszczeniu.

Problem jest realizowany za pomocą logiki rozmytej (fuzzy logic),
która pozwala na płynne przejścia między stanami i naturalne
uwzględnianie niepewności pomiarów.

AUTORZY:
--------
Jan Skulimowski (s27144)
Kamil Littwitz (s26966)

PRZYGOTOWANIE ŚRODOWISKA:
-------------------------
1. Wymagany Python 3.7 lub nowszy

2. Instalacja wymaganych bibliotek:
   pip install numpy scikit-fuzzy

3. Uruchomienie programu:
   python fuzzy_light_control.py

===================================================================
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# --- SEKCJA 1: Konfiguracja systemu logiki rozmytej ---
def configure_fuzzy_system():
    """Tworzy i konfiguruje system logiki rozmytej"""
    time_of_day_range = np.arange(0, 25, 1)
    natural_light_range = np.arange(0, 101, 1)
    bulb_brightness_range = np.arange(0, 101, 1)

    time_of_day = ctrl.Antecedent(time_of_day_range, 'time_of_day')
    natural_light = ctrl.Antecedent(natural_light_range, 'natural_light')
    bulb_brightness = ctrl.Consequent(bulb_brightness_range, 'bulb_brightness')

    # Metoda defuzzyfikacji 'som' dla wartości skrajnych (0% i 100%)
    bulb_brightness.defuzzify_method = 'som'

    # Definicje wejść - pory dnia
    time_of_day['night'] = fuzz.trapmf(time_of_day.universe, [0, 0, 4, 6])
    time_of_day['morning'] = fuzz.trapmf(time_of_day.universe, [5, 7, 9, 11])
    time_of_day['day'] = fuzz.trapmf(time_of_day.universe, [10, 12, 16, 18])
    time_of_day['evening'] = fuzz.trapmf(time_of_day.universe, [17, 19, 24, 24])

    # Definicje wejść - poziom światła naturalnego
    natural_light['very_dark'] = fuzz.trapmf(natural_light.universe, [0, 0, 10, 20])
    natural_light['dark'] = fuzz.trapmf(natural_light.universe, [15, 25, 35, 45])
    natural_light['moderate'] = fuzz.trapmf(natural_light.universe, [40, 50, 60, 70])
    natural_light['bright'] = fuzz.trapmf(natural_light.universe, [65, 80, 100, 100])

    # Definicje wyjścia - jasność żarówki
    bulb_brightness['off'] = fuzz.trimf(bulb_brightness.universe, [0, 0, 3])
    bulb_brightness['minimal'] = fuzz.trimf(bulb_brightness.universe, [2, 8, 14])
    bulb_brightness['very_low'] = fuzz.trimf(bulb_brightness.universe, [12, 20, 28])
    bulb_brightness['low'] = fuzz.trimf(bulb_brightness.universe, [25, 35, 45])
    bulb_brightness['medium'] = fuzz.trimf(bulb_brightness.universe, [40, 50, 60])
    bulb_brightness['high'] = fuzz.trimf(bulb_brightness.universe, [55, 70, 85])
    bulb_brightness['full'] = fuzz.trimf(bulb_brightness.universe, [95, 100, 100])

    # Reguły sterowania - 16 kombinacji
    rules = [
        # NOC - zawsze pełne oświetlenie
        ctrl.Rule(time_of_day['night'] & natural_light['very_dark'], bulb_brightness['full']),
        ctrl.Rule(time_of_day['night'] & natural_light['dark'], bulb_brightness['full']),
        ctrl.Rule(time_of_day['night'] & natural_light['moderate'], bulb_brightness['full']),
        ctrl.Rule(time_of_day['night'] & natural_light['bright'], bulb_brightness['full']),

        # PORANEK - stopniowe zmniejszanie
        ctrl.Rule(time_of_day['morning'] & natural_light['very_dark'], bulb_brightness['high']),
        ctrl.Rule(time_of_day['morning'] & natural_light['dark'], bulb_brightness['medium']),
        ctrl.Rule(time_of_day['morning'] & natural_light['moderate'], bulb_brightness['low']),
        ctrl.Rule(time_of_day['morning'] & natural_light['bright'], bulb_brightness['minimal']),

        # DZIEŃ - minimalne oświetlenie przy zachmurzeniu
        ctrl.Rule(time_of_day['day'] & natural_light['very_dark'], bulb_brightness['medium']),
        ctrl.Rule(time_of_day['day'] & natural_light['dark'], bulb_brightness['very_low']),
        ctrl.Rule(time_of_day['day'] & natural_light['moderate'], bulb_brightness['minimal']),
        ctrl.Rule(time_of_day['day'] & natural_light['bright'], bulb_brightness['off']),

        # WIECZÓR - wzrost w miarę zmierzchu
        ctrl.Rule(time_of_day['evening'] & natural_light['very_dark'], bulb_brightness['full']),
        ctrl.Rule(time_of_day['evening'] & natural_light['dark'], bulb_brightness['high']),
        ctrl.Rule(time_of_day['evening'] & natural_light['moderate'], bulb_brightness['medium']),
        ctrl.Rule(time_of_day['evening'] & natural_light['bright'], bulb_brightness['low']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)


# --- SEKCJA 2: Funkcja obliczania jasności ---
def calculate_brightness(controller, time, light):
    """Oblicza jasność żarówki dla danych wartości wejściowych"""
    try:
        controller.input['time_of_day'] = time
        controller.input['natural_light'] = light
        controller.compute()
        return controller.output['bulb_brightness']
    except (ValueError, KeyError):
        return 0.0


# --- SEKCJA 3: Symulacja 24h ---
def simulate_24h(controller):
    """Symulacja pełnego cyklu dobowego"""
    print("\n" + "=" * 60)
    print("--- Symulacja 24-godzinnego cyklu ---")
    print("=" * 60)

    for hour in range(24):
        for minute in [0, 30]:
            time = hour + minute / 60.0

            # Funkcja światła naturalnego (sinusoida)
            light_level = 100 * np.sin((time - 7) * np.pi / 13)
            light = np.clip(light_level, 0, 100)

            brightness = calculate_brightness(controller, time, light)

            print(f"Czas: {hour:02d}:{minute:02d} | "
                  f"Światło naturalne: {light:5.1f}% | "
                  f"==> Jasność żarówki: {brightness:5.1f}%")


# --- SEKCJA 4: Tryb interaktywny ---
def interactive_mode():
    """Tryb interaktywny - użytkownik wprowadza własne wartości"""
    controller = configure_fuzzy_system()

    print("\n" + "=" * 60)
    print("   INTERAKTYWNY SYSTEM INTELIGENTNEGO OŚWIETLENIA")
    print("=" * 60)
    print("\nWprowadź wartości wejściowe (lub 'q' aby zakończyć):\n")

    while True:
        try:
            # Pobierz godzinę
            time_input = input("Godzina (0-24): ").strip()
            if time_input.lower() == 'q':
                print("\nZakończono tryb interaktywny.")
                break

            time_of_day = float(time_input)
            if not (0 <= time_of_day <= 24):
                print("❌ Godzina musi być w zakresie 0-24!\n")
                continue

            # Pobierz poziom światła
            light_input = input("Światło naturalne (0-100%): ").strip()
            if light_input.lower() == 'q':
                print("\nZakończono tryb interaktywny.")
                break

            natural_light = float(light_input)
            if not (0 <= natural_light <= 100):
                print("❌ Światło naturalne musi być w zakresie 0-100%!\n")
                continue

            # Oblicz wynik
            brightness = calculate_brightness(controller, time_of_day, natural_light)

            # Wyświetl wynik
            godzina = int(time_of_day)
            minuta = int((time_of_day * 60) % 60)
            print(f"\n{'─' * 60}")
            print(f"⏰ Czas: {godzina:02d}:{minuta:02d}")
            print(f"☀️  Światło naturalne: {natural_light:.1f}%")
            print(f"💡 JASNOŚĆ ŻARÓWKI: {brightness:.2f}%")
            print(f"{'─' * 60}\n")

        except ValueError:
            print("❌ Wprowadź poprawną liczbę!\n")
        except KeyboardInterrupt:
            print("\n\nZakończono tryb interaktywny.")
            break
        except Exception as e:
            print(f"❌ Błąd: {e}\n")


# --- SEKCJA 5: Test warunków brzegowych ---
def test_edge_cases():
    """Testuje skrajne i typowe przypadki użycia"""
    print("\n" + "=" * 60)
    print("--- Test warunków brzegowych ---")
    print("=" * 60)

    controller = configure_fuzzy_system()

    test_cases = [
        (14, 100, "IDEALNIE: Słoneczne popołudnie"),
        (12, 50, "Dzień, umiarkowane światło"),
        (2, 0, "NAJGORZEJ: Głęboka noc, brak światła"),
        (3, 5, "Noc, minimalne światło"),
        (8, 15, "Wczesny poranek, przed wschodem"),
        (8, 85, "Poranek, jasno"),
        (20, 10, "Późny wieczór, ciemno"),
        (18, 50, "Wieczór, zmierzch"),
    ]

    for time, light, description in test_cases:
        brightness = calculate_brightness(controller, time, light)
        print(f"{description:40s} | Jasność: {brightness:6.2f}%")

    print("=" * 60)


# --- SEKCJA 6: Menu główne ---
def main():
    print("\n" + "=" * 60)
    print("   SYSTEM INTELIGENTNEGO OŚWIETLENIA")
    print("=" * 60)
    print("\nWybierz tryb działania:")
    print("  1. Tryb interaktywny (wprowadzanie własnych wartości)")
    print("  2. Symulacja 24-godzinnego cyklu")
    print("  3. Test warunków brzegowych")

    choice = input("\nWybór (1-3): ").strip()

    if choice == '1':
        interactive_mode()

    elif choice == '2':
        controller = configure_fuzzy_system()
        simulate_24h(controller)

    elif choice == '3':
        test_edge_cases()

    else:
        print("❌ Nieprawidłowy wybór!")

if __name__ == '__main__':
    main()