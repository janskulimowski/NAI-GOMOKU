"""
======================================================
SYSTEM ROZPOZNAWANIA FLAG NA EKRANIE (COMPUTER VISION)
======================================================

OPIS PROBLEMU:
Program rozwiązuje problem automatycznej detekcji określonych obiektów graficznych
(w tym przypadku flag państwowych: Polski, Rosji i Ukrainy) na ekranie komputera
w czasie rzeczywistym. Wykorzystuje technikę "Template Matching" (dopasowywanie wzorców),
aby znaleźć lokalizację obrazu wzorcowego na zrzucie ekranu.

INSTRUKCJA UŻYCIA:
1. Wymagania: Zainstaluj biblioteki poleceniem:
   pip install opencv-python numpy pyautogui pillow

2. Konfiguracja:
   W katalogu z tym skryptem muszą znajdować się pliki wzorcowe (wycinki flag):
   - 'pl.png' (Flaga Polski)
   - 'ru.png' (Flaga Rosji)
   - 'ua.png' (Flaga Ukrainy)
   UWAGA: Wzorce muszą być w formacie .png lub .jpg i powinny być wycięte
   bez tła (lub z tłem, które występuje na ekranie).

3. Uruchomienie:
   python nazwa_pliku.py

AUTORZY:
Kamil Littwitz
Jan Skulimowski
"""

import cv2
import numpy as np
import pyautogui
import time

class FlagDetector:
    def __init__(self, templates):
        """
        Inicjalizacja detektora z listą ścieżek do plików wzorców.
        templates: słownik { 'nazwa_kraju': 'sciezka_do_pliku.png' }
        """
        self.templates = {}
        # Wczytanie obrazów wzorcowych do pamięci
        for name, path in templates.items():
            img = cv2.imread(path)
            if img is None:
                print(f"Błąd: Nie można wczytać pliku {path}")
            else:
                self.templates[name] = img

    def find_flags_on_screen(self, threshold=0.8):
        """
        Robi zrzut ekranu i szuka flag.
        threshold: próg pewności (0.8 = 80% zgodności)
        """
        # 1. Wykonaj zrzut ekranu
        screenshot = pyautogui.screenshot()
        # Konwersja z RGB (Pillow) na BGR (OpenCV)
        screen_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        detected_flags = []

        # 2. Przeszukaj ekran pod kątem każdego wzorca
        for name, template in self.templates.items():
            # Pobierz wymiary wzorca
            h, w = template.shape[:2]

            # Zastosuj Template Matching
            result = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)

            # Znajdź lokalizacje, gdzie dopasowanie jest wyższe niż próg (threshold)
            loc = np.where(result >= threshold)

            # Iteruj po znalezionych punktach (zip łączy współrzędne y i x)
            # Używamy prostego mechanizmu, by nie wykrywać tej samej flagi wielokrotnie
            points = list(zip(*loc[::-1]))

            if points:
                # Bierzemy najlepsze dopasowanie (min_max_loc) dla uproszczenia przykładu,
                # lub listę punktów jeśli flag jest wiele.
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val >= threshold:
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)

                    detected_flags.append({
                        'kraj': name,
                        'pewność': round(max_val * 100, 2),
                        'pozycja': {
                            'x': top_left[0],
                            'y': top_left[1],
                            'szerokosc': w,
                            'wysokosc': h
                        }
                    })

        return detected_flags

# --- UŻYCIE SYSTEMU ---

if __name__ == "__main__":
    # Definicja wzorców
    templates_config = {
        'Polska': 'pl.png',
        'Rosja': 'ru.png',
        'Ukraina': 'ua.png'
    }

    detector = FlagDetector(templates_config)

    print("System uruchomiony. Szukam flag... (Ctrl+C aby przerwać)")

    try:
        while True:
            start_time = time.time()
            found = detector.find_flags_on_screen(threshold=0.9) # Wysoki próg dla precyzji

            if found:
                print(f"--- Znaleziono: {len(found)} flag(i) ---")
                for item in found:
                    print(f"Flaga: {item['kraj'].upper()}")
                    print(f"   Pozycja: X={item['pozycja']['x']}, Y={item['pozycja']['y']}")
                    print(f"   Pewność: {item['pewność']}%")
                print()
            # Odczekaj chwilę, żeby nie obciążać procesora w 100%
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nZakończono działanie programu.")