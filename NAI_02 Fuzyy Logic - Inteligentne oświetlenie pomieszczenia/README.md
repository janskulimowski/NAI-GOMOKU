# System Inteligentnego Oświetlenia 💡

System automatycznego sterowania oświetleniem wykorzystujący logikę rozmytą (Fuzzy Logic) do optymalizacji zużycia energii.

## 📋 Opis Projektu

System dynamicznie dostosowuje jasność żarówki w pomieszczeniu na podstawie:
- **Pory dnia** (0-24h)
- **Poziomu światła naturalnego** (0-100%)

### Cel
Minimalizacja zużycia energii elektrycznej przy jednoczesnym zapewnieniu odpowiedniego poziomu oświetlenia w pomieszczeniu.

### Zastosowana Technologia
Projekt wykorzystuje **logikę rozmytą** (Fuzzy Logic), która:
- ✅ Zapewnia płynne przejścia między stanami oświetlenia
- ✅ Naturalnie uwzględnia niepewność pomiarów
- ✅ Odzwierciedla ludzkie rozumowanie (np. "ciemno", "jasno")
- ✅ Osiąga wartości skrajne: 0% (pełne słońce) i 100% (głęboka noc)

## 👥 Autorzy

- Jan Skulimowski (s27144)
- Kamil Littwitz (s26966)

## 🔧 Wymagania

### Oprogramowanie
- Python 3.7 lub nowszy

### Biblioteki
- `numpy` (>=1.19.0) - operacje na tablicach
- `scikit-fuzzy` (>=0.4.2) - implementacja logiki rozmytej

### Przygotowanie środowiska

1. Wymagany Python 3.7 lub nowszy

2. Instalacja wymaganych bibliotek:
   pip install numpy scikit-fuzzy

3. Uruchomienie programu:
      python fuzzy_light_control.py

### 1️⃣ Tryb Interaktywny
Wprowadzaj własne wartości i obserwuj wyniki w czasie rzeczywistym:
```
Godzina (0-24): 14.5
Światło naturalne (0-100%): 85

────────────────────────────────────────
⏰ Czas: 14:30
☀️  Światło naturalne: 85.0%
💡 JASNOŚĆ ŻARÓWKI: 0.00%
────────────────────────────────────────
```

### 2️⃣ Symulacja 24-godzinnego Cyklu
Automatyczne przejście przez cały dzień z wyświetleniem co 30 minut:
```
Czas: 00:00 | Światło naturalne:   0.0% | ==> Jasność żarówki: 100.0%
Czas: 00:30 | Światło naturalne:   0.0% | ==> Jasność żarówki: 100.0%
Czas: 01:00 | Światło naturalne:   0.0% | ==> Jasność żarówki: 100.0%
...
Czas: 14:00 | Światło naturalne: 100.0% | ==> Jasność żarówki:   0.0%
```

### 3️⃣ Test Warunków Brzegowych
Sprawdzenie działania w skrajnych i typowych przypadkach:
```
IDEALNIE: Słoneczne popołudnie          | Jasność:   0.00%
NAJGORZEJ: Głęboka noc, brak światła    | Jasność: 100.00%
Dzień, częściowe zachmurzenie           | Jasność:   8.00%
```

### Zmienne Wejściowe

#### Pora Dnia (4 kategorie):
- **Noc** (0:00 - 6:00)
- **Poranek** (5:00 - 11:00)
- **Dzień** (10:00 - 18:00)
- **Wieczór** (17:00 - 24:00)

#### Światło Naturalne (4 kategorie):
- **Bardzo ciemno** (0-20%)
- **Ciemno** (15-45%)
- **Umiarkowanie** (40-70%)
- **Jasno** (65-100%)

### Zmienna Wyjściowa

#### Jasność Żarówki (7 poziomów):
- **Off** (0%) - wyłączone
- **Minimal** (8%) - minimalne podświetlenie
- **Very Low** (20%) - bardzo niskie
- **Low** (35%) - niskie
- **Medium** (50%) - średnie
- **High** (70%) - wysokie
- **Full** (100%) - pełne

