#Gomoku (Pięć w Rzędzie)

Opis projektu

Projekt polega na implementacji turowej, deterministycznej gry dwuosobowej o sumie zerowej, w której dwóch graczy rywalizuje, aby ułożyć pięć swoich pionków w rzędzie (poziomo, pionowo lub na ukos) na planszy 10x10. Gra jest odmianą klasycznego Kółko i Krzyżyk, ale z większą planszą i celem.

W ramach zadania, poza samą grą, zaimplementowano również sztuczną inteligencję (SI), która potrafi grać przeciwko człowiekowi, wykorzystując bibliotekę EasyAI i algorytm Negamax.

Instrukcja przygotowania środowiska i uruchomienia

Wymagania wstępne

Projekt został zaimplementowany w języku Python. Wymaga zainstalowania biblioteki EasyAI.

Możesz zainstalować wymagane pakiety, używając poniższego polecenia:
Bash

pip install easyAI

Uruchomienie

Aby uruchomić grę, wykonaj następujące polecenie w terminalu, będąc w głównym katalogu projektu:
Bash

python gomoku.py

Zasady gry

Celem gry jest ułożenie pięciu pionków tego samego koloru w jednym, ciągłym rzędzie. Może to być linia pozioma, pionowa lub diagonalna.

    Gra toczy się na planszy o wymiarach 10x10.

    Gracze na zmianę stawiają swoje pionki na planszy, podając współrzędne x,y.

    Pierwszy gracz, który ułoży pięć pionków w rzędzie, wygrywa.

    Jeśli cała plansza zostanie zapełniona i żaden z graczy nie ułoży pięciu w rzędzie, gra kończy się remisem.

Zaimplementowane algorytmy

    Negamax: Jest to wariant algorytmu minimax przeznaczony do gier o sumie zerowej. Działa na zasadzie, że gracz, który ma się ruszyć, chce zmaksymalizować swoją ocenę, podczas gdy jego przeciwnik chce ją zminimalizować.

    Heurystyka punktowa: W kodzie znajduje się funkcja scoring, która ocenia stan planszy, dając wyższą wartość ciągom pionków własnego gracza i niższą ciągom pionków przeciwnika. Zapewnia to, że SI podejmuje strategiczne decyzje, dążąc do tworzenia własnych linii, jednocześnie blokując przeciwnika.

Zrzut ekranu z rozgrywki

************************

Autorzy

    Jan Skulimowski (s27144)

    Kamil Littwitz (s26966)
