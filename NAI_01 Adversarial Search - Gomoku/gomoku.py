# Zasady/zadanie: https://pl.wikipedia.org/wiki/Gomoku
#Link do repozytorium (publiczne): https://github.com/janskulimowski/NAI-GOMOKU/
# Autorzy: Jan Skulimowski (s27144), Kamil Littwitz (s26966)                            
# Przygotowanie środowiska:
#   - Python 3.10+
#   - pip install easyAI
#   - Uruchom: python gomoku.py

"""
Gomoku (pięć w rzędzie) oparte o easyAI.

Opis:
    Implementacja gry dwuosobowej (człowiek vs AI). Gracz 1 = 'X', Gracz 2 = 'O'.
    Silnik wykorzystuje Negamax z zadaną głębokością.
    Jako ruch wpisywać x,y (rząd, kolumna)
Wymagania:
    Python 3.10+, easyAI

Uruchomienie:
    python gomoku.py
"""

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class Gomoku(TwoPlayerGame):
    def __init__(self, players, size=10, win_len=5):
        """
        Inicjalizuje grę Gomoku.

        Parameters:
            players (list): [Human_Player(), AI_Player(...)] – dwóch graczy w kolejności tur.
            size (int): Rozmiar planszy (size x size).
            win_len (int): Długość linii jednakowych znaków wymagana do wygranej.

        Returns:
            None
        """
        self.players = players
        self.size = size
        self.win_len = win_len

        # Reprezentacja planszy: kropka oznacza puste pole
        self.board = [["." for _ in range(size)] for _ in range(size)]

        # Numer aktualnego gracza wg easyAI: 1 lub 2
        # Gracz 1 zaczyna i stawia 'X', gracz 2 stawia 'O'
        self.nplayer = 1

    # 🩹 Zgodność z nowszym easyAI: część wersji korzysta z current_player
    @property
    def current_player(self):
        # Zwracamy numer aktualnego gracza (easyAI odczytuje tę właściwość)
        return self.nplayer

    @current_player.setter
    def current_player(self, value):
        # Umożliwia easyAI ustawianie aktualnego gracza
        self.nplayer = value

    def possible_moves(self):
        """Zwraca listę możliwych ruchów w formacie 'x,y' (współrzędne wiersz,kolumna)."""
        # Przechodzimy po całej planszy i zwracamy współrzędne pustych pól
        return [
            f"{i},{j}"
            for i in range(self.size)
            for j in range(self.size)
            if self.board[i][j] == "."
        ]

    def make_move(self, move):
        """Wykonuje ruch na planszy."""
        # Ruch przychodzi jako "i,j" -> parsujemy na int
        i, j = map(int, move.split(","))
        # Znak zależy od tego, który gracz ma ruch (1 -> 'X', 2 -> 'O')
        self.board[i][j] = "X" if self.nplayer == 1 else "O"

    def unmake_move(self, move):
        """Cofa ruch (potrzebne dla przeszukiwania przez AI)."""
        # Przywracamy puste pole – używa tego silnik przy sprawdzaniu wariantów
        i, j = map(int, move.split(","))
        self.board[i][j] = "."

    def is_over(self):
        """Zwraca True, jeśli gra się skończyła: ktoś wygrał albo brak ruchów."""
        # Gra kończy się, gdy jest zwycięzca (win()!=0) albo plansza jest pełna
        return self.win() != 0 or not self.possible_moves()

    def win(self):
        """Sprawdza, czy któryś z graczy wygrał. Zwraca 1 dla 'X', 2 dla 'O', 0 gdy brak zwycięzcy."""
        # Sprawdzamy każdy punkt planszy jako potencjalny element linii
        for i in range(self.size):
            for j in range(self.size):
                p = self.board[i][j]
                if p == ".":
                    continue  # puste pole nas nie interesuje
                # Kierunki: poziomo, pionowo, ukośnie (obie przekątne)
                for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    count = 1  # długość aktualnego ciągu jednakowych znaków

                    # Idziemy do przodu w danym kierunku
                    ni, nj = i + dx, j + dy
                    while (
                        0 <= ni < self.size
                        and 0 <= nj < self.size
                        and self.board[ni][nj] == p
                    ):
                        count += 1
                        ni += dx
                        nj += dy

                    # Idziemy w przeciwnym kierunku, żeby policzyć cały ciąg
                    ni, nj = i - dx, j - dy
                    while (
                        0 <= ni < self.size
                        and 0 <= nj < self.size
                        and self.board[ni][nj] == p
                    ):
                        count += 1
                        ni -= dx
                        nj -= dy

                    # Jeśli długość ciągu osiągnęła win_len, mamy zwycięzcę
                    if count >= self.win_len:
                        return 1 if p == "X" else 2

        # Brak zwycięzcy
        return 0

    def show(self):
        """Wyświetla planszę w konsoli (koordynaty + zawartość pól)."""
        # Nagłówek kolumn
        print("  " + " ".join(str(i) for i in range(self.size)))
        # Każdy wiersz z jego numerem (wyrównanym do 2 znaków dla czytelności)
        for idx, row in enumerate(self.board):
            print(f"{idx:2} " + " ".join(row))
        print()

    def scoring(self):
        """
        Heurystyka dla AI — ocenia pozycję z perspektywy aktualnego gracza.
        - Jeśli ktoś już wygrał: zwracamy bardzo dużą wartość dodatnią/ujemną.
        - W przeciwnym razie liczymy "siłę" pozycji na podstawie długości ciągów
          jednakowych symboli w czterech kierunkach. Im dłuższy ciąg, tym większa premia (count**3).
        - Wynik zwracamy jako różnicę między 'X' i 'O' z uwzględnieniem, kto jest przy ruchu.
        """
        scores = {"X": 0, "O": 0}
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        # Liczymy „potencjał” wszystkich istniejących ciągów
        for i in range(self.size):
            for j in range(self.size):
                p = self.board[i][j]
                if p == ".":
                    continue
                for (dx, dy) in directions:
                    count = 1
                    ni, nj = i + dx, j + dy
                    while (
                        0 <= ni < self.size
                        and 0 <= nj < self.size
                        and self.board[ni][nj] == p
                    ):
                        count += 1
                        ni += dx
                        nj += dy
                    # Premia rośnie potęgowo, aby faworyzować dłuższe sekwencje
                    scores[p] += count ** 3

        # Twarda wygrana/porazka ma pierwszeństwo przed heurystyką
        winner = self.win()
        if winner == 1:
            return 10000     # 'X' wygrał -> maksymalna korzyść dla gracza X
        elif winner == 2:
            return -10000    # 'O' wygrał -> maksymalna strata dla gracza X

        # Różnica ocen: jeżeli ruch ma 'X' (nplayer == 1) patrzymy X - O,
        # a jeżeli 'O' ma ruch, patrzymy O - X (odwrócenie perspektywy).
        if self.nplayer == 1:
            return scores["X"] - scores["O"]
        else:
            return scores["O"] - scores["X"]

    def ttentry(self):
        """
        Zwraca hashowalny (niemodyfikowalny) opis stanu gry dla tabeli transpozycji.
        Dzięki temu easyAI może zapamiętywać i ponownie wykorzystywać wyniki ocen
        dla identycznych pozycji napotkanych podczas przeszukiwania.
        """
        return tuple(tuple(row) for row in self.board)


if __name__ == "__main__":
    # Konfigurujemy algorytm AI: Negamax z głębokością 4.
    # Większa głębokość = lepsza gra, ale wolniejsze działanie.
    ai_algo = Negamax(3)

    # Tworzymy instancję gry:
    # - Gracz 1: człowiek (stawia 'X', zaczyna),
    # - Gracz 2: AI (stawia 'O'),
    # - Plansza 10x10,
    # - Wygrywa linia 5 znaków.
    game = Gomoku([Human_Player(), AI_Player(ai_algo)], size=10, win_len=5)

    # Uruchamiamy pętlę gry (easyAI zajmuje się naprzemiennymi turami,
    # pytaniem człowieka o ruch, wołaniem AI itd.)
    game.play()

    # --- komunikat o wyniku po zakończeniu gry ---
    w = game.win()
    if w == 1:
        print("Wygrał gracz (X)!\n")
    elif w == 2:
        print("Wygrało SI (O)!\n")
    else:
        print("Remis!")
