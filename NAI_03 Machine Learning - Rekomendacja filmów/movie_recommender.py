"""
================================================================================
OPIS PROBLEMU:
Implementacja silnika rekomendacji (algorytm K-Means)
opartego na aktualnym zbiorze danych MovieLens Latest (Small).

System jest trenowany na bazie MovieLens. Następnie, pobiera profil
użytkownika (z pliku ankiety CSV), mapuje jego oceny na filmy
z bazy MovieLens i przewiduje jego przynależność do klastra.
Na tej podstawie generuje 5 rekomendacji i 5 anty-rekomendacji
z puli filmów MovieLens, których nowy użytkownik jeszcze nie widział.

System wykorzystuje klasę OMDbEngine do pobierania metadanych
(rok, gatunek, fabuła) dla polecanych tytułów z OMDb API.

AUTORZY:
Jan Skulimowski
Kamil Littwitz

INSTRUKCJA UŻYCIA:
1.  Pobierz zbiór MovieLens Latest (Small) (ml-latest-small.zip)
    ze strony: https://grouplens.org/datasets/movielens/latest/
2.  Umieść pliki `ratings.csv` i `movies.csv` w tym samym katalogu co skrypt.
3.  Upewnij się, że plik `ankieta.csv` również
    znajduje się w tym katalogu.
4.  Zdobądź darmowy klucz API z OMDb API:
    Zarejestruj się na https://www.omdbapi.com/apikey.aspx
5.  Wklej swój klucz API w zmiennej `OMDB_API_KEY` poniżej (linia 58).
6.  Zainstaluj wymagane biblioteki:
    pip install pandas scikit-learn requests
7.  Uruchom skrypt:
    python ml_recommender.py
8.  Możesz zmienić użytkownika do testów w zmiennej `TEST_USER_ID`
    na dole skryptu.
================================================================================
"""

# --- Krok 1: Importowanie bibliotek ---
import pandas as pd
import numpy as np
import requests
import re
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict
import warnings

# Ignorowanie ostrzeżeń
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfiguracja ---
OMDB_API_KEY = "bbf5ce75"  # ZASTĄP MNIE SWOIM KLUCZEM
SURVEY_FILE_PATH = "ankieta.csv"  # Źródło użytkowników
ML_DATA_PATH = "ratings.csv"     # ZMIANA: Źródło treningowe (oceny)
ML_ITEM_PATH = "movies.csv"     # ZMIANA: Źródło treningowe (tytuły)

N_CLUSTERS = 25 # Można dostosować
RANDOM_STATE = 42

#
# =============================================================================
# CZĘŚĆ 1: SILNIK WYSZUKIWANIA OMDb
# =============================================================================
#

class OMDbEngine:
    """
    Hermetyzuje logikę zapytań do OMDb API.
    (Constraint 6: Dokumentacja - Docstring)
    """

    def __init__(self, api_key: str):
        self.base_url = "https://www.omdbapi.com/"
        self.api_key = api_key

        if self.api_key == "TWOJ_KLUCZ_API_OMDB":
            print("OSTRZEŻENIE API: Używasz domyślnego klucza API. "
                  "Zastąp 'TWOJ_KLUCZ_API_OMDB' swoim prawdziwym kluczem.")

    def _make_request(self, params: dict) -> dict:
        if self.api_key == "TWOJ_KLUCZ_API_OMDB":
             return {"Response": "False", "Error": "Klucz API nie został ustawiony."}

        params['apikey'] = self.api_key
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            return {"Response": "False", "Error": f"Błąd połączenia: {e}"}
        except json.JSONDecodeError:
            return {"Response": "False", "Error": "Błąd parsowania JSON."}

    def get_movie_details(self, title: str = None) -> dict:
        """
        Pobiera szczegółowe informacje na podstawie tytułu.
        (Constraint 6: Dokumentacja - Docstring)
        """
        if not title:
            return {"Response": "False", "Error": "Musisz podać tytuł."}
        params = {"t": title, "plot": "short"}
        return self._make_request(params)

#
# =============================================================================
# CZĘŚĆ 2: PRZETWARZANIE DANYCH (Ankieta)
# =============================================================================
#

def load_survey_user(file_path: str, user_id: str) -> list:
    """
    Wczytuje oceny dla JEDNEGO użytkownika z pliku ankiety CSV.

    Args:
        file_path (str): Ścieżka do `ankieta.csv`
        user_id (str): Identyfikator użytkownika

    Returns:
        list: Lista słowników, np. [{'title': 'Matrix', 'rating': 10}, ...]
    """
    print(f"\nWczytuję profil użytkownika: {user_id}")
    try:
        df_raw = pd.read_csv(file_path, header=None, skiprows=1, index_col=0)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku ankiety: {file_path}")
        return []
    except Exception as e:
        print(f"Błąd wczytywania ankiety: {e}")
        return []

    user_ratings = []
    try:
        user_row = df_raw.loc[user_id]
    except KeyError:
        print(f"BŁĄD: Nie znaleziono użytkownika '{user_id}' w pliku ankiety.")
        return []

    # POPRAWKA: Używamy .iloc do iteracji po pozycjach,
    # aby uniknąć błędu KeyError: 0
    for i in range(0, len(user_row), 2):
        try:
            movie_title = user_row.iloc[i]
            rating = user_row.iloc[i+1]

            if pd.isna(movie_title) or pd.isna(rating):
                break

            numeric_rating = float(rating)
            if isinstance(movie_title, str) and movie_title.strip() != "":
                clean_title = movie_title.strip()
                user_ratings.append({
                    "title": clean_title,
                    "rating": numeric_rating
                })
        except (ValueError, TypeError):
            pass # Pomijamy błędne wpisy
        except IndexError:
            break # Zakończ pętlę, jeśli wyjdziemy poza zakres

    print(f"Znaleziono {len(user_ratings)} ocen dla '{user_id}'.")
    return user_ratings

#
# =============================================================================
# CZĘŚĆ 3: SILNIK REKOMENDACJI K-MEANS (na bazie MovieLens Latest)
# =============================================================================
#

class ClusteringRecommender:
    """
    Silnik rekomendacji trenowany na MovieLens "Latest-Small", potrafiący
    generować rekomendacje dla nowych użytkowników.
    (Constraint 6: Dokumentacja - Docstring)
    """

    def __init__(self, n_clusters: int, random_state: int, api_key: str):
        """
        Inicjalizuje silnik.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.omdb_engine = OMDbEngine(api_key)
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state,
                             n_init='auto')
        self.scaler = MaxAbsScaler()
        self.item_id_to_title = {}
        self.title_to_item_id = {}
        self.n_movies = 0
        self.n_users = 0
        self.user_item_matrix_raw = None
        self.cluster_avg_ratings = {}

    def _load_movielens_data(self):
        """
        Wczytuje dane MovieLens "latest-small" (ratings.csv i movies.csv).
        Tworzy mapowania tytułów.
        (ZAKTUALIZOWANE DLA `ml-latest-small`)
        """
        print("Wczytuję dane MovieLens 'latest-small'...")
        try:
            # Wczytaj movies.csv (ma nagłówki: movieId, title, genres)
            i_cols = ['movieId', 'title']
            movies = pd.read_csv(ML_ITEM_PATH, sep=',', usecols=i_cols)

            # Wczytaj ratings.csv (ma nagłówki: userId, movieId, rating)
            r_cols = ['userId', 'movieId', 'rating']
            self.ratings = pd.read_csv(ML_DATA_PATH, sep=',', usecols=r_cols)

        except FileNotFoundError as e:
            print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku MovieLens: {e.filename}")
            print("Upewnij się, że pobrałeś `ml-latest-small.zip` i rozpakowałeś")
            print("pliki `ratings.csv` i `movies.csv` do tego samego katalogu.")
            return False

        # Tworzenie mapowań
        for _, row in movies.iterrows():
            item_id = row['movieId'] # ZMIANA: Kolumna nazywa się 'movieId'
            full_title = row['title'] # Np. "Toy Story (1995)"

            clean_title = re.sub(r'\s\(\d{4}\)$', '', full_title).lower().strip()

            self.item_id_to_title[item_id] = full_title
            if clean_title not in self.title_to_item_id:
                self.title_to_item_id[clean_title] = item_id

        # ZMIANA: Używamy nowych nazw kolumn do znalezienia rozmiaru macierzy
        self.n_movies = self.ratings['movieId'].max()
        self.n_users = self.ratings['userId'].max()

        print(f"Wczytano {len(self.item_id_to_title)} filmów i {len(self.ratings)} ocen.")
        print(f"Rozmiar macierzy: {self.n_users} użytkowników x {self.n_movies} filmów.")
        return True

    def _create_user_item_matrix(self):
        """
        Tworzy i normalizuje macierz użytkownik-film dla MovieLens.
        (ZAKTUALIZOWANE DLA `ml-latest-small`)
        """
        print(f"Tworzenie macierzy użytkownik-film ({self.n_users} użytkowników x {self.n_movies} filmów)...")

        # ZMIANA: Używamy 'userId' i 'movieId'
        self.user_item_matrix_raw = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        )

        # POPRAWKA (z błędu NaN): Uzupełniamy macierz do pełnego rozmiaru
        # i zastępujemy WSZYSTKIE NaN zerami.
        self.user_item_matrix_raw = self.user_item_matrix_raw.reindex(
            index=range(1, self.n_users + 1),
            columns=range(1, self.n_movies + 1),
        ).fillna(0)

        print("Macierz gotowa. Normalizuję dane...")
        self.matrix_scaled = self.scaler.fit_transform(self.user_item_matrix_raw)
        print("Normalizacja zakończona.")

    def fit(self):
        """
        Trenuje model K-Means na danych MovieLens.
        (Constraint 1 i 2)
        """
        if not self._load_movielens_data():
            return

        self._create_user_item_matrix()

        print(f"\nTrenowanie modelu K-Means z {self.n_clusters} klastrami...")
        self.kmeans.fit(self.matrix_scaled)

        score = silhouette_score(self.matrix_scaled, self.kmeans.labels_)

        print("Obliczanie średnich ocen dla klastrów...")
        labels = self.kmeans.labels_
        for cluster_id in range(self.n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_matrix = self.user_item_matrix_raw[cluster_mask]
            cluster_matrix_no_zeros = cluster_matrix.replace(0, np.nan)
            cluster_avg = cluster_matrix_no_zeros.mean(axis=0, skipna=True).fillna(0)
            self.cluster_avg_ratings[cluster_id] = cluster_avg
        print("Model gotowy do generowania rekomendacji.")

    def _create_new_user_vector(self, new_user_ratings: list) -> (np.ndarray, set):
        """
        Tworzy wektor ocen nowego użytkownika, pasujący do macierzy MovieLens.
        """
        # Tworzy pusty wektor o rozmiarze liczby filmów w MovieLens
        # Musimy użyć int(), ponieważ n_movies może być typu numpy.int64
        user_vector = np.zeros(int(self.n_movies))
        seen_item_ids = set()

        matched_count = 0

        for item in new_user_ratings:
            clean_survey_title = item['title'].lower().strip()

            if clean_survey_title in self.title_to_item_id:
                item_id = self.title_to_item_id[clean_survey_title]

                # Skalujemy 1-10 na 0.5-5.0 (tak jak w nowej bazie)
                # Np. ocena 10 -> 5.0; ocena 7 -> 3.5; ocena 1 -> 0.5
                rating_0_5_to_5 = item['rating'] / 2.0

                # Indeksy w numpy są od 0, a item_id od 1
                # Upewnijmy się, że item_id jest w granicach
                if 0 <= (item_id - 1) < len(user_vector):
                    user_vector[item_id - 1] = rating_0_5_to_5
                    seen_item_ids.add(item_id)
                    matched_count += 1

        return user_vector.reshape(1, -1), seen_item_ids

    def get_recommendations_for_new_user(self, new_user_ratings: list, n_rec: int = 5, n_anti_rec: int = 5) -> dict:
        """
        Generuje rekomendacje dla nowego użytkownika na podstawie jego profilu.
        (Constraint 3, 4)
        """
        if self.user_item_matrix_raw is None:
            return {"error": "Model nie został jeszcze wytrenowany. Wywołaj .fit()"}

        user_vector, seen_item_ids = self._create_new_user_vector(new_user_ratings)
        user_vector_scaled = self.scaler.transform(user_vector)
        predicted_cluster = self.kmeans.predict(user_vector_scaled)[0]

        cluster_avg = self.cluster_avg_ratings[predicted_cluster]

        # Filtruj filmy: usuń te, które użytkownik już widział
        # Musimy upewnić się, że usuwamy tylko ID, które istnieją w indeksie cluster_avg
        valid_seen_ids = [id for id in seen_item_ids if id in cluster_avg.index]
        potential_recs = cluster_avg.drop(labels=valid_seen_ids, errors='ignore')

        top_recs_series = potential_recs.sort_values(ascending=False).head(n_rec)

        # 1. Stwórz nową serię TYLKO z filmami, które miały ocenę > 0
        rated_movies = potential_recs[potential_recs > 0]
        # 2. Sortuj tylko spośród filmów FAKTYCZNIE ocenionych
        anti_recs_series = rated_movies.sort_values(ascending=True).head(n_anti_rec)

        print("\nPobieranie szczegółów API dla rekomendacji...")
        recommendations = self._format_output(top_recs_series)
        print("Pobieranie szczegółów API dla anty-rekomendacji...")
        anti_recommendations = self._format_output(anti_recs_series)

        return {
            "recommendations": recommendations,
            "anti_recommendations": anti_recommendations
        }

    def _format_output(self, movie_series: pd.Series) -> list:
        """
        Prywatna metoda pomocnicza do formatowania i wywoływania API OMDb.
        (Constraint 5)
        """
        output_list = []
        for item_id, avg_rating in movie_series.items():

            ml_title = self.item_id_to_title.get(item_id, "Nieznany Tytuł")
            clean_title_for_api = re.sub(r'\s\(\d{4}\)$', '', ml_title)

            api_details = self.omdb_engine.get_movie_details(title=clean_title_for_api)

            if api_details.get("Response") == "True":
                overview = api_details.get("Plot", "Brak opisu.")
                poster_url = api_details.get("Poster", "Brak plakatu")
                api_title = api_details.get("Title", "Brak tytułu")
                year = api_details.get("Year", "Brak roku")
                genre = api_details.get("Genre", "Brak gatunku")
            else:
                overview = f"(OMDb: {api_details.get('Error', 'Nie znaleziono')})"
                poster_url = "N/A"
                api_title = "Nie znaleziono"
                year = "N/A"
                genre = "N/A"

            output_list.append({
                "title": ml_title, # Tytuł z bazy MovieLens
                "cluster_avg_rating": avg_rating, # To już jest float
                "overview": overview,
                "poster_url": poster_url,
                "api_title": api_title, # Tytuł znaleziony w OMDb
                "year": year,
                "genre": genre
            })
        return output_list

#
# =============================================================================
# CZĘŚĆ 4: URUCHOMIENIE (GŁÓWNY SKRYPT)
# =============================================================================
#

def main():
    """Główna funkcja uruchomieniowa skryptu."""

    if OMDB_API_KEY == "TWOJ_KLUCZ_API_OMDB":
        print("="*70)
        print("!!! OSTRZEŻENIE !!!")
        print("Nie ustawiono klucza OMDB_API_KEY.")
        print("Rekomendacje zostaną wygenerowane, ale bez opisów fabuły i plakatów.")
        print("="*70)

    # 1. Inicjalizuj i TRENUJ silnik na MovieLens
    engine = ClusteringRecommender(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        api_key=OMDB_API_KEY
    )
    engine.fit()

    # 2. Wybierz użytkownika z ankiety
    TEST_USER_ID = "Michał Fritza" # ([imię] [nazwisko])

    # 3. Załaduj profil tego użytkownika
    new_user_ratings = load_survey_user(SURVEY_FILE_PATH, TEST_USER_ID)

    if not new_user_ratings:
        print(f"Nie można wygenerować rekomendacji dla '{TEST_USER_ID}'.")
        return

    # 4. Wygeneruj dla niego rekomendacje
    print(f"\n{'-'*70}\nGenerowanie rekomendacji dla: {TEST_USER_ID}\n{'-'*70}")

    results = engine.get_recommendations_for_new_user(new_user_ratings, n_rec=5, n_anti_rec=5)

    if "error" in results:
        print(results["error"])
    else:
        # 5. Wyświetl wyniki
        print("\n" + "="*70)
        print(f"TOP 5 REKOMENDACJI dla użytkownika '{TEST_USER_ID}'")
        print("(Filmy z bazy MovieLens, które powinieneś obejrzeć)")
        print("="*70)
        if not results["recommendations"]:
            print("(Brak rekomendacji do wyświetlenia)")
        else:
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"\n{i}. {rec['title']} (API: {rec['api_title']})")
                print(f"   Rok: {rec['year']} | Gatunek: {rec['genre']}")
                print(f"   Średnia ocena w klastrze: {rec['cluster_avg_rating']:.2f}/5.0")
                # ZMIANA: Formatujemy ocenę do 2 miejsc po przecinku
                print(f"   Fabuła: {rec['overview']}")
                print(f"   Plakat: {rec['poster_url']}")

        print("\n" + "="*70)
        print(f"TOP 5 ANTY-REKOMENDACJI dla użytkownika '{TEST_USER_ID}'")
        print("(Filmy z bazy MovieLens, których prawdopodobnie nie polubisz)")
        print("="*70)
        if not results["anti_recommendations"]:
            print("(Brak anty-rekomendacji do wyświetlenia)")
        else:
            for i, anti_rec in enumerate(results["anti_recommendations"], 1):
                print(f"\n{i}. {anti_rec['title']} (API: {anti_rec['api_title']})")
                print(f"   Rok: {anti_rec['year']} | Gatunek: {anti_rec['genre']}")
                print(f"   Średnia ocena w klastrze: {anti_rec['cluster_avg_rating']:.2f}/5.0")
                print(f"   Fabuła: {anti_rec['overview']}")
                print(f"   Plakat: {anti_rec['poster_url']}")

if __name__ == "__main__":
    main()