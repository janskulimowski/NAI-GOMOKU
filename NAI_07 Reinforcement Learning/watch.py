"""
NAZWA PLIKU: watch.py

OPIS PROBLEMU:
    Skrypt do ewaluacji i wizualizacji wytrenowanego modelu RL w grze Galaxian.
    Rozwiązuje problem ograniczeń renderowania biblioteki Gymnasium (blokada FPS)
    poprzez użycie zewnętrznej biblioteki OpenCV.
    Dodatkowo skrypt naprawia problem "ucinania nagród" (Reward Clipping),
    wyświetlając rzeczywisty wynik punktowy z gry, a nie znormalizowaną nagrodę agenta.

AUTORZY:
    Kamil Littwitz
    Jan Skulimowski

INSTRUKCJA UŻYCIA:
    1. Upewnij się, że plik modelu (np. 'models/PPO/galaxian_bot_v2.zip') istnieje.
    2. Uruchom skrypt:
       python watch.py
    3. Sterowanie oknem podglądu:
       - 'q': Zamknij grę.
       - Edycja kodu (cv2.waitKey): Zmiana prędkości gry (Turbo/Normal/Slow).

WYMAGANIA:
    - opencv-python (pip install opencv-python)
"""

import gymnasium as gym
import ale_py
import shimmy
import cv2
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- KONFIGURACJA ---
ENV_ID = "ALE/Galaxian-v5"
# Ścieżka do wytrenowanego modelu (upewnij się, że nazwa pliku jest poprawna)
MODEL_PATH = "models/PPO/galaxian_bot"


def main():
    """
    Główna pętla obsługująca wczytanie modelu, renderowanie gry i logikę punktacji.
    """
    print(f"Tworzenie środowiska: {ENV_ID}...")

    # WAŻNE: wrapper_kwargs={"clip_reward": False}
    # Wyłącza normalizację nagród (np. spłaszczanie +100 do +1).
    # Dzięki temu środowisko zwraca oryginalne punkty z gry Atari.
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=0,
        env_kwargs={"render_mode": "rgb_array"},  # Pobieramy surowe piksele dla OpenCV
        wrapper_kwargs={"clip_reward": False}
    )
    # Stosowanie 4 klatek (musi być takie samo jak w train.py)
    env = VecFrameStack(env, n_stack=4)

    # Wczytywanie modelu
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku modelu: {MODEL_PATH}.zip")
        print("Uruchom najpierw skrypt 'train.py'.")
        return

    print(f"Wczytuję model z: {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Gra uruchomiona! (Wciśnij 'q' na oknie gry, aby wyjść)")

    # Inicjalizacja zmiennych
    obs = env.reset()
    current_score = 0.0

    try:
        while True:
            # 1. Predykcja akcji przez model
            # deterministic=True sprawia, że bot zawsze wybiera akcję o najwyższym prawdopodobieństwie
            action, _states = model.predict(obs, deterministic=True)

            # 2. Wykonanie kroku w środowisku
            obs, rewards, dones, info = env.step(action)

            # Pobranie informacji dodatkowych
            infos = info[0]
            lives = infos.get("lives", 0)

            # Aktualizacja bieżącego wyniku (dzięki clip_reward=False, rewards[0] to punkty gry)
            current_score += rewards[0]

            # 3. Wyświetlanie statystyk w terminalu
            print(f"Życia: {lives} | Wynik bieżący: {int(current_score)}    ", end="\r")

            # 4. Renderowanie obrazu za pomocą OpenCV
            img = env.render()
            # Konwersja RGB -> BGR (OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Powiększenie obrazu x3 (z ~210px do 630px) bez rozmycia (INTER_NEAREST)
            img = cv2.resize(img, (640, 420), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Galaxian Bot AI", img)

            # Kontrola prędkości (waitKey):
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

            # 5. Obsługa końca gry lub śmierci
            if dones[0]:
                # Pobranie oficjalnego wyniku z podsumowania epizodu (najbardziej precyzyjne)
                if "episode" in infos:
                    final_score = infos["episode"]["r"]
                else:
                    final_score = current_score

                if lives == 0:
                    # Wyczyszczenie linii w terminalu
                    print(" " * 60, end="\r")
                    print(f">>> GAME OVER! Oficjalny wynik: {int(final_score)}")
                    print("-" * 40)

                    # Reset licznika i pauza dla widza
                    current_score = 0
                    cv2.waitKey(2000)
                else:
                    # Tylko strata życia, gra toczy się dalej
                    print(" " * 60, end="\r")
                    print(f">>> Strata życia! Pozostało: {lives}")
                    cv2.waitKey(500)

    except KeyboardInterrupt:
        print("\nZamykanie przez Ctrl+C...")
    finally:
        # Sprzątanie zasobów
        print("\nZamykanie środowiska...")
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()