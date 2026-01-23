"""
NAZWA PLIKU: train.py

OPIS PROBLEMU:
    Celem jest wytrenowanie agenta sztucznej inteligencji (AI) do gry w Galaxian
    na platformie Atari 2600 przy użyciu uczenia ze wzmocnieniem (Reinforcement Learning).
    Agent musi nauczyć się sterować statkiem, unikać pocisków wrogów i zdobywać punkty.
    Zastosowano algorytm PPO (Proximal Policy Optimization) z biblioteki Stable Baselines3.

AUTORZY:
    Kamil Littwitz
    Jan Skulimowski

INSTRUKCJA UŻYCIA:
    1. Zainstaluj wymagane biblioteki:
       pip install gymnasium[atari] gymnasium[accept-rom-license] stable-baselines3 shimmy ale-py
    2. Uruchom skrypt w terminalu:
       python train.py
    3. Proces uczenia stworzy katalogi 'checkpoints' (zapisy tymczasowe)
       oraz 'models/PPO' (zapis końcowy).
    4. Czas trwania: od 30 min do kilku godzin (zależnie od sprzętu i liczby kroków).

REFERENCJE:
    - Stable Baselines3 Docs: https://stable-baselines3.readthedocs.io/
    - Gymnasium Atari: https://gmail.farama.org/environments/atari/
    - PPO Paper: https://arxiv.org/abs/1707.06347
"""

import gymnasium as gym
import ale_py
import shimmy
import os
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Tworzy funkcję harmonogramu dla współczynnika uczenia (learning rate).
    Współczynnik ten maleje liniowo w czasie trwania treningu.

    Args:
        initial_value (float): Początkowa wartość learning rate (np. 2.5e-4).

    Returns:
        Callable[[float], float]: Funkcja przyjmująca 'progress_remaining' (od 1.0 do 0.0)
        i zwracająca aktualny learning rate.
    """

    def func(progress_remaining: float) -> float:
        # progress_remaining spada od 1.0 (start) do 0.0 (koniec)
        return progress_remaining * initial_value

    return func


# --- KONFIGURACJA GŁÓWNA ---
ENV_ID = "ALE/Galaxian-v5"  # ID środowiska w Gymnasium
TOTAL_TIMESTEPS = 2_000_000  # Całkowita liczba kroków symulacji
N_ENVS = 8  # Liczba równoległych środowisk (przyspiesza trening)
SEED = 0  # Ziarno losowości dla powtarzalności wyników

# Ścieżki do zapisu
CHECKPOINT_DIR = "checkpoints"
MODELS_DIR = "models/PPO"


def main():
    """
    Główna funkcja uruchamiająca proces treningowy.
    """
    # 1. Przygotowanie katalogów
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 2. Inicjalizacja środowiska wektorowego
    # make_atari_env automatycznie aplikuje wrappery (zmiana rozmiaru, skala szarości)
    print(f"Tworzenie {N_ENVS} równoległych środowisk: {ENV_ID}...")
    vec_env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=SEED)

    # Nakładanie klatek - kluczowe dla widzenia ruchu
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # 3. Konfiguracja modelu PPO
    # Hiperparametry dobrane pod gry Atari
    model = PPO(
        policy="CnnPolicy",  # Użycie splotowych sieci neuronowych (CNN) do analizy obrazu
        env=vec_env,
        verbose=1,
        learning_rate=linear_schedule(2.5e-4),  # Dynamiczne zmniejszanie LR
        n_steps=128,  # Liczba kroków na środowisko w jednej aktualizacji
        batch_size=256,  # Rozmiar partii danych
        n_epochs=4,  # Liczba epok optymalizacji na partii
        gamma=0.99,  # Współczynnik dyskontowania
        clip_range=0.1,  # Ograniczenie zmian polityki
        ent_coef=0.01,  # Współczynnik entropii
        vf_coef=0.5,  # Waga funkcji wartości
    )

    # 4. Konfiguracja Callbacka (Auto-Save)
    # Zapisuje model co 100,000 kroków, zabezpieczając przed awarią
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,  # save_freq dotyczy wywołań env.step(), więc dzielimy przez liczbę środowisk
        save_path=CHECKPOINT_DIR,
        name_prefix="galaxian_autosave"
    )

    # 5. Uruchomienie pętli uczenia
    print(f"Rozpoczynam trening na {TOTAL_TIMESTEPS} kroków...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

        # 6. Zapis końcowy modelu
        save_path = f"{MODELS_DIR}/galaxian_bot_v2"
        model.save(save_path)
        print(f"SUKCES! Model zapisany w: {os.path.abspath(save_path)}.zip")

    except KeyboardInterrupt:
        print("\nTrening przerwany przez użytkownika. Zapisuję obecny stan...")
        model.save(f"{MODELS_DIR}/galaxian_interrupted")
        print("Model awaryjny zapisany.")


if __name__ == "__main__":
    main()