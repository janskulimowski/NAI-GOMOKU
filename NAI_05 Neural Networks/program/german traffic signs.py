"""
PROJEKT: Rozpoznawanie Znaków Drogowych (GTSRB - German Traffic Sign Recognition Benchmark)
AUTORZY: Kamil Littwitz, Jan Skulimowski

OPIS PROBLEMU:
Najbardziej zaawansowany projekt w zestawieniu. Polega na klasyfikacji 43 różnych
typów znaków drogowych na podstawie rzeczywistych zdjęć o różnej jakości i oświetleniu.
Wymaga zastosowania głębszej sieci neuronowej, mechanizmów regularyzacji (Dropout)
oraz wstępnego przetwarzania obrazów (skalowanie do 30x30 pikseli).

INSTRUKCJA UŻYCIA:
1. Pobierz zbiór danych GTSRB (np. z Kaggle) i rozpakuj go.
2. W kodzie, w zmiennej 'data_dir', podaj ścieżkę do rozpakowanego folderu.
   Struktura musi zawierać folder 'Train' z podfolderami '0', '1'... oraz plik 'Test.csv'.
3. Wymagane biblioteki: tensorflow, pandas, numpy, matplotlib, seaborn, PIL (Pillow), sklearn.
4. Uruchom skrypt. Może on działać dłużej ze względu na dużą liczbę zdjęć.
   Po treningu zostanie wyświetlona ostateczna dokładność oraz wielka macierz pomyłek dla 43 znaków.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix
import os

# Słownik nazw znaków (dla czytelności wykresu)
classes = {
    0:'Ogr. prędkości (20km/h)', 1:'Ogr. prędkości (30km/h)',
    2:'Ogr. prędkości (50km/h)', 3:'Ogr. prędkości (60km/h)',
    4:'Ogr. prędkości (70km/h)', 5:'Ogr. prędkości (80km/h)',
    6:'Koniec ogr. (80km/h)', 7:'Ogr. prędkości (100km/h)',
    8:'Ogr. prędkości (120km/h)', 9:'Zakaz wyprzedzania',
    10:'Zakaz wyprzedzania (ciężarowe)', 11:'Pierwszeństwo na skrzyżowaniu',
    12:'Droga z pierwszeństwem', 13:'Ustąp pierwszeństwa',
    14:'STOP', 15:'Zakaz ruchu',
    16:'Zakaz wjazdu (ciężarowe)', 17:'Zakaz wjazdu',
    18:'Inne niebezpieczeństwo', 19:'Niebezpieczny zakręt w lewo',
    20:'Niebezpieczny zakręt w prawo', 21:'Podwójny zakręt',
    22:'Nierówna droga', 23:'Śliska jezdnia',
    24:'Zwężenie jezdni (prawa)', 25:'Roboty drogowe',
    26:'Sygnalizacja świetlna', 27:'Przejście dla pieszych',
    28:'Dzieci', 29:'Rowerzyści',
    30:'Oszronienie/Lód', 31:'Dzikie zwierzęta',
    32:'Koniec ograniczeń', 33:'Nakaz skrętu w prawo',
    34:'Nakaz skrętu w lewo', 35:'Nakaz prosto',
    36:'Nakaz prosto lub w prawo', 37:'Nakaz prosto lub w lewo',
    38:'Nakaz objazdu z prawej', 39:'Nakaz objazdu z lewej',
    40:'Rondo', 41:'Koniec zakazu wyprzedzania',
    42:'Koniec zakazu wyprz. (ciężarowe)'
}

# USTAWIENIA (Muszą być takie same jak przy treningu)
IMG_HEIGHT = 30
IMG_WIDTH = 30
data_dir = 'gtsrb'  # np. 'gtsrb'

# 1. WCZYTANIE MODELU (jeśli nie masz go w pamięci)
model = tf.keras.models.load_model('traffic_classifier.h5')

# 2. PRZYGOTOWANIE DANYCH TESTOWYCH
# W GTSRB plik Test.csv zawiera ścieżki do plików testowych
y_test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))

labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data = []

print("Wczytywanie i przetwarzanie obrazów testowych...")
for img in imgs:
    try:
        image = Image.open(os.path.join(data_dir, img))
        image = image.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(image))
    except:
        print(f"Błąd przy: {img}")

X_test = np.array(data)
X_test = X_test / 255.0  # Normalizacja

# 3. PREDYKCJA
print("Generowanie predykcji...")
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)

# 4. RYSOWANIE MACIERZY POMYŁEK
# Pobieramy nazwy klas w odpowiedniej kolejności
class_names = [classes[i] for i in range(43)]

cm = confusion_matrix(labels, pred_classes)

plt.figure(figsize=(25, 25)) # BARDZO DUŻY ROZMIAR, bo 43 klasy
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False) # Ukrywamy pasek legendy, żeby było więcej miejsca

plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title('Macierz Pomyłek - Znaki Drogowe (GTSRB)', fontsize=20)
plt.ylabel('Prawdziwy znak', fontsize=15)
plt.xlabel('Przewidziany znak', fontsize=15)

plt.tight_layout()
plt.savefig('confusion_matrix_gtsrb.png')
plt.show()