"""
PROJEKT: Klasyfikacja Odzieży (Fashion-MNIST)
AUTORZY: Kamil Littwitz, Jan Skulimowski

OPIS PROBLEMU:
Zadanie polega na rozpoznawaniu typów ubrań na podstawie obrazów w skali szarości (28x28 pikseli).
Zbiór danych zastępuje klasyczne cyfry (MNIST) bardziej złożonymi kształtami.
Klasy to: T-shirt/Top, Spodnie, Sweter, Sukienka, Płaszcz, Sandał, Koszula, Trampek, Torba, Botek.
Projekt wymaga odpowiedniego przygotowania danych (reshape) dla sieci CNN pracującej na 1 kanale koloru.

INSTRUKCJA UŻYCIA:
1. Wymagane biblioteki: tensorflow, numpy, matplotlib, seaborn, sklearn.
2. Zbiór danych jest wbudowany w bibliotekę TensorFlow/Keras i pobierze się automatycznie.
3. Uruchom skrypt. Zobaczysz proces treningu oraz wizualizację predykcji
   wraz z macierzą pomyłek dla poszczególnych elementów garderoby.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. WCZYTANIE DANYCH
print("Pobieranie danych Fashion-MNIST...")
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Nazwy klas w języku polskim
class_names = ['Koszulka/Top', 'Spodnie', 'Sweter', 'Sukienka', 'Płaszcz',
               'Sandał', 'Koszula', 'Trampek', 'Torba', 'Botek']

# 2. PRZYGOTOWANIE DANYCH
# Normalizacja (0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Zmiana kształtu dla sieci CNN (dodanie 1 kanału koloru - szarość)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 3. BUDOWA MODELU
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRENING
print("Rozpoczynam trening...")
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 5. GENEROWANIE MACIERZY POMYŁEK
print("Tworzenie macierzy pomyłek...")
# Predykcja dla całego zbioru testowego
y_pred_probs = model.predict(test_images)
y_pred = np.argmax(y_pred_probs, axis=1)

# Obliczenie macierzy
cm = confusion_matrix(test_labels, y_pred)

# 6. RYSOWANIE WYKRESU
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title('Macierz Pomyłek - Rozpoznawanie Ubrań')
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidziana klasa')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"Osiągnięta dokładność: {test_acc:.2%}")