"""
PROJEKT: Rozpoznawanie Obiektów na Obrazach Kolorowych (CIFAR-10)
AUTORZY: Kamil Littwitz, Jan Skulimowski

OPIS PROBLEMU:
Projekt wykorzystuje Głębokie Konwolucyjne Sieci Neuronowe (CNN) do klasyfikacji
małych, kolorowych obrazów (32x32 piksele) należących do 10 różnych klas:
samolot, samochód, ptak, kot, jeleń, pies, żaba, koń, statek, ciężarówka.
Jest to klasyczny benchmark sprawdzający zdolność sieci do widzenia kształtów i kolorów.

INSTRUKCJA UŻYCIA:
1. Wymagane biblioteki: tensorflow, numpy, matplotlib, seaborn, sklearn.
2. Przy pierwszym uruchomieniu wymagane jest połączenie z internetem
   (dataset CIFAR-10, ok. 170 MB, zostanie pobrany automatycznie przez Keras).
3. Uruchom skrypt. Program wytrenuje sieć CNN, pokaże wykres procesu uczenia
   oraz wygeneruje macierz pomyłek pokazującą, które obiekty są mylone ze sobą.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Pobranie i przygotowanie danych
# (Pobierze się automatycznie przy pierwszym uruchomieniu)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizacja (0-1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Polskie nazwy klas dla czytelności wykresu
class_names = ['Samolot', 'Samochód', 'Ptak', 'Kot', 'Jeleń',
               'Pies', 'Żaba', 'Koń', 'Statek', 'Ciężarówka']

# 2. Szybka budowa modelu CNN (jeśli nie masz zapisanego)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
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

# Krótki trening (np. 10 epok), aby macierz miała sensowne dane
print("Trwa trening modelu...")
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 3. Generowanie predykcji dla zbioru testowego
print("Obliczanie macierzy pomyłek...")
y_pred_probs = model.predict(test_images)
y_pred = np.argmax(y_pred_probs, axis=1) # Wybieramy klasę z najwyższym prawdopodobieństwem
y_true = test_labels.flatten()           # Spłaszczamy etykiety do wymiaru 1D

# 4. Rysowanie Macierzy Pomyłek
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title('Macierz Pomyłek - CIFAR-10')
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidziana klasa')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"Osiągnięta dokładność: {test_acc:.2%}")