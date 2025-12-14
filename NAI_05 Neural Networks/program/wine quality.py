"""
PROJEKT: Klasyfikacja Jakości Wina (Wine Quality Prediction)
AUTORZY: Kamil Littwitz, Jan Skulimowski

OPIS PROBLEMU:
Celem projektu jest porównanie skuteczności różnych algorytmów uczenia maszynowego
(Drzewo Decyzyjne, SVM, Sieci Neuronowe) w zadaniu klasyfikacji jakości wina.
Model analizuje parametry fizykochemiczne (np. kwasowość, poziom cukru, alkohol)
i przypisuje winu ocenę punktową (klasę jakości).
Porównujemy również wpływ rozmiaru sieci neuronowej (mała vs duża) na wynik końcowy.

INSTRUKCJA UŻYCIA:
1. Upewnij się, że plik z danymi 'wine_dataset.csv' znajduje się w tym samym folderze co skrypt.
2. Wymagane biblioteki: pandas, numpy, sklearn, tensorflow, matplotlib, seaborn.
3. Uruchom skrypt. Program automatycznie:
   - Wczyta i oczyści dane.
   - Wytrenuje trzy rodzaje modeli.
   - Wyświetli porównanie dokładności oraz macierze pomyłek (confusion matrix).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# 1. Load Data
try:
    df = pd.read_csv('wine_dataset.csv', sep=';')
except:
    df = pd.read_csv('wine_dataset.csv')

# Cleanup
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

X = df.drop('quality', axis=1)
y = df['quality']

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODELS ---

# 1. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
acc_dt = accuracy_score(y_test, y_pred_dt)

# 2. SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)

# 3. Neural Network
nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn.fit(X_train_scaled, y_train, epochs=50, verbose=0)
loss_nn, acc_nn = nn.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_probs = nn.predict(X_test_scaled)
y_pred_nn = np.argmax(y_pred_probs, axis=1)

# --- RESULTS ---
print(f"Dokładność Drzewa Decyzyjnego: {acc_dt:.2%}")
print(f"Dokładność SVM: {acc_svm:.2%}")
print(f"Dokładność Sieci Neuronowej: {acc_nn:.2%}")

# Bar Chart
models = ['Drzewo Decyzyjne', 'SVM', 'Sieć Neuronowa']
accuracies = [acc_dt, acc_svm, acc_nn]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'orange', 'green'])
plt.ylabel('Dokładność')
plt.title('Porównanie skuteczności modeli (Jakość Wina)')
plt.ylim(0, 1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom')
plt.savefig('comparison_chart.png')

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Drzewo Decyzyjne')
axes[0].set_xlabel('Przewidziana')
axes[0].set_ylabel('Prawdziwa')

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title('SVM')
axes[1].set_xlabel('Przewidziana')
axes[1].set_ylabel('Prawdziwa')

sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', cmap='Greens', ax=axes[2])
axes[2].set_title('Sieć Neuronowa')
axes[2].set_xlabel('Przewidziana')
axes[2].set_ylabel('Prawdziwa')

plt.savefig('confusion_matrices_comparison.png')