import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Wczytanie i przygotowanie danych (wine_dataset.csv)
try:
    df = pd.read_csv('wine_dataset.csv', sep=';')
except:
    df = pd.read_csv('wine_dataset.csv', sep=',')

# Proste czyszczenie
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)
df = df.dropna()

X = df.drop('quality', axis=1)
y = df['quality']

# Kodowanie etykiet
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Podział i skalowanie
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Definicja modeli

# MODEL MAŁY (Płytka sieć)
model_small = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(num_classes, activation='softmax')
])
model_small.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# MODEL DUŻY (Głęboka sieć)
model_large = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model_large.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Trening
epochs = 50
print("Trenowanie małego modelu...")
history_small = model_small.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)
print("Trenowanie dużego modelu...")
history_large = model_large.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, verbose=0)

# 4. Porównanie wyników
loss_s, acc_s = model_small.evaluate(X_test_scaled, y_test, verbose=0)
loss_l, acc_l = model_large.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nWyniki na zbiorze testowym:")
print(f"Mały model - Dokładność: {acc_s:.2%}, Strata: {loss_s:.4f}")
print(f"Duży model - Dokładność: {acc_l:.2%}, Strata: {loss_l:.4f}")

# Wizualizacja
plt.figure(figsize=(14, 5))

# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(history_small.history['val_accuracy'], label='Mały Model (Val)', linestyle='--')
plt.plot(history_large.history['val_accuracy'], label='Duży Model (Val)')
plt.title('Porównanie Dokładności (Validation Accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

# Wykres straty
plt.subplot(1, 2, 2)
plt.plot(history_small.history['val_loss'], label='Mały Model (Val)', linestyle='--')
plt.plot(history_large.history['val_loss'], label='Duży Model (Val)')
plt.title('Porównanie Straty (Validation Loss)')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.savefig('model_comparison.png')