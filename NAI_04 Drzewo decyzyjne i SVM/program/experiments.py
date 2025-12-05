# experiments.py
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def run_svm_kernel_experiments(X_train, X_test, y_train, y_test, average_method):
    """
    Uruchamia serię testów dla różnych jąder SVM: Linear, Poly, RBF, Sigmoid.
    Zwraca DataFrame z wynikami.
    """
    # Skalowanie jest kluczowe dla porównania jąder
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lista konfiguracji
    configs = [
        {'label': 'Linear (C=0.01)', 'model': SVC(kernel='linear', C=0.01)},
        {'label': 'Linear (C=1.0)', 'model': SVC(kernel='linear', C=1.0)},
        {'label': 'Poly (d=2, C=1)', 'model': SVC(kernel='poly', degree=2, C=1)},
        {'label': 'Poly (d=3, C=1)', 'model': SVC(kernel='poly', degree=3, C=1)},
        {'label': 'RBF (g=0.1, C=1)', 'model': SVC(kernel='rbf', gamma=0.1, C=1)},
        {'label': 'RBF (g=1.0, C=1)', 'model': SVC(kernel='rbf', gamma=1.0, C=1)},
        {'label': 'Sigmoid (C=1)', 'model': SVC(kernel='sigmoid', C=1)}
    ]

    results = []
    print(f"\n--- EKSPERYMENTY Z JĄDRAMI SVM ---")
    print(f"{'Konfiguracja':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 45)

    for config in configs:
        model = config['model']
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)

        results.append({'label': config['label'], 'accuracy': acc, 'f1': f1})
        print(f"{config['label']:<20} | {acc:.4f}     | {f1:.4f}")

    return pd.DataFrame(results)


def print_svm_summary():
    """Wypisuje wnioski edukacyjne."""
    summary = """
    PODSUMOWANIE WPŁYWU JĄDER (KERNELS):
    1. Linear: Idealne dla prostych danych. Szybkie, ale ograniczone do linii prostych/płaszczyzn.
    2. Poly: Tworzy krzywe. Wysoki stopień wielomianu (degree) drastycznie wydłuża czas i grozi przeuczeniem.
    3. RBF (Radial Basis Function): Najpotężniejsze jądro ogólnego przeznaczenia.
       Parametr Gamma decyduje o "zasięgu" punktu - zbyt duża Gamma = "pamiętanie" danych (overfitting).
    4. Sigmoid: Zachowuje się jak sieć neuronowa. Często trudne w treningu.
    """
    print(summary)