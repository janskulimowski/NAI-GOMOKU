# evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Importujemy funkcję wizualizacji z naszego innego pliku
from visualization import plot_confusion_matrix_heatmap


def calculate_metrics(y_true, y_pred, model_name, average='binary'):
    """
    Oblicza, wypisuje metryki i wywołuje rysowanie macierzy pomyłek.
    """
    print(f"\n{'=' * 10} WYNIKI: {model_name} {'=' * 10}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1-Score:       {f1:.4f}")

    # Wywołanie wizualizacji
    plot_confusion_matrix_heatmap(y_true, y_pred, model_name)


def demonstrate_predictions_detailed(X_test, y_test, model, scaler=None, model_name="Model", num_samples=3):
    """
    Pokazuje szczegółowe dane wejściowe i decyzję modelu dla losowych próbek.
    """
    print(f"\n--- SZCZEGÓŁOWA DEMONSTRACJA: {model_name} ---")

    available_samples = len(X_test)
    n = min(num_samples, available_samples)

    # Losowanie
    indices = np.random.choice(X_test.index, size=n, replace=False)
    X_samples = X_test.loc[indices]
    y_true = y_test.loc[indices]

    # Skalowanie (jeśli wymagane, np. dla SVM)
    if scaler:
        X_input = scaler.transform(X_samples)
    else:
        X_input = X_samples

    y_pred = model.predict(X_input)
    feature_names = X_test.columns

    for i in range(n):
        print(f"\nPRÓBKA #{i + 1}:")
        row_values = X_samples.iloc[i]

        # Ładne formatowanie cech
        features_str = ""
        for idx, (col, val) in enumerate(zip(feature_names, row_values)):
            features_str += f"{col[:10]}: {val:<8} "
            if (idx + 1) % 3 == 0:
                print(f"    {features_str}")
                features_str = ""
        if features_str:
            print(f"    {features_str}")

        # Wynik
        true_val = y_true.iloc[i]
        pred_val = y_pred[i]
        status = "✅ POPRAWNA" if true_val == pred_val else "❌ BŁĘDNA"
        print(f"  -> Rzeczywista: {true_val} | Predykcja: {pred_val} [{status}]")
    print("-" * 40)