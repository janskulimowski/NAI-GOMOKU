"""
========================================================================================
TEMAT: Klasyfikacja danych medycznych i jakości wina przy użyciu Drzew Decyzyjnych i SVM
========================================================================================

OPIS PROBLEMU:
Program realizuje zadanie klasyfikacji na dwóch odrębnych zbiorach danych:
1. Heart Disease (klasyfikacja binarna): Przewidywanie obecności choroby serca (0/1).
2. Wine Quality (klasyfikacja wieloklasowa): Ocena jakości wina w skali punktowej (1-10).

Aplikacja przeprowadza kompletny proces uczenia maszynowego:
- Analiza Eksploracyjna Danych (EDA): wizualizacja rozkładu klas i korelacji.
- Trening i Ewaluacja: Porównanie Drzew Decyzyjnych i SVM (RBF) przy użyciu metryk
  takich jak Accuracy, Precision, Recall, F1-Score oraz Macierzy Pomyłek.
- Eksperymenty SVM: Badanie wpływu różnych funkcji jądra (Linear, Poly, RBF, Sigmoid)
  oraz ich parametrów na jakość klasyfikacji.

AUTORZY:
Kamil Littwitz
Jan Skulimowski

INSTRUKCJA UŻYCIA:
1. Wymagane biblioteki: pandas, numpy, matplotlib, seaborn, scikit-learn.
   Instalacja: pip install pandas numpy matplotlib seaborn scikit-learn
2. Pliki danych: Upewnij się, że pliki 'heart_dataset.csv' oraz 'wine_dataset.csv'
   znajdują się w tym samym katalogu co skrypt.
3. Moduły pomocnicze: Skrypt wymaga obecności plików 'visualization.py',
   'evaluation.py' oraz 'experiments.py' w tym samym folderze.
4. Uruchomienie: Wpisz w konsoli: python main.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import visualization as viz
import evaluation as eval_tools
import experiments as exp_tools


def run_full_pipeline(X, y, dataset_name, average_method='binary'):
    # 1. Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Skalowanie (dla głównego SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- CZĘŚĆ A: Standardowe Modele ---
    print(f"\n>>> TRENING STANDARDOWYCH MODELI: {dataset_name} <<<")

    # Drzewo Decyzyjne
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    eval_tools.calculate_metrics(y_test, y_pred_dt, f"Drzewo Decyzyjne ({dataset_name})", average=average_method)
    eval_tools.demonstrate_predictions_detailed(X_test, y_test, dt, None, "Drzewo Decyzyjne")

    # SVM (RBF)
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    eval_tools.calculate_metrics(y_test, y_pred_svm, f"SVM RBF ({dataset_name})", average=average_method)
    eval_tools.demonstrate_predictions_detailed(X_test, y_test, svm, scaler, "SVM RBF")

    # --- CZĘŚĆ B: Eksperymenty SVM ---
    print(f"\n>>> EKSPERYMENTY Z JĄDRAMI DLA: {dataset_name} <<<")
    results_df = exp_tools.run_svm_kernel_experiments(X_train, X_test, y_train, y_test, average_method)
    viz.plot_svm_kernel_results(results_df, dataset_name)
    exp_tools.print_svm_summary()

if __name__ == "__main__":

    # 1. Heart Disease
    try:
        print("\n" + "#" * 60)
        print("ANALIZA ZBIORU: HEART DISEASE")
        heart_df = pd.read_csv('heart_dataset.csv')
        viz.visualize_data_distribution(heart_df, 'target', "Heart Disease")

        run_full_pipeline(
            heart_df.drop('target', axis=1),
            heart_df['target'],
            "Heart Dataset",
            average_method='binary'
        )
    except FileNotFoundError:
        print("Nie znaleziono pliku heart_dataset.csv")

    # 2. Wine Quality
    try:
        print("\n" + "#" * 60)
        print("ANALIZA ZBIORU: WINE QUALITY")
        wine_df = pd.read_csv('wine_dataset.csv', sep=';')
        viz.visualize_data_distribution(wine_df, 'quality', "Wine Quality")

        run_full_pipeline(
            wine_df.drop('quality', axis=1),
            wine_df['quality'],
            "Wine Dataset",
            average_method='weighted'
        )
    except FileNotFoundError:
        print("Nie znaleziono pliku wine_dataset.csv")