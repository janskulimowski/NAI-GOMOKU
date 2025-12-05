# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ustawienie stylu dla wszystkich wykresów
sns.set(style="whitegrid")


def visualize_data_distribution(df, target_col, dataset_name):
    """
    Rysuje rozkład klas (czy zbiór jest zbalansowany) oraz mapę korelacji.
    """
    print(f"\n--- Generowanie wykresów dla: {dataset_name} ---")

    # 1. Rozkład klas
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df, hue=target_col, legend=False, palette='viridis')
    plt.title(f'Rozkład klas w zbiorze: {dataset_name}')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba próbek')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name):
    """
    Rysuje macierz pomyłek jako mapę ciepła.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Macierz Pomyłek: {model_name}')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Rzeczywista klasa')
    plt.tight_layout()
    plt.show()


def plot_svm_kernel_results(results_df, dataset_name):
    """
    Rysuje wykres słupkowy porównujący wyniki różnych jąder SVM.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='accuracy', y='label', data=results_df, hue='label', legend=False, palette='viridis')
    plt.title(f'Porównanie Jąder SVM: {dataset_name}')
    plt.xlabel('Accuracy')
    plt.ylabel('Konfiguracja SVM')
    plt.xlim(0.0, 1.0)

    # Linia max accuracy
    max_acc = results_df['accuracy'].max()
    plt.axvline(x=max_acc, color='red', linestyle='--', alpha=0.5, label=f'Max: {max_acc:.2f}')

    plt.legend()
    plt.tight_layout()
    plt.show()