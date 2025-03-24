import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# === Load data ===
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Label", "Q5_original"])
    y = df["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# === Evaluation ===
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"../plots/cm_{name.lower().replace(' ', '_')}.png", bbox_inches="tight")
    plt.close()

    return y_pred

# === ROC plot (for binary case only) ===
def plot_roc_curves(models, X_test, y_test, label_names):
    if len(np.unique(y_test)) > 2:
        print("Skipping ROC curve (not binary classification).")
        return

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary Only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../plots/roc_curves.png", bbox_inches="tight")
    plt.close()

# === Main ===
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("../data/final_processed.csv")

    # Optional: detect class names
    label_names = sorted(y_train.unique())

    # Fit models
    models = {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=0)
    }

    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = evaluate_model(name, model, X_test, y_test)

    # Optional ROC/AUC
    plot_roc_curves(models, X_test, y_test, label_names)

    from sklearn.metrics import accuracy_score, f1_score

    # === Compare all models numerically
    print("\n===== MODEL COMPARISON =====")
    scores = []
    for name, y_pred in predictions.items():
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        scores.append((name, acc, f1_macro))

    scores.sort(key=lambda x: x[2], reverse=True)  # sort by F1-macro

    print("\nModel Performance Summary (sorted by macro F1):")
    print("{:<20s} {:<10s} {:<10s}".format("Model", "Accuracy", "F1 (macro)"))
    print("-" * 42)
    for name, acc, f1 in scores:
        print(f"{name:<20s} {acc:<10.3f} {f1:<10.3f}")

    best_model = scores[0][0]
    print(f"\nRecommended model based on F1-macro: {best_model}")

