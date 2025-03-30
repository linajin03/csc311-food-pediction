import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from models.neuralnetwork import feature_columns  # This list should match the NN cleaned file


# === Naive Bayes ===
def evaluate_naive_bayes():
    import csv
    from models.naive_bayes_model import naive_bayes_map, make_prediction

    # Use the NB branch processed file (e.g. bow_processed_nb.csv)
    with open("../data/bow_processed_nb.csv") as f:
        reader = csv.DictReader(f)
        data_list = list(reader)

    # Build vocabulary from the combined_text field
    vocab_set = set()
    for row in data_list:
        text = row["combined_text"]
        words = text.split()  # simple split; use a better tokenizer if desired
        vocab_set.update(words)
    vocab = list(vocab_set)

    # Function to build a bag-of-words representation for NB
    def make_bow_nb(data_list, vocab):
        N = len(data_list)
        V = len(vocab)
        X = np.zeros((N, V))
        t = np.zeros(N)
        vocab_dict = {word: idx for idx, word in enumerate(vocab)}
        for i, row in enumerate(data_list):
            review = row["combined_text"].split()
            for word in review:
                if word in vocab_dict:
                    X[i, vocab_dict[word]] = 1
            # Map labels (assuming they are stored as strings "Pizza", "Shawarma", "Sushi")
            label = row["Label"]
            if label == "Pizza":
                t[i] = 0
            elif label == "Shawarma":
                t[i] = 1
            elif label == "Sushi":
                t[i] = 2
        return X, t

    X_train, t_train = make_bow_nb(data_list[:820], vocab)
    X_test, t_test = make_bow_nb(data_list[820:], vocab)

    pi0, pi1, pi2, theta = naive_bayes_map(X_train, t_train)
    y_pred = make_prediction(X_test, pi0, pi1, pi2, theta)

    print("\n=== Naive Bayes ===")
    print(classification_report(t_test, y_pred, digits=3))
    return "Naive Bayes", accuracy_score(t_test, y_pred), f1_score(t_test, y_pred, average="macro")


# === Decision Tree ===
def evaluate_decision_tree():
    from models.decisiontrees import data_fets
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X = data_fets[:, :-1]
    y = data_fets[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Decision Tree ===")
    print(classification_report(y_test, y_pred, digits=3))
    return "Decision Tree", accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")

def evaluate_neural_network():
    from models.neuralnetwork import FoodNeuralNetwork, train_sgd, train_test_split
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    df = pd.read_csv("../data/final_processed_nn.csv")
    # Suppose your CSV's label is "Label" with string classes (Pizza, Shawarma, Sushi).
    label_map = {"Pizza": 0, "Shawarma": 1, "Sushi": 2}
    df["Label"] = df["Label"].map(label_map)
    df = df.dropna(subset=["Label"])  # remove any rows missing label
    df["Label"] = df["Label"].astype(int)

    # Make sure we have all feature columns:
    # remove rows that have NaN in any feature column
    df = df.dropna(subset=feature_columns)

    # Instead of forcing int, just do:
    X = df[feature_columns].values  # remain float if Q2/Q4 were normalized
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = FoodNeuralNetwork(num_features=X.shape[1], num_hidden=100, num_classes=3)
    train_sgd(
        model, X_train, y_train,
        alpha=0.1, n_epochs=200, batch_size=128,
        X_valid=X_test, t_valid=y_test,
        plot=False
    )
    y_pred = model.forward(X_test).argmax(axis=1)

    print("\n=== Neural Network ===")
    print(classification_report(y_test, y_pred, digits=3))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.3f}, Macro-F1: {f1:.3f}")
    return "Neural Network", acc, f1


# === Run All Models ===
if __name__ == "__main__":
    results = [
        evaluate_naive_bayes(),
        evaluate_decision_tree(),
        evaluate_neural_network()
    ]

    print("\n=== Model Comparison ===")
    print("{:<20} {:<10} {:<10}".format("Model", "Accuracy", "F1 (macro)"))
    for name, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{name:<20} {acc:<10.3f} {f1:<10.3f}")

    print(f"\nBest Model Based on F1: {max(results, key=lambda x: x[2])[0]}")
