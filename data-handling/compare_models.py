import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

# === Naive Bayes ===

def evaluate_naive_bayes():
    import csv
    from models.naive_bayes_model import make_bow, naive_bayes_map, make_prediction
    with open("data/worded_data.csv") as f:
        data_list = list(csv.reader(f))[1:]
    np.random.seed(42)
    np.random.shuffle(data_list)

    # Build vocab
    vocab_set = set()
    for row in data_list:
        words = [word for sublist in row[1:9] for word in sublist.split(",")]
        vocab_set.update(words)
    vocab = list(vocab_set)

    # Make BoW
    X_train, t_train = make_bow(data_list[:820], vocab)
    X_test, t_test = make_bow(data_list[820:], vocab)

    # Train and predict
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

# === Neural Network ===


def evaluate_neural_network():
    from models.neuralnetwork import FoodNeuralNetwork, train_sgd, train_test_split
    import pandas as pd

    df = pd.read_csv("data/final_processed.csv")
    X = df.drop(columns=["Label"]).values.astype(int)
    y = df["Label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    model = FoodNeuralNetwork(num_features=X.shape[1], num_hidden=100, num_classes=3)
    train_sgd(model, X_train, y_train, alpha=0.1, n_epochs=200, batch_size=128, X_valid=X_test, t_valid=y_test, plot=False)

    y_pred = model.forward(X_test).argmax(axis=1)

    print("\n=== Neural Network ===")
    print(classification_report(y_test, y_pred, digits=3))
    return "Neural Network", accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")

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