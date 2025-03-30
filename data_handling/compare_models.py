# compare_models.py
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Use the same feature_columns as your NN code:
# Make sure you list EXACTLY the columns that "process_nn" is producing for Q3, Q6, Q8, Q5, etc.
# Example partial:
feature_columns = [
    "Q1","Q2","Q4",
    "Q3_Week day lunch","Q3_Week day dinner","Q3_Weekend lunch","Q3_Weekend dinner","Q3_At a party","Q3_Late night snack",
    "Q6_soda","Q6_milk","Q6_alcohol","Q6_soup", # etc...
    # plus "Q7_Parents","Q7_Siblings","Q7_Friends","Q7_Teachers","Q7_Strangers"
    # plus "Q8_None","Q8_A little (mild)", ...
    # plus your "genre_" columns from Q5
]

############################
# Naive Bayes
############################
def evaluate_naive_bayes():
    import csv
    from models.naive_bayes_model import naive_bayes_map, make_prediction

    with open("../data/bow_processed_nb.csv", encoding="utf-8") as f:
        data_list = list(csv.DictReader(f))

    # Build vocabulary
    vocab_set = set()
    for row in data_list:
        text = row["combined_text"]
        words = text.split()
        vocab_set.update(words)
    vocab = list(vocab_set)

    def make_bow_nb(data_list, vocab):
        N = len(data_list)
        V = len(vocab)
        X = np.zeros((N, V))
        t = np.zeros(N)
        vocab_dict = {w: i for i,w in enumerate(vocab)}
        for i, row in enumerate(data_list):
            review = row["combined_text"].split()
            for word in review:
                if word in vocab_dict:
                    X[i, vocab_dict[word]] = 1
            label = row["Label"]
            if label == "Pizza": t[i] = 0
            elif label == "Shawarma": t[i] = 1
            elif label == "Sushi": t[i] = 2
        return X, t

    # Train/test split (0..820, 820..)
    X_train, t_train = make_bow_nb(data_list[:820], vocab)
    X_test, t_test = make_bow_nb(data_list[820:], vocab)

    pi0, pi1, pi2, theta = naive_bayes_map(X_train, t_train)
    y_pred = make_prediction(X_test, pi0, pi1, pi2, theta)

    print("\n=== Naive Bayes ===")
    print(classification_report(t_test, y_pred, digits=3))
    return "Naive Bayes", accuracy_score(t_test, y_pred), f1_score(t_test, y_pred, average="macro")

############################
# Decision Tree
############################
def evaluate_decision_tree():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # read final_processed_dt.csv
    df = pd.read_csv("../data/final_processed_dt.csv")

    # Build features using the "manual stack" approach
    # (just like the snippet in your code):
    X = np.stack([
        df["Q1"].values,
        df["Q2"].values,
        [1 if "Week day lunch" in str(v) else 0 for v in df["Q3"]],
        [1 if "Week day dinner" in str(v) else 0 for v in df["Q3"]],
        [1 if "Weekend lunch" in str(v) else 0 for v in df["Q3"]],
        [1 if "Weekend dinner" in str(v) else 0 for v in df["Q3"]],
        [1 if "Late night snack" in str(v) else 0 for v in df["Q3"]],
        df["Q4"].values,
        [1 if "Parents" in str(v) else 0 for v in df["Q7"]],
        [1 if "Siblings" in str(v) else 0 for v in df["Q7"]],
        [1 if "Friends" in str(v) else 0 for v in df["Q7"]],
        [1 if "Teachers" in str(v) else 0 for v in df["Q7"]],
        [1 if "Strangers" in str(v) else 0 for v in df["Q7"]],
        df["Q8"].values  # numeric ordinal
    ], axis=1)

    label_map = {"Pizza":0, "Shawarma":1, "Sushi":2}
    y = df["Label"].map(label_map).values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    model = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Decision Tree ===")
    print(classification_report(y_test, y_pred, digits=3))
    return "Decision Tree", accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")

############################
# Neural Network
############################
def evaluate_neural_network():
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from models.neuralnetwork import FoodNeuralNetwork, train_sgd, train_test_split, feature_columns

    df = pd.read_csv("../data/final_processed_nn.csv")

    # label map
    label_map = {"Pizza":0,"Shawarma":1,"Sushi":2}
    df["Label"] = df["Label"].map(label_map)
    df = df.dropna(subset=["Label"])

    # check for missing columns:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print("Missing columns in NN data:", missing)

    # drop rows with NaN in the expected columns
    df = df.dropna(subset=feature_columns)

    X = df[feature_columns].values
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

    model = FoodNeuralNetwork(num_features=X.shape[1], num_hidden=100, num_classes=3)
    train_sgd(model, X_train, y_train, alpha=0.1, n_epochs=200, batch_size=128,
              X_valid=X_test, t_valid=y_test, plot=False)
    y_pred = model.forward(X_test).argmax(axis=1)

    print("\n=== Neural Network ===")
    print(classification_report(y_test, y_pred, digits=3))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy:", acc, "F1 (macro):", f1)
    return "Neural Network", acc, f1

############################
# Main
############################
if __name__ == "__main__":
    # This file presupposes you ALREADY ran data_processing.py to create
    # final_processed_dt.csv, final_processed_nn.csv, bow_processed_nb.csv

    results = [
        evaluate_naive_bayes(),
        evaluate_decision_tree(),
        evaluate_neural_network()
    ]

    print("\n=== Model Comparison ===")
    print("{:<20} {:<10} {:<10}".format("Model", "Accuracy", "F1 (macro)"))
    for name, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{name:<20} {acc:<10.3f} {f1:<10.3f}")

    best_model = max(results, key=lambda x: x[2])[0]
    print(f"\nBest Model Based on F1: {best_model}")
