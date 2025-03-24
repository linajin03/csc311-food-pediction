import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

path_to_data = "../data/cleaned_data.csv"

# Load the cleaned dataset
data = pd.read_csv(path_to_data)

data["Q4"] = data["Q4"].astype(str).str.replace("$", "", regex=False)
data["Q4"] = pd.to_numeric(data["Q4"], errors="coerce")
data["Q1"] = pd.to_numeric(data["Q1"], errors="coerce")
data["Q2"] = pd.to_numeric(data["Q2"], errors="coerce")
data["Q8"] = pd.to_numeric(data["Q8"], errors="coerce")


# === Manually construct basic features ===
data_fets = np.stack([
    data["Q1"],
    data["Q2"],
    [1 if "Week day lunch" in str(item) else 0 for item in data["Q3"]],
    [1 if "Week day dinner" in str(item) else 0 for item in data["Q3"]],
    [1 if "Weekend lunch" in str(item) else 0 for item in data["Q3"]],
    [1 if "Weekend dinner" in str(item) else 0 for item in data["Q3"]],
    [1 if "Late night snack" in str(item) else 0 for item in data["Q3"]],
    data["Q4"],
    [1 if "Parents" in str(item) else 0 for item in data["Q7"]],
    [1 if "Siblings" in str(item) else 0 for item in data["Q7"]],
    [1 if "Friends" in str(item) else 0 for item in data["Q7"]],
    [1 if "Teachers" in str(item) else 0 for item in data["Q7"]],
    [1 if "Strangers" in str(item) else 0 for item in data["Q7"]],
    data["Q8"],
    data["Label"]
], axis=1)

# For use in compare_models.py
__all__ = ["data_fets"]

# === Train/test split ===
X = data_fets[:, :-1]
y = data_fets[:, -1]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Train model ===
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=20)
clf.fit(X_train, y_train)

# === Evaluate ===
print(f"Train Accuracy: {accuracy_score(y_train, clf.predict(X_train)):.3f}")
print(f"Val Accuracy: {accuracy_score(y_val, clf.predict(X_val)):.3f}")
print(f"Test Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")

# === Hyperparameter Tuning ===
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Test Accuracy with Best Model: {accuracy_score(y_test, best_model.predict(X_test)):.3f}")

# === Visualization ===
def visualize_tree(model, out_file="tree.dot", max_depth=5):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=[
            "Q1", "Q2", "wdl", "wdd", "wel", "wed", "lns", "Q4",
            "Parents", "Siblings", "Friends", "Teachers", "Strangers", "Q8"
        ],
        class_names=["Pizza", "Shawarma", "Sushi"],
        filled=True,
        rounded=True,
        max_depth=max_depth
    )
    graph = graphviz.Source(dot_data)
    graph.render(out_file.replace(".dot", ""), format="pdf", cleanup=True)
    print(f"Saved decision tree visualization as {out_file.replace('.dot', '.pdf')}")