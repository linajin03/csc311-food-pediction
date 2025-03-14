import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('data-handling', 'data')
os.chdir(dir)
df = pd.read_csv("multi_hot_encoded_df.csv")

print("===== BEFORE FILLNA =====")
print(df.info())

# Separate columns by type
numeric_cols = df.select_dtypes(include=["number"]).columns
object_cols = df.select_dtypes(exclude=["number"]).columns

for col in numeric_cols:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)

for col in object_cols:
    df[col] = df[col].fillna("Unknown")


print("===== AFTER FILLNA =====")
print(df.info())

# Now do label encoding of any remaining object columns
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoder if needed later

# Normalize numerical features
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Train-Test Split
X = df.drop(columns=['Label'])  # or whichever columns you consider the target
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Save processed data for model training
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("Y_train.csv", index=False)
y_test.to_csv("Y_test.csv", index=False)