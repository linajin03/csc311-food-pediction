import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('data-handling', 'data')
os.chdir(dir)
df = pd.read_csv("cleaned_data.csv")

# Display basic information
print(df.info())
print(df.describe())

# Handle missing values
df.fillna("Unknown", inplace=True)

# Encode categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoder for later decoding

# Normalize numerical features
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Train-Test Split
X = df.drop(columns=['Label'])  # Replace 'Label' with actual target column name
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data for model training
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
