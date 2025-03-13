import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load final numeric data (adjust filename as needed)
df = pd.read_csv("/mnt/data/multi_hot_encoded_df.csv")

# 2. Basic info
print(df.info())
print(df.head())

# 3. Distribution of Labels (assuming "Label" is numeric after encoding)
plt.figure(figsize=(8, 6))
sns.countplot(x='Label', data=df)
plt.title("Distribution of Food Categories")
plt.xticks(rotation=45)
plt.show()

# 4. Correlation Matrix (all numeric columns)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Boxplot for numerical features
numerical_columns = df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(10, 6))
df[numerical_columns].boxplot(rot=45)
plt.title("Boxplot of Numerical Features")
plt.show()
