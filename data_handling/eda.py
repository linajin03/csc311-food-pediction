import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(path):
    return pd.read_csv(path)

def summarize(df):
    print("===== DATAFRAME INFO =====")
    print(df.info())
    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())
    print("\n===== BASIC STATS =====")
    print(df.describe(include="all"))
    print("\n===== CLASS DISTRIBUTION (Label) =====")
    print(df["Label"].value_counts())

def plot_structured_eda(df, filename_suffix=""):
    # Boxplot for Q1–Q4
    core_numeric = ["Q1", "Q2", "Q4"]
    df_melted = df.melt(id_vars="Label", value_vars=core_numeric)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_melted, x="variable", y="value", hue="Label")
    plt.title("Boxplots of Q1–Q4 by Label")
    plt.tight_layout()
    plt.savefig(f"../plots/boxplot_q1_q4{filename_suffix}.png", bbox_inches="tight")

    # Grouped bar plots
    for prefix in ["Q3_", "Q6_", "Q7_", "Q8_"]:
        group_cols = [col for col in df.columns if col.startswith(prefix)]
        if not group_cols:
            continue
        plt.figure(figsize=(8, max(3, len(group_cols) * 0.3)))
        counts = df[group_cols].sum().sort_values()
        sns.barplot(x=counts.values, y=counts.index, palette="viridis")
        plt.title(f"{prefix[:-1]} Response Totals")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(f"../plots/barplot_{prefix[:-1].lower()}{filename_suffix}.png", bbox_inches="tight")

    # Correlation heatmap (subset of structured features)
    corr_cols = core_numeric + [c for c in df.columns if c.startswith(("Q3_", "Q6_", "Q7_", "Q8_"))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"../plots/correlation_heatmap{filename_suffix}.png", bbox_inches="tight")

def plot_bow_summary(df, filename_suffix=""):
    if "combined_text" in df.columns:
        # Word length distribution
        df["num_words"] = df["combined_text"].str.split().apply(len)
        plt.figure(figsize=(7, 5))
        sns.histplot(data=df, x="num_words", bins=30, kde=True)
        plt.title("Distribution of Word Counts in BoW Inputs")
        plt.xlabel("Number of Words")
        plt.tight_layout()
        plt.savefig(f"../plots/bow_wordcount{filename_suffix}.png", bbox_inches="tight")

if __name__ == "__main__":
    for path in ["../data/final_processed.csv", "../data/bow_processed.csv"]:
        print(f"\n=== Running EDA on {path} ===")
        df = load_data(path)
        summarize(df)

        suffix = f"_{os.path.splitext(os.path.basename(path))[0]}"

        if "combined_text" in df.columns:
            plot_bow_summary(df, filename_suffix=suffix)
        else:
            plot_structured_eda(df, filename_suffix=suffix)