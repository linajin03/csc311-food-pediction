import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_csv(path)
    return df

def summarize(df):
    print("===== DATAFRAME INFO =====")
    print(df.info())
    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())
    print("\n===== BASIC STATS =====")
    print(df.describe(include="all"))
    print("\n===== CLASS DISTRIBUTION (Label) =====")
    print(df["Label"].value_counts())

def correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    path = "../data/final_processed.csv"
    df = load_data(path)

    summarize(df)

    # Minimal boxplot for core features
    core_numeric = ["Q1", "Q2", "Q4"]
    df_melted = df.melt(id_vars="Label", value_vars=core_numeric)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_melted, x="variable", y="value", hue="Label")
    plt.title("Boxplots of Q1–Q4 by Label")
    plt.tight_layout()
    plt.savefig("../plots/boxplot_q1_q4.png", bbox_inches="tight")

    # One grouped bar plot for each Q3–Q8 group
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
        plot_name = f"../plots/barplot_{prefix[:-1]}.png"
        plt.savefig(plot_name, bbox_inches="tight")

    # Correlation heatmap (subset)
    corr_cols = core_numeric + [c for c in df.columns if c.startswith(("Q3_", "Q6_", "Q7_", "Q8_"))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("../plots/correlation_heatmap.png", bbox_inches="tight")


