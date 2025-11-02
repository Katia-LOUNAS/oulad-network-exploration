import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

def summarize_datasets(datasets):
    """
    Explore multiple datasets (OULAD CSVs).
    For each dataset:
      - Show basic info (rows, columns)
      - Count NaN values per column
    Optionally shows a heatmap comparing missing values across datasets.

    Parameters:
    -----------
    datasets : dict
        Dictionary of name: DataFrame, e.g.
        {"studentInfo": studentInfo, "courses": courses}
    show_plot : bool
        If True, displays a heatmap of missing values.
    """

    summaries = []

    # --- Build summary table ---
    for name, df in datasets.items():
        total_rows, total_cols = df.shape
        total_nans = df.isna().sum().sum()
        nan_ratio = (total_nans / (total_rows * total_cols)) * 100

        summary = {
            "Dataset": name,
            "Rows": total_rows,
            "Columns": total_cols,
            "Total NaN": total_nans,
            "% NaN": round(nan_ratio, 2)
        }
        summaries.append(summary)

        print(f"\n {name}")
        print("-" * (len(name) + 3))
        print(f"Shape: {total_rows} rows × {total_cols} cols")
        print("Missing values per column:")
        print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

    summary_df = pd.DataFrame(summaries).set_index("Dataset")

    return summary_df


def global_summary(summary_df):
    
    display(summary_df)


    plt.figure(figsize=(8, 4))
    sns.barplot(
        x=summary_df.index,
        y=summary_df["% NaN"],
        palette="Blues_d"
    )
    plt.title("Percentage of Missing Values per Dataset")
    plt.ylabel("% NaN")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
