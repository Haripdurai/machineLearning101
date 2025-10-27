import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns

"""
simple.py

Quick EDA for Retail_sales.csv
Saves a text report and several plots under ./eda_outputs/
"""

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_data(path="Retail_sales.csv"):
    # Try to read with a few common encodings if default fails
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    return df


def detect_date_column(df):
    common = ["date", "Date", "DATE", "invoice_date", "InvoiceDate", "order_date", "OrderDate"]
    for col in df.columns:
        if col in common:
            return col
    # fuzzy detect by dtype or name containing 'date' or 'day'
    for col in df.columns:
        if "date" in col.lower() or "day" in col.lower():
            return col
    # none found
    return None


def basic_report(df, out_txt_path):
    with open(out_txt_path, "w", encoding="utf8") as f:
        f.write("Basic EDA Report\n")
        f.write("================\n\n")
        f.write("Shape: {}\n\n".format(df.shape))
        f.write("Columns and dtypes:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nMemory usage (MB): {:.2f}\n\n".format(df.memory_usage(deep=True).sum() / 1024 ** 2))
        f.write("Top 5 rows:\n")
        f.write(df.head().to_string(index=False))
        f.write("\n\nMissing values (count):\n")
        f.write(df.isna().sum().to_string())
        f.write("\n\nMissing values (percent):\n")
        f.write((df.isna().mean() * 100).round(2).to_string())
        f.write("\n\nDescriptive statistics (numeric):\n")
        f.write(df.describe().to_string())
        f.write("\n\nDescriptive statistics (object):\n")
        f.write(df.describe(include=["object", "category"]).to_string())


def plot_numeric_distributions(df, numeric_cols, prefix="numeric"):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=40)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        fn = OUTPUT_DIR / f"{prefix}_hist_{col}.png"
        plt.savefig(fn)
        plt.close()

        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        fn = OUTPUT_DIR / f"{prefix}_box_{col}.png"
        plt.savefig(fn)
        plt.close()


def plot_correlation(df, numeric_cols):
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation matrix")
    plt.tight_layout()
    fn = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(fn)
    plt.close()


def categorical_counts(df, cat_cols, top_n=10):
    out_path = OUTPUT_DIR / "categorical_value_counts.txt"
    with open(out_path, "w", encoding="utf8") as f:
        for col in cat_cols:
            f.write(f"Column: {col}\n")
            vc = df[col].value_counts(dropna=False).head(top_n)
            f.write(vc.to_string())
            f.write("\n\n")


def time_series_plots(df, date_col, numeric_cols):
    if date_col is None:
        return
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    # Resample monthly for each numeric column (if enough span)
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        monthly = series.resample("M").sum()
        if monthly.shape[0] < 2:
            continue
        plt.figure(figsize=(8, 4))
        monthly.plot()
        plt.title(f"Monthly aggregated {col}")
        plt.tight_layout()
        fn = OUTPUT_DIR / f"time_monthly_{col}.png"
        plt.savefig(fn)
        plt.close()


def main(path="Retail_sales.csv"):
    print("Reading:", path)
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    df = read_data(path)
    print("Loaded. Shape:", df.shape)

    # Basic textual report
    basic_report(df, OUTPUT_DIR / "eda_report.txt")
    print("Wrote eda_report.txt")

    # Columns by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Plot numeric distributions (limit to first 12 numeric columns)
    plot_numeric_distributions(df, numeric_cols[:12])
    print("Wrote numeric plots")

    # Correlation heatmap
    plot_correlation(df, numeric_cols[:20])
    print("Wrote correlation matrix")

    # Categorical counts
    if cat_cols:
        categorical_counts(df, cat_cols)
        print("Wrote categorical counts")

    # Time series plots if date column found and numeric columns available
    date_col = detect_date_column(df)
    if date_col:
        print("Detected date column:", date_col)
        time_series_plots(df, date_col, numeric_cols[:6])
        print("Wrote time series plots (if applicable)")

    # Duplicates
    dup_count = df.duplicated().sum()
    with open(OUTPUT_DIR / "extras.txt", "w", encoding="utf8") as f:
        f.write(f"Duplicate rows: {dup_count}\n")
        f.write("Sample null columns per column (first 10 rows showing nulls):\n")
        f.write(df.isna().head(10).to_string())
    print("Wrote extras.txt")

    print("EDA completed. Outputs in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    # Allow passing a custom path as first arg
    infile = sys.argv[1] if len(sys.argv) > 1 else "Retail_sales.csv"
    main(infile)