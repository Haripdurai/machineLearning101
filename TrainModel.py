import os
import pickle
import pandas as pd

# Note: the distribution on PyPI is named "scikit-learn", but the import name in Python
# remains "sklearn" (i.e. you still do `import sklearn`). Some editors or linters may
# report "could not be resolved from source" if the package isn't installed or the
# language server can't find site-packages. To make failures clearer at runtime and
# provide a helpful message, import in a try/except and show how to install the package.
try:
    from sklearn.linear_model import LinearRegression  # type: ignore[import]
except Exception as e:
    raise ImportError(
        "scikit-learn is required but could not be imported.\n"
        "Install it with: pip install scikit-learn\n"
        "Note: the PyPI package name is 'scikit-learn' but you import it as 'sklearn'.\n"
        "Original error: "
    ) from e

# trainModel.py

CSV_PATH = "retail_sales.csv"
PKL_PATH = "trained_model.pkl"
X_COL = "Marketing Spend (USD)"
Y_COL = "Units Sold"

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    if X_COL not in df.columns or Y_COL not in df.columns:
        raise ValueError(f"CSV must contain columns: '{X_COL}' and '{Y_COL}'")
    # ensure numeric and drop rows with missing values in these columns
    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna(subset=[X_COL, Y_COL])
    X = df[[X_COL]].values  # 2D array
    y = df[Y_COL].values    # 1D array
    return X, y

def train_and_save_model(csv_path=CSV_PATH, pkl_path=PKL_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    X, y = load_and_prepare(csv_path)
    model = LinearRegression()
    model.fit(X, y)
    payload = {"model": model, "X": X, "y": y}
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved trained model and data to: {pkl_path}")

if __name__ == "__main__":
    train_and_save_model()