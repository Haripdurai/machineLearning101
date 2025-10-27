import pickle
import numpy as np

import matplotlib.pyplot as plt

# Load the trained linear regression model
with open("trained_model.pkl", "rb") as f:
    payload = pickle.load(f)

# The training script saved a dict payload: {"model": model, "X": X, "y": y}.
# Handle both cases: payload is the dict, or the pickle contains the model object directly.
if isinstance(payload, dict):
    if "model" in payload:
        model = payload["model"]
    else:
        raise RuntimeError("Pickle file is a dict but does not contain a 'model' key.")
elif hasattr(payload, "predict"):
    # old-style pickle where the estimator was saved directly
    model = payload
else:
    raise RuntimeError("Unrecognized pickle payload: expected a scikit-learn estimator or a dict containing one.")

# Example data (replace with your actual data)
# X: marketing spend, y: units sold
# For demonstration, let's assume you have these arrays:
X = np.array([1000, 2000, 3000, 4000, 5000]).reshape(-1, 1)
y = np.array([150, 300, 450, 600, 750])

# Predict using the model
y_pred = model.predict(X)

# Plot scatter and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Marketing Spend (USD)')
plt.ylabel('Units Sold')
plt.title('Marketing Spend vs Units Sold')
plt.legend()
plt.tight_layout()
plt.savefig('plot.png')
plt.close()