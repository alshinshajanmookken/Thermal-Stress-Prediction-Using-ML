import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# =====================================
# 1. LOAD DATA
# =====================================
file_name = "data-honors.xlsx"

if not os.path.exists(file_name):
    raise FileNotFoundError(f"{file_name} not found")

data = pd.read_excel(file_name)

# Clean column names
data.columns = data.columns.str.strip()

# Rename columns (if required)
data = data.rename(columns={
    "FA_ %": "FA",
    "Thickness_mm": "Thickness"
})

print("Columns in dataset:\n", data.columns)

# =====================================
# 2. FEATURE ENGINEERING
# =====================================
data["dT_total"] = data["T_top"] - data["T_bot"]

# Curvature (non-linear temperature profile)
data["curvature"] = data["T_top"] - 2*data["T_mid"] + data["T_bot"]

# =====================================
# 3. FEATURES & TARGETS
# =====================================
features = [
    "FA",
    "Thickness",
    "E",
    "dT_total",
    "curvature"
]

targets = ["Sigma_top", "Sigma_mid", "Sigma_bot"]

X = data[features]
y = data[targets]

# =====================================
# 4. TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 5. MODEL TRAINING
# =====================================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================
# 6. CROSS VALIDATION
# =====================================
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("\nCROSS VALIDATION")
print("Mean R2:", round(cv_scores.mean(), 4))
print("Std Dev :", round(cv_scores.std(), 4))

# =====================================
# 7. MODEL PERFORMANCE
# =====================================
y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE\n")

for i, target in enumerate(targets):
    r2 = r2_score(y_test[target], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[:, i]))
    mae = mean_absolute_error(y_test[target], y_pred[:, i])

    print(target)
    print("R2   :", round(r2, 4))
    print("RMSE :", round(rmse, 4))
    print("MAE  :", round(mae, 4))
    print()

# =====================================
# 8. ACTUAL vs PREDICTED PLOTS
# =====================================
os.makedirs("plots", exist_ok=True)

print("Saving Actual vs Predicted plots...")

for i, target in enumerate(targets):

    actual = y_test[target].values
    predicted = y_pred[:, i]

    fig, ax = plt.subplots()

    ax.scatter(actual, predicted)

    # Ideal 45° line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    r2 = r2_score(actual, predicted)

    ax.set_xlabel("Actual Stress (MPa)")
    ax.set_ylabel("Predicted Stress (MPa)")
    ax.set_title(f"{target} (R2 = {r2:.3f})")

    ax.grid(True)

    plt.savefig(f"plots/{target}_actual_vs_pred.png", dpi=300)
    plt.close()

print("Plots saved in 'plots' folder.")

# =====================================
# 9. FEATURE IMPORTANCE
# =====================================
print("\nFEATURE IMPORTANCE\n")

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

for f, imp in zip(features, perm.importances_mean):
    print(f"{f}: {round(imp,4)}")

# =====================================
# 10. SAVE MODEL
# =====================================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/thermal_model.pkl")

print("\nModel saved successfully in 'model' folder!")