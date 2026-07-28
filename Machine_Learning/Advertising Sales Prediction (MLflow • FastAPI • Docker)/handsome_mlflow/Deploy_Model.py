import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from joblib import dump

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True , help='Path to CSV file')
parser.add_argument("--n_estimators", type=int, default=50, help='Number of trees')
parser.add_argument("--random_state", type=int, default=101, help='Random state for splitting data')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.filename)
X = df.drop("sales", axis=1)
y = df["sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.random_state)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=args.random_state)
# Create pipeline
scale = StandardScaler()
model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state)
pipe = Pipeline([("scale", scale), ("model", model)])

# Train model
pipe.fit(X_train, y_train)

# Evaluate
preds = pipe.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"RMSE: {rmse}")

# Log with MLflow
with mlflow.start_run():
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(pipe, "model")

# Save artifacts locally
dump(pipe, "final_model.pkl")
dump(list(X.columns), "column_name.pkl")
