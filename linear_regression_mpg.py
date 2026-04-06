import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================================================
# 1) Load and Prepare Data
# ==================================================
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, header=1)
    
    # We are predicting the ACTUAL MPG
    target_column = 'MPG'
    feature_columns = ['Pts/100', 'ORtg', 'PER', 'WS/48', 'BPM', 'On/Off']
    
    # Clean data
    df = df.dropna(subset=feature_columns + ['Player', 'Year', target_column])
    
    # --- Correlation Analysis ---
    print("\n" + "-"*40)
    print("STATISTICAL CORRELATION WITH MPG")
    print("-"*40)
    # This shows how much each stat "moves" with Minutes
    correlations = df[feature_columns + [target_column]].corr()[target_column].drop(target_column)
    print(correlations.sort_values(ascending=False))
    print("-" * 40 + "\n")
    
    X_raw = df[feature_columns].values
    y = df[target_column].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return df, X_scaled, y, scaler, feature_columns

file_path = "statsheet.csv"
df, X_scaled, y, scaler, features = load_and_preprocess(file_path)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
m, n_with_bias = X_train_bias.shape

# ==================================================
# 2) Linear Regression Functions
# ==================================================
def predict_values(X, w):
    return X @ w

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient_linear(X, y_true, y_pred):
    m = X.shape[0]
    return (2/m) * X.T @ (y_pred - y_true)

# ==================================================
# 3) Training Loop
# ==================================================
w = np.zeros(n_with_bias)
learning_rate = 0.01 
num_iterations = 3000
cost_history = []

for i in range(num_iterations):
    y_pred = predict_values(X_train_bias, w)
    cost = mean_squared_error(y_train, y_pred)
    cost_history.append(cost)
    grad = gradient_linear(X_train_bias, y_train, y_pred)
    w -= learning_rate * grad

# ==================================================
# 4) Analysis: Underestimated vs Overestimated
# ==================================================
X_all_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
df['Predicted_MPG'] = predict_values(X_all_bias, w)
df['MPG_Gap'] = df['MPG'] - df['Predicted_MPG']

# Underestimated (Positive Gap): Predicted low, but got high minutes
underestimated = df[df['MPG_Gap'] > 0].sort_values(by='MPG_Gap', ascending=False)

# Overestimated (Negative Gap): Predicted high, but got low minutes
overestimated = df[df['MPG_Gap'] < 0].sort_values(by='MPG_Gap', ascending=True)

print("="*85)
print("TOP 50 UNDERESTIMATED (Predicted Low Minutes but actually high)")
print("="*85)
for _, row in underestimated.head(50).iterrows():
    print(f"{row['Player']:<20} | {row['Year']:<5} | Actual: {row['MPG']:<5.1f} | Pred: {row['Predicted_MPG']:<5.1f} | Gap: +{row['MPG_Gap']:.1f}")

print("\n" + "="*85)
print("TOP 50 OVERESTIMATED (Predicted High Minutes but actually low)")
print("="*85)
for _, row in overestimated.head(50).iterrows():
    print(f"{row['Player']:<20} | {row['Year']:<5} | Actual: {row['MPG']:<5.1f} | Pred: {row['Predicted_MPG']:<5.1f} | Gap: {row['MPG_Gap']:.1f}")

# ==================================================
# 5) Search Utility
# ==================================================
def search_system():
    print("\n" + "="*40)
    print("MPG CALCULATOR (Type 'exit' to quit)")
    print("="*40)
    while True:
        query = input("\nEnter player name: ").strip().lower()
        if query == 'exit': break
        match = df[df['Player'].str.lower().str.contains(query)]
        if match.empty:
            print("No player found.")
        else:
            for _, row in match.iterrows():
                print(f"\n{row['Player']} ({row['Year']}) - {row['Tm']}")
                print(f"  Actual:    {row['MPG']:.1f} MPG")
                print(f"  Predicted: {row['Predicted_MPG']:.1f} MPG")
                print(f"  Gap:       {row['MPG_Gap']:.1f} minutes")

search_system()
