import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================================================
# 1) Load and Prepare Data
# ==================================================
def load_and_preprocess(file_path):
    # Read the CSV, skipping the complex header
    df = pd.read_csv(file_path, header=1)
    
    # Define the target: 1 if MPG > 30, else 0
    df['Actual_Class'] = (df['MPG'] > 30).astype(int)
    
    # Feature columns (ensure these match your CSV headers exactly)
    feature_columns = ['Pts/100', 'ORtg', 'PER', 'WS/48', 'BPM', 'On/Off']
    
    # Clean data: Remove rows with missing values in our key columns
    df = df.dropna(subset=feature_columns + ['Player', 'Year', 'MPG'])
    
    X_raw = df[feature_columns].values
    y = df['Actual_Class'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return df, X_scaled, y, scaler

# Load data
file_path = "statsheet.csv"
df, X_scaled, y, scaler = load_and_preprocess(file_path)

# Split for training evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Add bias column (x0 = 1) to the training set
X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
m, n_with_bias = X_train_bias.shape

# ==================================================
# 2) Logistic Regression Functions
# ==================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    return sigmoid(X @ w)

def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def gradient(X, y_true, y_prob):
    m = X.shape[0]
    return (1/m) * X.T @ (y_prob - y_true)

# ==================================================
# 3) Training Loop
# ==================================================
w = np.zeros(n_with_bias)
learning_rate = 0.1  # Bumped up slightly for faster convergence
num_iterations = 2000
cost_history = []

for i in range(num_iterations):
    y_prob = predict_proba(X_train_bias, w)
    cost = binary_cross_entropy(y_train, y_prob)
    cost_history.append(cost)
    
    grad = gradient(X_train_bias, y_train, y_prob)
    w -= learning_rate * grad

print(f"Training Complete. Final Loss: {cost_history[-1]:.4f}")

# ==================================================
# 4) Generate All Predictions & Find Mismatches
# ==================================================
X_all_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
all_probs = predict_proba(X_all_bias, w)
all_preds = (all_probs >= 0.5).astype(int)

df['Predicted_Class'] = all_preds
df['Probability'] = all_probs

# Define the two specific mismatch groups
# Group 1: Predicted Low (0), Actually High (1)
underrated = df[(df['Predicted_Class'] == 0) & (df['Actual_Class'] == 1)].sort_values(by='Probability')

# Group 2: Predicted High (1), Actually Low (0)
overrated = df[(df['Predicted_Class'] == 1) & (df['Actual_Class'] == 0)].sort_values(by='Probability', ascending=False)

print("\n" + "!"*60)
print("MISMATCH REPORT: SORTED BY MODEL CERTAINTY")
print("!"*60)

# --- Print Group 1 ---
print(f"\n[GROUP 1] PREDICTED LOW / ACTUALLY HIGH")
print("-" * 75)
print(f"{'Player':<25} | {'Year':<5} | {'Actual MPG':<10} | {'Prob of High MPG'}")
print("-" * 75)
for _, row in underrated.iterrows():
    print(f"{row['Player']:<25} | {row['Year']:<5} | {row['MPG']:<10} | {row['Probability']:.2%}")

# --- Print Group 2 ---
print(f"\n[GROUP 2] PREDICTED HIGH / ACTUALLY LOW")
print("-" * 75)
print(f"{'Player':<25} | {'Year':<5} | {'Actual MPG':<10} | {'Prob of High MPG'}")
print("-" * 75)
for _, row in overrated.iterrows():
    print(f"{row['Player']:<25} | {row['Year']:<5} | {row['MPG']:<10} | {row['Probability']:.2%}")

accuracy = (1 - (len(underrated) + len(overrated)) / len(df)) * 100
print("\n" + "="*75)
print(f"Total Mismatches: {len(underrated) + len(overrated)} | Overall Accuracy: {accuracy:.2f}%")
print("="*75)

# ==================================================
# 5) Search Utility
# ==================================================
def search_system():
    print("\n" + "="*40)
    print("MPG CATEGORY SEARCH (Type 'exit' to stop)")
    print("="*40)
    while True:
        query = input("\nEnter player name: ").strip().lower()
        if query == 'exit': break
        
        match = df[df['Player'].str.lower().str.contains(query)]
        
        if match.empty:
            print("No player found.")
        else:
            for _, row in match.iterrows():
                actual = "High (>30)" if row['Actual_Class'] == 1 else "Low (<=30)"
                pred = "High (>30)" if row['Predicted_Class'] == 1 else "Low (<=30)"
                print(f"\n{row['Player']} ({row['Year']}) - {row['Tm']}")
                print(f"  Actual MPG: {row['MPG']} ({actual})")
                print(f"  Predicted:  {pred} (Confidence: {row['Probability']:.2%})")

# Run search
search_system()
