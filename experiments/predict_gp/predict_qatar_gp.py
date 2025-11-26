import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

print("--- Qatar Grand Prix (Round 23) Prediction Simulation ---")

# --- 1. Load Data ---
DATA_FILE = "data/processed/improved_feature_engineered_data.csv"

# Robust path finding
possible_paths = [DATA_FILE, f"../{DATA_FILE}", f"../../{DATA_FILE}"]
for path in possible_paths:
    if os.path.exists(path):
        DATA_FILE = path
        break

try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: Data file not found.")
    exit()

# --- 2. Configuration ---
TARGET_ROUND = 23 # Qatar is Round 23 in 2024
TARGET_YEAR = 2024

# Features (Same as training)
target = 'IsPodium' 
non_feature_cols = [
    'Year', 'RoundNumber', 'FullName', 'EventName', 'TeamName', # Removed TeamName from here just in case, but it's not in df anyway
    'RacePos', 'Points', 'Laps', 'FinishedRace', 
    'IsRaceWinner', 'IsPodium', 'Time', 'Driver', 'Constructor',
    'GridPosition', 'QualyPos' 
]
features = [col for col in df.columns if col not in non_feature_cols]

# --- 3. Split Data: Past vs Qatar ---

# Train on EVERYTHING before Qatar (Rounds 1-22 of 2024)
train_mask = (df['Year'] == TARGET_YEAR) & (df['RoundNumber'] < TARGET_ROUND)

# The "New" Data: Qatar Only
qatar_mask = (df['Year'] == TARGET_YEAR) & (df['RoundNumber'] == TARGET_ROUND)

X_train = df[train_mask][features]
y_train = df[train_mask][target]

X_qatar = df[qatar_mask][features]

# FIX: Removed 'TeamName' from this list because it was One-Hot Encoded away
qatar_context = df[qatar_mask][['FullName', 'QualyPos', 'RacePos']].copy()

print(f"Training on {len(X_train)} historical driver results.")
print(f"Predicting for {len(X_qatar)} drivers in Qatar.")

if len(X_qatar) == 0:
    print("Error: No data found for Round 23. Please check your CSV to ensure Round 23 exists.")
    exit()

# --- 4. Train the Model ---
print("\nTraining model on pre-Qatar data...")
model = RandomForestClassifier(
    n_estimators=500, 
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train, y_train)

# --- 5. Predict Qatar ---
print("Predicting outcomes...\n")

# Get probabilities (Confidence)
probs = model.predict_proba(X_qatar)[:, 1]
predictions = model.predict(X_qatar)

# --- 6. The Report ---
qatar_context['Predicted_Podium'] = predictions
qatar_context['Podium_Probability'] = probs
qatar_context['Confidence'] = qatar_context['Podium_Probability'].apply(lambda x: f"{x*100:.1f}%")

# Sort by highest probability
qatar_context = qatar_context.sort_values(by='Podium_Probability', ascending=False)

# Display Columns
cols = ['FullName', 'QualyPos', 'Confidence', 'Predicted_Podium', 'RacePos']

print(f"--- PREDICTIONS FOR QATAR GP (Round {TARGET_ROUND}) ---")
print("(Ranked by Model Confidence)")
print(qatar_context[cols].to_string(index=False))

# --- 7. Analysis ---
top_3_picks = qatar_context.head(3)
print("\n--- Model's Top 3 Picks ---")
for i, row in top_3_picks.iterrows():
    print(f"{row['FullName']}: {row['Confidence']} chance.")

actual_podium = qatar_context[qatar_context['RacePos'] <= 3]
print("\n--- Actual Podium ---")
for i, row in actual_podium.iterrows():
    print(f"P{int(row['RacePos'])}: {row['FullName']}")