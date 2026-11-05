# train.py : reads the data, prepares it, trains 2 models, saves the best one.

import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # lets charts be saved without opening a window
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- SETTINGS
BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "us_accidents_sample.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

N_ROWS = 200_000        # how many accidents we use (make it 100_000 if memory error)
THRESHOLD = 0.3         # probability >= this means "high impact" (change to your chosen value)

WEATHER_COLS = ["Temperature(F)", "Humidity(%)", "Pressure(in)",
                "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
ROAD_COLS = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
             "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal"]
COLUMNS_NEEDED = (["Severity", "Start_Time", "Start_Lat", "Start_Lng", "City", "County",
                   "State", "Weather_Condition", "Sunrise_Sunset"] + WEATHER_COLS + ROAD_COLS)

# ------------------------------------------------------- STEP 1: LOAD DATA
print("Step 1: loading data...")
df = pd.read_csv(DATA_PATH, usecols=lambda c: c in COLUMNS_NEEDED)
df = df.sample(min(N_ROWS, len(df)), random_state=42).reset_index(drop=True)
print("Rows and columns:", df.shape)

# --------------------------------------------------- STEP 2: CLEAN DATA
print("Step 2: cleaning data...")

# 2a. Make new columns from the time of the accident
df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
df = df.dropna(subset=["Start_Time"]).reset_index(drop=True)
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Start_Time"].dt.dayofweek            # 0 = Monday
df["Month"] = df["Start_Time"].dt.month
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["IsRushHour"] = (df["Hour"].isin([7, 8, 9, 16, 17, 18]) & (df["IsWeekend"] == 0)).astype(int)
df = df.drop(columns=["Start_Time"])

# 2b. Group the many weather labels into 6 simple groups
def weather_group(text):
    text = str(text).lower()
    if "snow" in text or "ice" in text or "sleet" in text: return "Snow/Ice"
    if "rain" in text or "drizzle" in text or "storm" in text or "thunder" in text: return "Rain"
    if "fog" in text or "mist" in text or "haze" in text or "smoke" in text: return "Fog/Haze"
    if "cloud" in text or "overcast" in text: return "Cloudy"
    if "clear" in text or "fair" in text: return "Clear"
    return "Other"

df["Weather_Group"] = df["Weather_Condition"].apply(weather_group)
df = df.drop(columns=["Weather_Condition"])

# 2c. Fill missing values (numbers -> median, text -> "Unknown")
df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0)
number_cols = WEATHER_COLS + ["Start_Lat", "Start_Lng"]
df[number_cols] = df[number_cols].fillna(df[number_cols].median())
for col in ["City", "County", "State", "Sunrise_Sunset"]:
    df[col] = df[col].fillna("Unknown").astype(str)
df[ROAD_COLS] = df[ROAD_COLS].astype(int)

# 2d. Target: 0 = low impact (severity 1-2), 1 = high impact (severity 3-4)
df["High_Severity"] = (df["Severity"] >= 3).astype(int)
print("High-impact share:", round(df["High_Severity"].mean() * 100, 1), "%")

# ----------------------------------------------------- STEP 3: CHARTS (EDA)
print("Step 3: saving charts...")
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
df["Severity"].value_counts().sort_index().plot(kind="bar", ax=axes[0], title="Severity count")
df.groupby("Hour")["High_Severity"].mean().plot(ax=axes[1], title="High-impact rate by hour")
df.groupby("Weather_Group")["High_Severity"].mean().sort_values().plot(
    kind="barh", ax=axes[2], title="High-impact rate by weather")
plt.tight_layout()
plt.savefig(OUT / "eda.png", dpi=150)
plt.close()

# ------------------------------------------- STEP 4: MAKE DATA READY FOR MODELS
print("Step 4: preparing features...")

# Small tables the web app needs later
locations = (df.groupby(["State", "City"])
             .agg(County=("County", "first"), Start_Lat=("Start_Lat", "median"),
                  Start_Lng=("Start_Lng", "median"), Count=("Severity", "size"))
             .reset_index())
locations = locations[(locations["Count"] >= 5) & (locations["City"] != "Unknown")]
locations = locations.drop(columns="Count").reset_index(drop=True)
defaults = df[WEATHER_COLS].median().to_dict()

# City/County/State have thousands of names, so each name is replaced by
# how often it appears in the data (frequency encoding)
freq_maps = {}
for col in ["City", "County", "State"]:
    freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
    df[col + "_freq"] = df[col].map(freq_maps[col])

y = df["High_Severity"]                                   # what we want to predict
X = df.drop(columns=["City", "County", "State", "Severity", "High_Severity"])
X = pd.get_dummies(X, columns=["Weather_Group", "Sunrise_Sunset"], dtype=np.int8)  # text -> 0/1
X = X.astype(np.float32)

# 80% of data to learn from, 20% kept hidden for the final test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("Train rows:", len(X_train), "| Test rows:", len(X_test))

# ------------------------------------------------------- STEP 5: TRAIN MODELS
print("Step 5: training models (a few minutes)...")

# Model 1: Logistic Regression = simple baseline
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
logreg.fit(X_train, y_train)
print("  Logistic Regression done")

# Model 2: Random Forest = many decision trees voting together
forest = RandomForestClassifier(n_estimators=60, max_depth=18, min_samples_leaf=5,
                                max_samples=0.5, n_jobs=1, random_state=42)
forest.fit(X_train, y_train)
print("  Random Forest done")

# ------------------------------------------------------ STEP 6: EVALUATE
print("Step 6: evaluating on the test set...")
rows = []
for name, model in [("Logistic Regression", logreg), ("Random Forest", forest)]:
    prob = model.predict_proba(X_test)[:, 1]              # probability of "high impact"
    pred = (prob >= THRESHOLD).astype(int)
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Macro_F1": f1_score(y_test, pred, average="macro"),
                 "Precision": precision_score(y_test, pred, zero_division=0),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC_AUC": roc_auc_score(y_test, prob)})
results = pd.DataFrame(rows).round(4)
results.to_csv(OUT / "model_comparison.csv", index=False)
print(results.to_string(index=False))
baseline = float(1 - y_test.mean())
print("Baseline accuracy (always say 'low'):", round(baseline, 4))

# Train accuracy vs test accuracy (to check for overfitting), saved for the app
print("\nModel                | Train accuracy | Test accuracy")
acc_rows = []
for name, model in [("Logistic Regression", logreg), ("Random Forest", forest)]:
    train_pred = (model.predict_proba(X_train)[:, 1] >= THRESHOLD).astype(int)
    test_pred = (model.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    tr_acc = accuracy_score(y_train, train_pred)
    te_acc = accuracy_score(y_test, test_pred)
    print(f"{name:20s} |    {tr_acc:.4f}      |    {te_acc:.4f}")
    acc_rows.append({"Model": name, "Train_Accuracy": tr_acc, "Test_Accuracy": te_acc})
pd.DataFrame(acc_rows).round(4).to_csv(OUT / "train_test_accuracy.csv", index=False)

# Which inputs mattered most to the Random Forest
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
importance[::-1].plot(kind="barh", figsize=(8, 6), title="Top 15 important features")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

#STEP 7: SAVE
print("Step 7: saving model...")
joblib.dump({"model": forest, "model_name": "Random Forest", "columns": X.columns.tolist(),
             "threshold": THRESHOLD, "freq_maps": freq_maps, "locations": locations,
             "defaults": defaults, "baseline": baseline},
            BASE / "model.pkl", compress=3)
joblib.dump({"X_test": X_test, "y_test": y_test}, BASE / "data" / "test_data.joblib", compress=3)
print("Done! Now run: python test.py")