import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import joblib

from src.data_preprocessing import preprocess_features, save_preprocessed_data
from src.encoding import encode_target


# Paths

RAW_PATH = "data/raw/loan_approval_dataset.csv"
PREPROCESSED_PATH = "data/preprocessed/loan_preprocessed.csv"
MODEL_DIR = "models/"


# Step 1: Load RAW data

df = pd.read_csv(RAW_PATH)


# Step 2: Preprocess features

df = preprocess_features(df)       # cleaning, encoding, dropping loan_id
df = encode_target(df)             # encode target column

# Drop rows with missing target
df = df.dropna(subset=["loan_status"])


# Step 3: Save preprocessed data

save_preprocessed_data(df, PREPROCESSED_PATH)
print("Preprocessed dataset saved!")


# Step 4: Split features/target

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Target distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 5: Helper function for threshold evaluation

def evaluate_threshold(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model using a custom probability threshold.
    Returns accuracy, confusion matrix, and ROC-AUC.
    """
    probs = model.predict_proba(X_test)[:, 1]  # probability of class 1 (Approved)
    y_pred = (probs >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, probs)
    return acc, cm, roc


# Step 6: Train Logistic Regression

print("\nTraining Logistic Regression...")
log_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Evaluate at threshold 0.55
threshold = 0.55
log_acc, log_cm, log_roc = evaluate_threshold(log_model, X_test, y_test, threshold)
print(f"\nLogistic Regression Results at threshold {threshold}")
print("Accuracy:", log_acc)
print("Confusion Matrix:\n", log_cm)
print("ROC-AUC:", log_roc)

# Save Logistic Regression model
joblib.dump(log_model, MODEL_DIR + "logistic_model.pkl")


# Step 7: Train Random Forest

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluate at threshold 0.55
rf_acc, rf_cm, rf_roc = evaluate_threshold(rf_model, X_test, y_test, threshold)
print(f"\nRandom Forest Results at threshold {threshold}")
print("Accuracy:", rf_acc)
print("Confusion Matrix:\n", rf_cm)
print("ROC-AUC:", rf_roc)

# Save Random Forest model
joblib.dump(rf_model, MODEL_DIR + "random_forest_model.pkl")

# Step 8: Comparison

print("\nComparison at threshold", threshold)
print("-----------")
print(f"Logistic Regression -> Accuracy: {log_acc:.3f}, ROC-AUC: {log_roc:.3f}")
print(f"Random Forest       -> Accuracy: {rf_acc:.3f}, ROC-AUC: {rf_roc:.3f}")

print("\nAll models saved to:", MODEL_DIR)
