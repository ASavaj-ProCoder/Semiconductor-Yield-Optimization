import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1.LOAD DATASET FROM LOCAL COMBINED CSV
df = pd.read_csv("uci-secom.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1] 
X = X.select_dtypes(include=[np.number])
y = y.replace(-1, 0)

# 2.DATA CLEANING & IMPUTATION
missing_pct = X.isnull().mean()
X = X.loc[:, missing_pct < 0.50]

imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)

X_clean = pd.DataFrame(X_clean)
X_clean = X_clean.loc[:, X_clean.var() > 0.0]

# 3.TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

# 4.TUNED MODEL TRAINING(Ditching SMOTE for deep penalization)
#We use max_features='sqrt' and a deep balanced weight class to force attention on faults
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=8,              #Limits overfitting on majority class
    class_weight='balanced_subsample', #Dynamically balances every single tree bootstrap
    random_state=42
)
model.fit(X_train, y_train)

# 5.THRESHOLD TUNING EVALUATION
#Instead of standard 50% cutoff, we flag a fault if the model is even 15% suspicious
probabilities = model.predict_proba(X_test)[:, 1]
custom_threshold = 0.15
y_pred = (probabilities >= custom_threshold).astype(int)

print(f"\n--- Fixed Evaluation (Threshold: {custom_threshold}) ---")
print("\n--- Factory Yield Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Operations Performance Report ---")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Fail (1)']))
