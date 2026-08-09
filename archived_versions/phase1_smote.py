import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Run: pip install imbalanced-learn

# 1.LOAD DATASET FROM LOCAL COMBINED CSV
df = pd.read_csv("uci-secom.csv")

#Separate features (X) and labels (y)
#Assumes target label is the very last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1] 

#FIX: Automatically filter and keep ONLY numeric features(drops timestamps and text)
X = X.select_dtypes(include=[np.number])

#Convert labels from -1/1 to 0/1 for easier processing
y = y.replace(-1, 0)

print(f"Cleaned Feature Shape: {X.shape} | Failures: {sum(y==1)} | Passes: {sum(y==0)}")

# 2.DATA CLEANING & IMPUTATION
#Drop columns that have more than 50% missing values
missing_pct = X.isnull().mean()
X = X.loc[:, missing_pct < 0.50]

#Imputer will now work perfectly since all columns are guaranteed numeric
imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)

#Remove features with zero variance(constant values across all wafers)
X_clean = pd.DataFrame(X_clean)
X_clean = X_clean.loc[:, X_clean.var() > 0.0]

# 3.TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

# 4.FIX CLASS IMBALANCE USING SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Resampled Training Target Shape: {np.bincount(y_train_res)}")

# 5.MODEL TRAINING
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# 6.EVALUATION
y_pred = model.predict(X_test)

print("\n--- Factory Yield Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Operations Performance Report ---")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Fail (1)']))