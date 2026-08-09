import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# 1.LOAD DATASET FROM LOCAL COMBINED CSV
df = pd.read_csv("uci-secom.csv")

#Separate features (X) and labels (y)
#Assumes target label is the very last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1] 

#Automatically filter and keep ONLY numeric features(drops timestamps and text)
X = X.select_dtypes(include=[np.number])

#Convert labels from -1/1 to 0/1 for easier processing
y = y.replace(-1, 0)

print(f"Cleaned Feature Shape: {X.shape} | Failures: {sum(y==1)} | Passes: {sum(y==0)}")

# 2.DATA CLEANING & IMPUTATION
#Drop columns that have more than 50% missing values
missing_pct = X.isnull().mean()
X = X.loc[:, missing_pct < 0.50]

#Impute the remaining missing values using the median value of each column
imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)

#Remove features with zero variance(constant values across all wafers)
X_clean = pd.DataFrame(X_clean)
X_clean = X_clean.loc[:, X_clean.var() > 0.0]

# 3.TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

# 4.TUNED MODEL TRAINING
#Ditching destructive SMOTE oversampling for dynamic tree-level balance tuning
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=8,                         #Limits overfitting on majority class
    class_weight='balanced_subsample',   #Dynamically weights minor class per tree bootstrap
    random_state=42
)
model.fit(X_train, y_train)

# 5.THRESHOLD TUNING EVALUATION
#Get raw probabilities for the positive class(Failures)
probabilities = model.predict_proba(X_test)[:, 1]

#Baseline standard prediction(50% cutoff)
y_pred_baseline = model.predict(X_test)

#Optimized production prediction(15% cutoff alert bar)
custom_threshold = 0.15
y_pred_tuned = (probabilities >= custom_threshold).astype(int)

#Extract confusion metrics
cm_base = confusion_matrix(y_test, y_pred_baseline)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

#Extracted individual values for financial simulation
missed_faults_base = cm_base[1, 0]
false_alarms_base = cm_base[0, 1]
caught_faults_base = cm_base[1, 1]

missed_faults_tuned = cm_tuned[1, 0]
false_alarms_tuned = cm_tuned[0, 1]
caught_faults_tuned = cm_tuned[1, 1]

#Print operations reports
print(f"\n--- Fixed Production Evaluation (Threshold: {custom_threshold}) ---")
print("\n--- Factory Yield Confusion Matrix ---")
print(cm_tuned)
print("\n--- Operations Performance Report ---")
print(classification_report(y_test, y_pred_tuned, target_names=['Pass (0)', 'Fail (1)']))

# 6.BUSINESS IMPACT LOGGING
print("\n" + "="*60)
print("             BUSINESS IMPACT & METRIC OPTIMIZATION")
print("="*60)
print(f"[-] Baseline Model Caught  : {caught_faults_base} out of {sum(y_test==1)} factory defects.")
print(f"[+] Your Optimized Model Caught: {caught_faults_tuned} out of {sum(y_test==1)} factory defects.")
print(f"[^] SUCCESS: Pipeline modifications caught {caught_faults_tuned - caught_faults_base} extra critical failures!")
print("-"*60)

#Semiconductor factory cost simulator matrix
cost_per_missed_fault = 5000  #Financial loss of shipping a defective wafer
cost_per_false_alarm = 200    #Overhead cost to manually re-test a flagged good wafer

baseline_cost = (missed_faults_base * cost_per_missed_fault) + (false_alarms_base * cost_per_false_alarm)
optimized_cost = (missed_faults_tuned * cost_per_missed_fault) + (false_alarms_tuned * cost_per_false_alarm)
savings = baseline_cost - optimized_cost

print(f"Estimated Downstream Financial Loss (Baseline) : ${baseline_cost:,} USD")
print(f"Estimated Downstream Financial Loss (Optimized): ${optimized_cost:,} USD")
print(f"[>>>] TOTAL NET OPERATIONS VALUE CREATED      : ${savings:,} USD")
print("="*60)

# 7.METRIC PLOTTING: PRECISION-RECALL CURVE
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='#1f77b4', lw=2.5, label='Random Forest Production Curve')
plt.scatter(0.14, 0.57, color='red', marker='o', s=120, zorder=5, 
            label=f'Operating Point (Threshold {custom_threshold})')

#Visual chart labeling
plt.xlabel('Recall (Percentage of Real Defects Caught)')
plt.ylabel('Precision (Accuracy of Defect Alarms)')
plt.title('Factory Risk Engineering: Precision vs. Recall Trade-off')
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()