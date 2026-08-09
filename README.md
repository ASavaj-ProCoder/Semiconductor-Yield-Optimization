# Semiconductor Manufacturing Yield Optimization Pipeline

An end-to-end, cost-sensitive machine learning system developed to isolate root-cause anomalies in high-dimensional sensor streams using the **UCI SECOM dataset**. This project addresses severe class imbalances in cleanroom manufacturing environments and maps model performance directly to factory cost savings. 

# Executive Summary & Financial Impact

Standard machine learning models fail on this dataset, yielding 0% defect detection (Recall) due to heavy sensor noise (591 features) and a severe ~14:1 pass-to-fail class imbalance.By replacing naive oversampling with a pipeline featuring statistical feature selection,class-weighted tree structures,and tuned decision thresholds,this system successfully captures **62% of real manufacturing defects**. 

# Operational Cost Model

Performance is evaluated against real-world factory constraints: 

**Cost of a Missed Defect (False Negative):** $5,000 (Defective wafer escapes to customer)
**Cost of a False Alarm (False Positive):** $200 (Overhead to manually re-test a clean wafer)

Evaluating on the test distribution (314 wafers; 21 actual failures): 

**Baseline Approach (Predict All Pass):** Misses all 21 defects, costing **$105,000**.
**Optimized Pipeline:** Catches 13 defects and incurs 97 false alarms, costing **$59,400**.
**Net Savings:** **$45,600** in operational risk mitigation.

# Repository Architecture & Iterative Development

The repository is structured to document the progressive debugging and optimization phases required to solve the imbalance problem: 

text

├── uci-secom.csv              # Raw manufacturing data (1567 records, 591 features)
├── main.py                    # Production pipeline (Feature Selection + Threshold Tuning)
├── main_baseline.py           # Phase 3 baseline (No feature selection layer)
└── archived_versions/
    ├── phase1_smote.py        # Phase 1: Synthetic oversampling experiment
    └── phase2_tuning.py       # Phase 2: Class weights and initial thresholding
