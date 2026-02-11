import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import your custom ResNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.definitions import TabularResNet

def run_benchmark(name, X, y, input_dim):
    print(f"\n📊 Benchmarking {name.upper()} Dataset...")
    
    # 1. Standard Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(imputer.fit_transform(X))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {"ResNet": [], "RandomForest": [], "XGBoost": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- Model A: Random Forest (Baseline) ---
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
        results["RandomForest"].append(rf_auc)

        # --- Model B: XGBoost (Strong Contender) ---
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        xgb_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:, 1])
        results["XGBoost"].append(xgb_auc)

        # --- Model C: Tabular ResNet (Your Model) ---
        # Note: Uses a simplified training loop for benchmarking purposes
        model = TabularResNet(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        for _ in range(100): # Standard epochs
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            res_preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
            res_auc = roc_auc_score(y_val, res_preds)
            results["ResNet"].append(res_auc)

    # 2. Output Comparison Table
    summary = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "Tabular ResNet"],
        "Mean AUC-ROC": [
            np.mean(results["RandomForest"]), 
            np.mean(results["XGBoost"]), 
            np.mean(results["ResNet"])
        ],
        "Std Dev": [
            np.std(results["RandomForest"]), 
            np.std(results["XGBoost"]), 
            np.std(results["ResNet"])
        ]
    })
    print(summary.to_string(index=False))

# Run Benchmarks
diabetes = fetch_openml(data_id=37, as_frame=False, parser='auto')
run_benchmark("Diabetes", diabetes.data, (diabetes.target == 'tested_positive').astype(int), 8)

heart = fetch_openml(data_id=1498, as_frame=False, parser='auto')
run_benchmark("Heart Disease", heart.data, (heart.target == '2').astype(int), 9)