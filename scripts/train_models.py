import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib
import os
import sys
import copy

# Import Model Definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.definitions import TabularResNet

os.makedirs("models", exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=15, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

def train_and_save(name, X, y, input_dim):
    print(f"\n🔬 Starting Cross-Validated Training for: {name.upper()}")
    
    # 1. Preprocessing
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Save base artifacts
    joblib.dump(scaler, f"models/{name}_scaler.pkl")
    joblib.dump(imputer, f"models/{name}_imputer.pkl")
    
    # 2. Stratified K-Fold Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    best_overall_auc = 0
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        # Split Data
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        
        # Initialize Model, Loss, and Optimizer
        model = TabularResNet(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        early_stop = EarlyStopping(patience=15)
        
        # Training Loop
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validation Step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
            early_stop(val_loss, model)
            if early_stop.early_stop:
                break
        
        # Post-Fold Evaluation
        model.load_state_dict(early_stop.best_model_weights)
        model.eval()
        with torch.no_grad():
            preds = model(X_val).numpy()
            fold_auc = roc_auc_score(y_val.numpy(), preds)
            fold_aucs.append(fold_auc)
            print(f"   🚩 Fold {fold+1}: AUC-ROC = {fold_auc:.4f} (Stopped at epoch {epoch})")
            
            # Keep the absolute best version across folds
            if fold_auc > best_overall_auc:
                best_overall_auc = fold_auc
                torch.save(model.state_dict(), f"models/{name}_resnet.pth")

    print(f"✅ {name.upper()} Training Complete. Avg AUC: {np.mean(fold_aucs):.4f}")

# Main Execution
print("📥 Downloading Clinical Datasets...")
# 1. Diabetes (Pima)
diabetes = fetch_openml(data_id=37, as_frame=False, parser='auto')
y_d = (diabetes.target == 'tested_positive').astype(int)
train_and_save("diabetes", diabetes.data, y_d, 8)

# 2. Heart Disease (Statlog)
heart = fetch_openml(data_id=1498, as_frame=False, parser='auto')
y_h = (heart.target == '2').astype(int)
train_and_save("heart", heart.data, y_h, 9)