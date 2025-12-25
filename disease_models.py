"""
Symptom Checker - Disease Prediction Models with Hyperparameter Tuning
Includes: RandomForest, XGBoost, MLP with Grid Search optimization
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Classical ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                            top_k_accuracy_score, recall_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class SymptomDataset(Dataset):
    """PyTorch Dataset for symptoms"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron with Dropout and Batch Normalization
    Designed to prevent overfitting on medical data
    """
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class HyperparameterTuner:
    """
    Grid search and randomized search for hyperparameter optimization
    """
    
    @staticmethod
    def tune_random_forest(X_train, y_train, X_val, y_val, 
                          n_iter: int = 20, cv: int = 3) -> Dict:
        """
        Randomized search for Random Forest hyperparameters
        """
        print("\n" + "="*60)
        print("TUNING RANDOM FOREST HYPERPARAMETERS")
        print("="*60)
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            rf,
            param_distributions,
            n_iter=n_iter,
            scoring='f1_macro',
            cv=cv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\nSearching through {n_iter} random combinations...")
        random_search.fit(X_train, y_train)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        print(f"\n✓ Best CV F1 Score: {best_score:.4f}")
        print(f"✓ Best Parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        # Validate on validation set
        val_score = f1_score(y_val, random_search.predict(X_val), average='macro')
        print(f"✓ Validation F1 Score: {val_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_cv_score': best_score,
            'val_score': val_score,
            'best_estimator': random_search.best_estimator_
        }
    
    @staticmethod
    def tune_xgboost(X_train, y_train, X_val, y_val,
                    n_iter: int = 25, cv: int = 3) -> Dict:
        """
        Randomized search for XGBoost hyperparameters
        """
        print("\n" + "="*60)
        print("TUNING XGBOOST HYPERPARAMETERS")
        print("="*60)
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 7, 10, 12, 15],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.5, 1, 2, 5],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0],
            'min_child_weight': [1, 3, 5, 7]
        }
        
        xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        
        random_search = RandomizedSearchCV(
            xgb,
            param_distributions,
            n_iter=n_iter,
            scoring='f1_macro',
            cv=cv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\nSearching through {n_iter} random combinations...")
        random_search.fit(X_train, y_train)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        print(f"\n✓ Best CV F1 Score: {best_score:.4f}")
        print(f"✓ Best Parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        # Validate on validation set
        val_score = f1_score(y_val, random_search.predict(X_val), average='macro')
        print(f"✓ Validation F1 Score: {val_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_cv_score': best_score,
            'val_score': val_score,
            'best_estimator': random_search.best_estimator_
        }
    
    @staticmethod
    def tune_mlp(X_train, y_train, X_val, y_val,
                n_features: int, n_classes: int,
                device: torch.device) -> Dict:
        """
        Grid search for MLP hyperparameters
        """
        print("\n" + "="*60)
        print("TUNING MLP HYPERPARAMETERS")
        print("="*60)
        
        # Define search space
        param_grid = {
            'hidden_dims': [
                [512, 256, 128],
                [512, 512, 256],
                [384, 256, 128, 64]
            ],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.0005, 0.001, 0.002],
            'batch_size': [64, 128]
        }
        
        best_score = 0
        best_config = None
        results = []
        
        print(f"\nTesting {len(param_grid['hidden_dims']) * len(param_grid['dropout_rate']) * len(param_grid['learning_rate']) * len(param_grid['batch_size'])} configurations...")
        
        config_num = 0
        # Grid search (sample subset for speed)
        for hidden_dims in param_grid['hidden_dims']:
            for dropout in param_grid['dropout_rate']:
                for lr in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size'][:2]:  # Limit batch sizes
                        config_num += 1
                        
                        if config_num % 5 == 0:
                            print(f"\nTesting configuration {config_num}...")
                        
                        # Build and train model
                        model = MLPClassifier(
                            input_dim=n_features,
                            num_classes=n_classes,
                            hidden_dims=hidden_dims,
                            dropout_rate=dropout
                        ).to(device)
                        
                        # Quick training (fewer epochs for tuning)
                        train_dataset = SymptomDataset(X_train, y_train)
                        val_dataset = SymptomDataset(X_val, y_val)
                        
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        
                        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                        criterion = nn.CrossEntropyLoss()
                        
                        # Train for 20 epochs
                        for epoch in range(20):
                            model.train()
                            for X_batch, y_batch in train_loader:
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                loss = criterion(outputs, y_batch)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                        
                        # Evaluate
                        model.eval()
                        all_preds = []
                        all_labels = []
                        
                        with torch.no_grad():
                            for X_batch, y_batch in val_loader:
                                X_batch = X_batch.to(device)
                                outputs = model(X_batch)
                                _, predicted = torch.max(outputs, 1)
                                all_preds.extend(predicted.cpu().numpy())
                                all_labels.extend(y_batch.numpy())
                        
                        val_f1 = f1_score(all_labels, all_preds, average='macro')
                        
                        config = {
                            'hidden_dims': hidden_dims,
                            'dropout_rate': dropout,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'val_f1': val_f1
                        }
                        
                        results.append(config)
                        
                        if val_f1 > best_score:
                            best_score = val_f1
                            best_config = config
                            print(f"   ✓ New best F1: {val_f1:.4f}")
        
        print(f"\n✓ Best Validation F1 Score: {best_score:.4f}")
        print(f"✓ Best Configuration:")
        for param, value in best_config.items():
            if param != 'val_f1':
                print(f"   {param}: {value}")
        
        return {
            'best_params': best_config,
            'best_val_score': best_score,
            'all_results': results
        }


class DiseasePredictor:
    """
    Unified interface for training and evaluating disease prediction models
    With hyperparameter tuning support
    """
    
    def __init__(self, model_type: str, n_features: int, n_classes: int, 
                 output_dir: str = "models", use_enhanced_features: bool = True):
        self.model_type = model_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_enhanced_features = use_enhanced_features
        
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_params = None
        
    def build_model(self, params: Optional[Dict] = None):
        """Build model with optional custom parameters"""
        if params is None:
            params = self._get_default_params()
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric='mlogloss')
            
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                input_dim=self.n_features,
                num_classes=self.n_classes,
                **params
            ).to(self.device)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✓ Built {self.model_type} model")
        return self.model
    
    def _get_default_params(self) -> Dict:
        """Get default parameters if no tuning is performed"""
        defaults = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            'mlp': {
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.3
            }
        }
        return defaults.get(self.model_type, {})
    
    def train_with_tuning(self, X_train, y_train, X_val, y_val,
                         tune: bool = True, n_iter: int = 20):
        """
        Train model with optional hyperparameter tuning
        """
        if tune and self.model_type in ['random_forest', 'xgboost']:
            print(f"\n🔧 Performing hyperparameter tuning for {self.model_type}...")
            
            if self.model_type == 'random_forest':
                tuning_results = HyperparameterTuner.tune_random_forest(
                    X_train, y_train, X_val, y_val, n_iter=n_iter
                )
            else:
                tuning_results = HyperparameterTuner.tune_xgboost(
                    X_train, y_train, X_val, y_val, n_iter=n_iter
                )
            
            self.model = tuning_results['best_estimator']
            self.best_params = tuning_results['best_params']
            
            # Save tuning results
            tuning_path = self.output_dir / f'{self.model_type}_tuning_results.json'
            with open(tuning_path, 'w') as f:
                json.dump({
                    'best_params': tuning_results['best_params'],
                    'best_cv_score': float(tuning_results['best_cv_score']),
                    'val_score': float(tuning_results['val_score'])
                }, f, indent=2)
            
            print(f"✓ Tuning results saved to {tuning_path}")
            
        else:
            # Train with default or provided parameters
            self.build_model()
            
            if self.model_type in ['random_forest', 'xgboost']:
                self.train_classical(X_train, y_train, X_val, y_val)
            else:
                self.train_deep(X_train, y_train, X_val, y_val)
        
        return self.model
    
    def train_classical(self, X_train, y_train, X_val, y_val):
        """Train RandomForest or XGBoost"""
        print(f"\nTraining {self.model_type}...")
        
        if self.model_type == 'xgboost':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=10
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        val_acc = accuracy_score(y_val, self.model.predict(X_val))
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Check for overfitting
        if train_acc - val_acc > 0.1:
            print("⚠️  Possible overfitting detected (train-val gap > 10%)")
        
        return self.model
    
    def train_deep(self, X_train, y_train, X_val, y_val,
                   batch_size: int = 128,
                   epochs: int = 100,
                   lr: float = 0.001,
                   patience: int = 15):
        """Train MLP with early stopping"""
        print(f"\nTraining {self.model_type} on {self.device}...")
        
        # Create dataloaders
        train_dataset = SymptomDataset(X_train, y_train)
        val_dataset = SymptomDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 
                          self.output_dir / f'{self.model_type}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
        
        # Load best model
        self.model.load_state_dict(
            torch.load(self.output_dir / f'{self.model_type}_best.pth')
        )
        
        print(f"✓ Training complete. Best val loss: {best_val_loss:.4f}")
        return self.model
    
    def evaluate(self, X_test, y_test, class_names: List[str], 
                 k_values: List[int] = [1, 3, 5]) -> Dict:
        """Comprehensive evaluation"""
        print(f"\nEvaluating {self.model_type}...")
        
        if self.model_type == 'mlp':
            self.model.eval()
            test_dataset = SymptomDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = self.model(X_batch)
                    probs = F.softmax(outputs, dim=1)
                    
                    all_probs.append(probs.cpu().numpy())
                    _, predicted = torch.max(outputs, 1)
                    all_preds.append(predicted.cpu().numpy())
            
            y_pred = np.concatenate(all_preds)
            y_probs = np.concatenate(all_probs)
        else:
            y_pred = self.model.predict(X_test)
            y_probs = self.model.predict_proba(X_test)
        
        # Metrics
        metrics = {
            'model_name': self.model_type,
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Top-k accuracy with labels parameter to handle missing classes in test set
        for k in k_values:
            topk_acc = top_k_accuracy_score(y_test, y_probs, k=k, labels=np.arange(self.n_classes))
            metrics[f'top_{k}_accuracy'] = topk_acc
        
        # Print results
        print("\n" + "="*60)
        print(f"{self.model_type.upper()} EVALUATION RESULTS")
        print("="*60)
        for metric, value in metrics.items():
            if metric != 'model_name':
                print(f"{metric}: {value:.4f}")
        
        # Save metrics
        with open(self.output_dir / f'{self.model_type}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics, y_pred, y_probs
    
    def save_model(self):
        """Save trained model"""
        if self.model_type in ['random_forest', 'xgboost']:
            with open(self.output_dir / f'{self.model_type}.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        else:
            torch.save(self.model.state_dict(), 
                      self.output_dir / f'{self.model_type}.pth')
        
        print(f"✓ Model saved to {self.output_dir}")
    
    def load_model(self):
        """Load trained model"""
        if self.model_type in ['random_forest', 'xgboost']:
            with open(self.output_dir / f'{self.model_type}.pkl', 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model.load_state_dict(
                torch.load(self.output_dir / f'{self.model_type}.pth')
            )
        
        print(f"✓ Model loaded from {self.output_dir}")


# Training script with hyperparameter tuning
if __name__ == "__main__":
    # Load preprocessed data
    data = np.load('processed_data/data_splits.npz')
    with open('processed_data/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Check if enhanced features exist
    use_enhanced = 'X_train_enhanced' in data
    
    if use_enhanced:
        print("\n✓ Using ENHANCED features with advanced feature engineering")
        X_train = data['X_train_enhanced']
        X_val = data['X_val_enhanced']
        X_test = data['X_test_enhanced']
        X_train_scaled = data['X_train_enhanced_scaled']
        X_val_scaled = data['X_val_enhanced_scaled']
        X_test_scaled = data['X_test_enhanced_scaled']
        
        # Update n_features for enhanced
        n_features_enhanced = X_train.shape[1]
    else:
        print("\n⚠️  Using ORIGINAL features (run preprocessing with feature engineering)")
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        X_train_scaled = data['X_train_scaled']
        X_val_scaled = data['X_val_scaled']
        X_test_scaled = data['X_test_scaled']
        n_features_enhanced = metadata['n_features']
    
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    
    # Train all models with tuning
    all_metrics = []
    
    # 1. Random Forest with tuning
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST WITH HYPERPARAMETER TUNING")
    print("="*70)
    rf_predictor = DiseasePredictor('random_forest', n_features_enhanced, 
                                    metadata['n_classes'])
    rf_predictor.train_with_tuning(X_train, y_train, X_val, y_val, 
                                   tune=True, n_iter=15)
    rf_metrics, _, _ = rf_predictor.evaluate(X_test, y_test, metadata['classes'])
    rf_predictor.save_model()
    all_metrics.append(rf_metrics)
    
    # 2. XGBoost with tuning
    print("\n" + "="*70)
    print("TRAINING XGBOOST WITH HYPERPARAMETER TUNING")
    print("="*70)
    xgb_predictor = DiseasePredictor('xgboost', n_features_enhanced, 
                                     metadata['n_classes'])
    xgb_predictor.train_with_tuning(X_train, y_train, X_val, y_val, 
                                    tune=True, n_iter=15)
    xgb_metrics, _, _ = xgb_predictor.evaluate(X_test, y_test, metadata['classes'])
    xgb_predictor.save_model()
    all_metrics.append(xgb_metrics)
    
    # 3. MLP (use scaled features)
    print("\n" + "="*70)
    print("TRAINING MLP")
    print("="*70)
    mlp_predictor = DiseasePredictor('mlp', n_features_enhanced, 
                                     metadata['n_classes'])
    mlp_predictor.build_model()
    mlp_predictor.train_deep(X_train_scaled, y_train, X_val_scaled, y_val,
                            batch_size=128, epochs=100, lr=0.001, patience=15)
    mlp_metrics, _, _ = mlp_predictor.evaluate(X_test_scaled, y_test, 
                                               metadata['classes'])
    mlp_predictor.save_model()
    all_metrics.append(mlp_metrics)
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.set_index('model_name')
    print(comparison_df[['accuracy', 'top_3_accuracy', 'top_5_accuracy', 'macro_f1']])
    
    # Find best model
    best_model = comparison_df['top_3_accuracy'].idxmax()
    best_score = comparison_df['top_3_accuracy'].max()
    
    print(f"\n🏆 BEST MODEL: {best_model} (Top-3 Accuracy: {best_score:.4f})")
    
    print("\n✓ All models trained with hyperparameter tuning and evaluated!")
    print("\nFeature Engineering Impact:")
    if use_enhanced:
        print(f"  Total features used: {n_features_enhanced}")
        print(f"  Original features: {metadata['n_features']}")
        print(f"  Additional engineered features: {n_features_enhanced - metadata['n_features']}")
    else:
        print("  Run preprocessing again to enable enhanced features")