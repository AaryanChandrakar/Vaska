"""
Disease Prediction Model - Training Script
Simple script to train and evaluate ML models for disease prediction
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

def main():
    print("="*70)
    print("  Disease Prediction Model - Training Pipeline")
    print("="*70)
    print()
    
    # Step 1: Data Preprocessing
    print("[Step 1/5] Data Preprocessing")
    print("-"*70)
    
    from symptom_processor import SymptomDataProcessor
    
    processor = SymptomDataProcessor(
        data_path='data/disease_symptom_data.csv',
        output_dir='processed_data'
    )
    
    print("Running preprocessing pipeline...")
    processor.process_pipeline()
    print("✓ Data preprocessing complete\n")
    
    # Step 2: Load Processed Data
    print("[Step 2/5] Loading Processed Data")
    print("-"*70)
    
    data = np.load('processed_data/data_splits.npz')
    with open('processed_data/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_val_scaled = data['X_val_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {metadata['n_features']}")
    print(f"✓ Classes: {metadata['n_classes']}\n")
    
    # Step 3: Train Models
    print("[Step 3/5] Training Models")
    print("-"*70)
    
    from disease_models import DiseasePredictor
    
    models_to_train = ['random_forest', 'xgboost', 'mlp']
    
    for model_type in models_to_train:
        print(f"\nTraining {model_type.upper()}...")
        print("="*50)
        
        predictor = DiseasePredictor(
            model_type,
            metadata['n_features'],
            metadata['n_classes']
        )
        
        predictor.build_model()
        
        if model_type == 'mlp':
            predictor.train_deep(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                batch_size=256,
                epochs=50,
                patience=10
            )
            X_eval = X_test_scaled
        else:
            predictor.train_classical(X_train, y_train, X_val, y_val)
            X_eval = X_test
        
        # Evaluate
        metrics, _, _ = predictor.evaluate(
            X_eval, y_test,
            metadata['classes']
        )
        
        # Save model
        predictor.save_model()
        
        print(f"\n✓ {model_type} training complete")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Top-3 Accuracy: {metrics['top_3_accuracy']:.3f}")
        print(f"  F1 Score: {metrics['macro_f1']:.3f}")
    
    # Step 4: Model Comparison
    print("\n[Step 4/5] Model Comparison")
    print("-"*70)
    
    results = []
    for model_type in models_to_train:
        metrics_file = f'models/{model_type}_metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results.append({
                    'model': model_type,
                    'accuracy': metrics['accuracy'],
                    'top_3': metrics['top_3_accuracy'],
                    'f1': metrics['macro_f1']
                })
    
    print("\nModel Performance Summary:")
    print(f"{'Model':<15} {'Accuracy':<12} {'Top-3 Acc':<12} {'F1 Score':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['model']:<15} {r['accuracy']:<12.3f} {r['top_3']:<12.3f} {r['f1']:<10.3f}")
    
    # Step 5: Run Evaluation Framework
    print("\n[Step 5/5] Running Evaluation Framework")
    print("-"*70)
    
    from evaluation_framework import EvaluationFramework
    
    eval_framework = EvaluationFramework()
    eval_framework.run_full_evaluation()
    
    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  • Check results in models/ directory")
    print("  • Review evaluation_results/ for detailed analysis")
    print("  • Use trained models for predictions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
