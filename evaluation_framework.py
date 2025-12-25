"""
Symptom Checker - Comprehensive Evaluation Framework
Tests disease prediction, NLU, safety (red-flag recall), and calibration
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation suite for symptom checker models
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def evaluate_disease_model(self, 
                              model,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              class_names: List[str],
                              model_name: str) -> Dict:
        """
        Evaluate disease prediction model
        
        Metrics:
        - Top-k accuracy (k=1,3,5,10)
        - Macro/Weighted F1
        - Precision/Recall
        - Calibration
        - Per-class performance
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        else:
            # PyTorch model
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                logits = model(X_tensor)
                y_proba = torch.softmax(logits, dim=1).numpy()
                y_pred = np.argmax(y_proba, axis=1)
        
        # Core metrics
        metrics = {
            'model_name': model_name,
            'test_size': len(y_test),
            'n_classes': len(class_names),
            
            # Accuracy metrics
            'accuracy': accuracy_score(y_test, y_pred),
            'top_3_accuracy': top_k_accuracy_score(y_test, y_proba, k=3),
            'top_5_accuracy': top_k_accuracy_score(y_test, y_proba, k=5),
            'top_10_accuracy': top_k_accuracy_score(y_test, y_proba, k=10),
            
            # F1 scores
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            
            # Precision/Recall
            'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        # Print core metrics
        print("\n📊 CORE METRICS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        
        # Calibration analysis
        print("\n🎯 CALIBRATION ANALYSIS:")
        calibration_metrics = self._analyze_calibration(y_test, y_proba, model_name)
        metrics.update(calibration_metrics)
        
        # Overfitting check
        confidence = y_proba.max(axis=1).mean()
        if confidence > 0.9 and metrics['accuracy'] < 0.8:
            print("⚠️  WARNING: Model appears overconfident!")
            print(f"   Avg confidence: {confidence:.3f} but accuracy: {metrics['accuracy']:.3f}")
            metrics['overconfident'] = True
        else:
            metrics['overconfident'] = False
        
        # Per-class performance
        print("\n📋 PER-CLASS PERFORMANCE (Top 10 worst):")
        per_class = self._per_class_analysis(y_test, y_pred, class_names)
        
        # Save confusion matrix for high-frequency classes
        self._plot_confusion_matrix_subset(y_test, y_pred, class_names, model_name)
        
        # Save full report
        report_path = self.output_dir / f"{model_name}_evaluation.json"
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Detailed report saved to: {report_path}")
        
        return metrics
    
    def _analyze_calibration(self, y_true, y_proba, model_name: str) -> Dict:
        """
        Analyze model calibration
        Well-calibrated models: predicted probability matches actual probability
        """
        # Get predicted probabilities for predicted class
        y_pred = np.argmax(y_proba, axis=1)
        pred_probs = y_proba[np.arange(len(y_pred)), y_pred]
        
        # Create calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                (y_true == y_pred).astype(int),
                pred_probs,
                n_bins=10,
                strategy='quantile'
            )
            
            # Expected Calibration Error (ECE)
            ece = np.abs(prob_true - prob_pred).mean()
            
            print(f"Expected Calibration Error (ECE): {ece:.4f}")
            print(f"Average Confidence: {pred_probs.mean():.4f}")
            
            # Plot calibration curve
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            plt.plot(prob_pred, prob_true, 's-', label=f'{model_name}')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Probability')
            plt.title(f'Calibration Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / f'{model_name}_calibration.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return {
                'ece': ece,
                'avg_confidence': float(pred_probs.mean()),
                'calibrated': ece < 0.1
            }
        
        except Exception as e:
            print(f"Could not compute calibration: {e}")
            return {'ece': None, 'avg_confidence': float(pred_probs.mean())}
    
    def _per_class_analysis(self, y_true, y_pred, class_names: List[str]) -> pd.DataFrame:
        """Analyze per-class performance"""
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                      output_dict=True, zero_division=0)
        
        # Convert to DataFrame
        df = pd.DataFrame(report).T
        df = df[df.index.isin(class_names)]  # Only disease classes
        
        # Sort by F1 score
        df_sorted = df.sort_values('f1-score')
        
        # Show worst performers
        print(df_sorted.head(10)[['precision', 'recall', 'f1-score', 'support']])
        
        # Save full report
        df_sorted.to_csv(self.output_dir / 'per_class_performance.csv')
        
        return df_sorted
    
    def _plot_confusion_matrix_subset(self, y_true, y_pred, class_names: List[str], 
                                     model_name: str, top_n: int = 20):
        """Plot confusion matrix for most frequent classes"""
        # Find most frequent classes
        unique, counts = np.unique(y_true, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_n:]]
        
        # Filter to top classes
        mask = np.isin(y_true, top_classes)
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_sub, y_pred_sub, labels=top_classes)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=False, cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name} (Top {top_n} Classes)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_confusion_matrix.png', dpi=150)
        plt.close()
    
    def evaluate_nlu(self, nlu_model, test_data: List[Dict]) -> Dict:
        """
        Evaluate NLU symptom extraction
        
        Metrics:
        - Symptom extraction F1 (multi-label)
        - Intent classification F1
        """
        print(f"\n{'='*60}")
        print("EVALUATING NLU MODULE")
        print(f"{'='*60}")
        
        all_true_symptoms = []
        all_pred_symptoms = []
        all_true_intents = []
        all_pred_intents = []
        
        for sample in test_data:
            text = sample['text']
            true_symptoms = sample['symptoms']  # Binary vector
            true_intent = sample['intent']
            
            # Predict
            result = nlu_model.extract(text)
            
            # Convert to binary vector
            pred_symptoms = np.zeros_like(true_symptoms)
            for symp_id in result.get('symptom_ids', []):
                if symp_id < len(pred_symptoms):
                    pred_symptoms[symp_id] = 1
            
            all_true_symptoms.append(true_symptoms)
            all_pred_symptoms.append(pred_symptoms)
            all_true_intents.append(true_intent)
            all_pred_intents.append(result['intent'])
        
        # Calculate metrics
        all_true_symptoms = np.array(all_true_symptoms)
        all_pred_symptoms = np.array(all_pred_symptoms)
        
        symptom_f1 = f1_score(all_true_symptoms, all_pred_symptoms, 
                             average='macro', zero_division=0)
        
        intent_f1 = f1_score(all_true_intents, all_pred_intents, average='macro')
        
        metrics = {
            'symptom_extraction_f1': symptom_f1,
            'intent_classification_f1': intent_f1,
            'test_size': len(test_data)
        }
        
        print(f"\n📊 NLU METRICS:")
        print(f"Symptom Extraction F1: {symptom_f1:.4f}")
        print(f"Intent Classification F1: {intent_f1:.4f}")
        
        return metrics
    
    def evaluate_red_flag_recall(self, red_flag_kb, test_cases: List[Dict]) -> Dict:
        """
        Critical: Test red-flag recall (never miss emergencies!)
        
        Test cases should include:
        - True emergencies (must all be caught)
        - Non-emergencies (should not trigger false alarms)
        """
        print(f"\n{'='*60}")
        print("EVALUATING RED-FLAG SAFETY")
        print(f"{'='*60}")
        
        true_emergencies = [case for case in test_cases if case['is_emergency']]
        non_emergencies = [case for case in test_cases if not case['is_emergency']]
        
        # Recall on emergencies (most critical!)
        emergency_detected = 0
        emergency_missed = []
        
        for case in true_emergencies:
            result = red_flag_kb.check_emergency(case['symptoms'])
            if result['is_emergency']:
                emergency_detected += 1
            else:
                emergency_missed.append(case)
        
        recall = emergency_detected / len(true_emergencies) if true_emergencies else 0
        
        # False positive rate
        false_positives = 0
        for case in non_emergencies:
            result = red_flag_kb.check_emergency(case['symptoms'])
            if result['is_emergency']:
                false_positives += 1
        
        fpr = false_positives / len(non_emergencies) if non_emergencies else 0
        
        metrics = {
            'emergency_recall': recall,
            'false_positive_rate': fpr,
            'emergencies_caught': emergency_detected,
            'emergencies_missed': len(emergency_missed),
            'total_emergencies': len(true_emergencies)
        }
        
        print(f"\n🚨 SAFETY METRICS:")
        print(f"Emergency Recall: {recall:.4f} ({emergency_detected}/{len(true_emergencies)})")
        print(f"False Positive Rate: {fpr:.4f}")
        
        if recall < 1.0:
            print(f"\n⚠️  CRITICAL: {len(emergency_missed)} emergencies were MISSED!")
            print("Missed cases:")
            for case in emergency_missed:
                print(f"  - {case['symptoms']}")
        
        if recall == 1.0:
            print("\n✓ PERFECT: All emergencies detected!")
        
        return metrics
    
    def compare_models(self, metrics_list: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        """
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        df = pd.DataFrame(metrics_list)
        df = df.set_index('model_name')
        
        # Select key metrics
        key_metrics = ['accuracy', 'top_3_accuracy', 'top_5_accuracy', 
                      'macro_f1', 'ece', 'avg_confidence']
        
        comparison = df[key_metrics]
        print(comparison.to_string())
        
        # Save comparison
        comparison.to_csv(self.output_dir / 'model_comparison.csv')
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top-k accuracy
        axes[0, 0].bar(comparison.index, comparison['top_3_accuracy'])
        axes[0, 0].set_title('Top-3 Accuracy')
        axes[0, 0].set_ylim([0, 1])
        
        # F1 Score
        axes[0, 1].bar(comparison.index, comparison['macro_f1'])
        axes[0, 1].set_title('Macro F1 Score')
        axes[0, 1].set_ylim([0, 1])
        
        # Calibration
        axes[1, 0].bar(comparison.index, comparison['ece'])
        axes[1, 0].set_title('Expected Calibration Error (Lower is Better)')
        axes[1, 0].axhline(y=0.1, color='r', linestyle='--', label='Threshold')
        axes[1, 0].legend()
        
        # Confidence
        axes[1, 1].bar(comparison.index, comparison['avg_confidence'])
        axes[1, 1].set_title('Average Prediction Confidence')
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return comparison


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    # Load test data
    data = np.load('processed_data/data_splits.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    with open('processed_data/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load models
    with open('processed_data/disease_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Evaluate all models
    all_metrics = []
    
    for model_type in ['random_forest', 'xgboost', 'mlp']:
        print(f"\n{'='*60}")
        print(f"Loading {model_type}...")
        
        # Load model
        from disease_models import DiseasePredictor
        predictor = DiseasePredictor(model_type, metadata['n_features'], 
                                     metadata['n_classes'])
        predictor.load_model()
        
        # Evaluate
        if model_type == 'mlp':
            X_test_eval = data['X_test_scaled']
        else:
            X_test_eval = X_test
        
        metrics = evaluator.evaluate_disease_model(
            predictor.model,
            X_test_eval,
            y_test,
            metadata['classes'],
            model_type
        )
        
        all_metrics.append(metrics)
    
    # Compare models
    comparison = evaluator.compare_models(all_metrics)
    
    # Red-flag safety test
    print("\n" + "="*60)
    print("Creating red-flag test cases...")
    
    test_cases = [
        {'symptoms': ['severe chest pain', 'sweating'], 'is_emergency': True},
        {'symptoms': ['face drooping', 'arm weakness'], 'is_emergency': True},
        {'symptoms': ['cannot breathe', 'blue lips'], 'is_emergency': True},
        {'symptoms': ['minor headache'], 'is_emergency': False},
        {'symptoms': ['slight cough'], 'is_emergency': False},
    ]
    
    from rag_redflag import RedFlagKnowledgeBase
    kb = RedFlagKnowledgeBase()
    kb.load_index()
    
    safety_metrics = evaluator.evaluate_red_flag_recall(kb, test_cases)
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {evaluator.output_dir}/")