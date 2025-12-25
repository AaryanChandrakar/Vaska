"""
Disease Prediction Inference Pipeline
Loads trained models and makes predictions from symptom inputs
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DiseaseInference:
    """
    Inference pipeline for disease prediction from symptoms.
    Supports multiple model types and handles feature engineering.
    """
    
    def __init__(self, 
                 model_path: str = 'models/xgboost.pkl',
                 model_type: str = 'xgboost',
                 processed_data_dir: str = 'processed_data'):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model ('random_forest', 'xgboost', or 'mlp')
            processed_data_dir: Directory containing preprocessors and metadata
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.processed_data_dir = Path(processed_data_dir)
        
        # Will be loaded
        self.model = None
        self.symptom_to_id = None
        self.id_to_symptom = None
        self.disease_encoder = None
        self.scaler = None
        self.pca = None
        self.feature_metadata = None
        self.symptom_columns = None
        
        # Load all components
        self.load_model()
    
    def load_model(self):
        """Load trained model and all required preprocessors"""
        print(f"Loading inference pipeline for {self.model_type}...")
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.model_type == 'mlp':
            import torch
            self.model = torch.load(self.model_path)
            self.model.eval()
            print("  ✓ Loaded PyTorch MLP model")
        else:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"  ✓ Loaded {self.model_type} model")
        
        # Load symptom lexicon
        lexicon_path = self.processed_data_dir / 'symptom_lexicon.json'
        with open(lexicon_path, 'r') as f:
            lexicon = json.load(f)
            self.symptom_to_id = {k: int(v) for k, v in lexicon['symptom_to_id'].items()}
            self.id_to_symptom = {int(k): v for k, v in lexicon['id_to_symptom'].items()}
            self.symptom_columns = lexicon['symptoms']
        print(f"  ✓ Loaded {len(self.symptom_columns)} symptoms")
        
        # Load disease encoder
        encoder_path = self.processed_data_dir / 'disease_encoder.pkl'
        with open(encoder_path, 'rb') as f:
            self.disease_encoder = pickle.load(f)
        print(f"  ✓ Loaded {len(self.disease_encoder.classes_)} diseases")
        
        # Load scaler
        scaler_path = self.processed_data_dir / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  ✓ Loaded feature scaler")
        
        # Load PCA if exists
        pca_path = self.processed_data_dir / 'pca_model.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            print("  ✓ Loaded PCA model")
        
        # Load feature metadata
        metadata_path = self.processed_data_dir / 'feature_metadata.json'
        with open(metadata_path, 'r') as f:
            self.feature_metadata = json.load(f)
        
        print("Inference pipeline loaded successfully!\n")
    
    def create_symptom_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Convert list of symptom names to binary vector
        
        Args:
            symptoms: List of symptom names (strings)
            
        Returns:
            Binary vector of shape (n_symptoms,)
        """
        vector = np.zeros(len(self.symptom_columns))
        
        # Normalize symptom names for matching
        symptoms_normalized = [s.lower().strip().replace(' ', '_') for s in symptoms]
        
        matched_symptoms = []
        unmatched_symptoms = []
        
        for symptom in symptoms_normalized:
            # Try exact match first
            if symptom in self.symptom_to_id:
                idx = self.symptom_to_id[symptom]
                vector[idx] = 1
                matched_symptoms.append(symptom)
            else:
                # Try fuzzy matching
                best_match = None
                best_score = 0
                
                for known_symptom in self.symptom_columns:
                    # Simple similarity: check if symptom is substring or vice versa
                    if symptom in known_symptom or known_symptom in symptom:
                        score = len(symptom) / max(len(symptom), len(known_symptom))
                        if score > best_score:
                            best_score = score
                            best_match = known_symptom
                
                if best_match and best_score > 0.6:
                    idx = self.symptom_to_id[best_match]
                    vector[idx] = 1
                    matched_symptoms.append(f"{symptom} → {best_match}")
                else:
                    unmatched_symptoms.append(symptom)
        
        if matched_symptoms:
            print(f"Matched {len(matched_symptoms)} symptoms:")
            for s in matched_symptoms:
                print(f"  • {s}")
        
        if unmatched_symptoms:
            print(f"\nWarning: Could not match {len(unmatched_symptoms)} symptoms:")
            for s in unmatched_symptoms:
                print(f"  ✗ {s}")
        
        return vector
    
    def engineer_features(self, symptom_vector: np.ndarray) -> np.ndarray:
        """
        Apply same feature engineering as training
        
        Args:
            symptom_vector: Raw symptom binary vector
            
        Returns:
            Enhanced feature vector
        """
        X = symptom_vector.reshape(1, -1)
        n_features = X.shape[1]
        
        # Statistical features
        symptom_count = X.sum(axis=1).reshape(-1, 1)
        symptom_density = (X.sum(axis=1) / n_features).reshape(-1, 1)
        
        symptom_freq = X.sum(axis=0) / X.shape[0]
        rare_threshold = np.percentile(symptom_freq, 25) if symptom_freq.sum() > 0 else 0
        common_threshold = np.percentile(symptom_freq, 75) if symptom_freq.sum() > 0 else 1
        
        rare_symptoms = (X * (symptom_freq < rare_threshold)).sum(axis=1).reshape(-1, 1)
        common_symptoms = (X * (symptom_freq > common_threshold)).sum(axis=1).reshape(-1, 1)
        
        # System groups - simplified
        system_features = np.zeros((1, 7))  # 7 systems
        
        # PCA features
        if self.pca is not None:
            pca_features = self.pca.transform(X)
        else:
            pca_features = np.zeros((1, 50))
        
        # Co-occurrence features (placeholder for inference)
        cooccurrence_features = np.zeros((1, 50))
        
        # Combine all features
        enhanced = np.hstack([
            X,
            symptom_count,
            symptom_density,
            rare_symptoms,
            common_symptoms,
            system_features,
            cooccurrence_features,
            pca_features
        ])
        
        return enhanced[0]
    
    def predict_from_symptoms(self, 
                              symptoms: List[str], 
                              top_k: int = 5,
                              use_enhanced_features: bool = False) -> List[Tuple[str, float]]:
        """
        Predict diseases from list of symptom names
        
        Args:
            symptoms: List of symptom strings
            top_k: Number of top predictions to return
            use_enhanced_features: Whether to use enhanced features (default: False for simplicity)
            
        Returns:
            List of (disease_name, confidence) tuples, sorted by confidence
        """
        print(f"\n{'='*60}")
        print(f"PREDICTING FROM {len(symptoms)} SYMPTOMS")
        print(f"{'='*60}\n")
        
        # Create symptom vector
        symptom_vector = self.create_symptom_vector(symptoms)
        
        # Check if any symptoms matched
        if symptom_vector.sum() == 0:
            print("\n⚠️  No symptoms matched. Cannot make prediction.")
            return []
        
        # Predict
        return self.predict_from_vector(symptom_vector, top_k, use_enhanced_features)
    
    def predict_from_vector(self,
                           symptom_vector: np.ndarray,
                           top_k: int = 5,
                           use_enhanced_features: bool = False) -> List[Tuple[str, float]]:
        """
        Predict diseases from symptom vector
        
        Args:
            symptom_vector: Binary symptom vector
            top_k: Number of top predictions to return
            use_enhanced_features: Whether to use enhanced features
            
        Returns:
            List of (disease_name, confidence) tuples
        """
        # Prepare features
        if use_enhanced_features:
            features = self.engineer_features(symptom_vector)
        else:
            features = symptom_vector
        
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions based on model type
        if self.model_type == 'mlp':
            import torch
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                logits = self.model(features_tensor)
                probabilities = torch.softmax(logits, dim=1).numpy()[0]
        else:
            # Random Forest or XGBoost
            probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            disease = self.disease_encoder.classes_[idx]
            confidence = float(probabilities[idx])
            predictions.append((disease, confidence))
        
        # Display results
        print(f"\n{'='*60}")
        print(f"TOP {top_k} PREDICTIONS")
        print(f"{'='*60}\n")
        
        for i, (disease, conf) in enumerate(predictions, 1):
            bar_length = int(conf * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{i}. {disease:30s} {bar} {conf:.1%}")
        
        print(f"\n{'='*60}\n")
        
        return predictions
    
    def get_all_symptoms(self) -> List[str]:
        """Get list of all known symptoms"""
        return self.symptom_columns.copy()
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all known diseases"""
        return self.disease_encoder.classes_.tolist()


# Example usage
if __name__ == "__main__":
    # Initialize inference
    inference = DiseaseInference(
        model_path='models/xgboost.pkl',
        model_type='xgboost'
    )
    
    # Test prediction
    test_symptoms = [
        'fever',
        'cough',
        'fatigue',
        'headache'
    ]
    
    predictions = inference.predict_from_symptoms(test_symptoms, top_k=5)
    
    print("\\n⚕️  Please consult a healthcare professional for proper diagnosis.")
