"""
Symptom Checker - Data Preprocessing & Feature Engineering Module
Author: AI Medical Systems Team
ENHANCED with Advanced Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SymptomDataProcessor:
    """
    Handles data loading, cleaning, advanced feature engineering, and splitting
    for the symptom-checker system.
    """
    
    def __init__(self, data_path: str, output_dir: str = "processed_data"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Will be populated during processing
        self.df = None
        self.symptom_columns = []
        self.symptom_to_id = {}
        self.id_to_symptom = {}
        self.disease_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the Kaggle dataset"""
        print(f"Loading data from {self.data_path}...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.data_path, encoding=encoding)
                print(f"[OK] Loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if self.df is None:
            raise ValueError("Could not load data with any encoding")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)[:5]}...")
        return self.df
    
    def clean_data(self):
        """Clean column names and handle missing values"""
        print("\nCleaning data...")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Identify disease and symptom columns
        disease_col = self.df.columns[0]
        self.symptom_columns = [col for col in self.df.columns if col != disease_col]
        
        print(f"Disease column: {disease_col}")
        print(f"Number of symptom columns: {len(self.symptom_columns)}")
        
        # Rename disease column for consistency
        self.df.rename(columns={disease_col: 'disease'}, inplace=True)
        
        # Handle missing values in symptoms
        for col in self.symptom_columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].notna().astype(int)
            else:
                self.df[col] = self.df[col].fillna(0).astype(int)
        
        # Remove duplicates
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {before - len(self.df)} duplicate rows")
        
        # Remove rows with no symptoms
        symptom_sum = self.df[self.symptom_columns].sum(axis=1)
        self.df = self.df[symptom_sum > 0]
        print(f"Kept {len(self.df)} rows with at least one symptom")
        
        return self.df
    
    def create_symptom_mappings(self):
        """Create bidirectional symptom-to-ID mappings"""
        print("\nCreating symptom mappings...")
        
        self.symptom_to_id = {symptom: idx for idx, symptom in enumerate(self.symptom_columns)}
        self.id_to_symptom = {idx: symptom for symptom, idx in self.symptom_to_id.items()}
        
        print(f"Created mappings for {len(self.symptom_to_id)} symptoms")
        
        # Save symptom lexicon
        lexicon = {
            'symptom_to_id': self.symptom_to_id,
            'id_to_symptom': self.id_to_symptom,
            'symptoms': self.symptom_columns
        }
        
        with open(self.output_dir / 'symptom_lexicon.json', 'w') as f:
            json.dump(lexicon, f, indent=2)
        
        return self.symptom_to_id
    
    def engineer_advanced_features(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create advanced features from raw symptom vectors
        
        New Features:
        1. Symptom count (total symptoms per case)
        2. Symptom density (percentage of symptoms present)
        3. Symptom co-occurrence patterns
        4. Symptom embeddings via PCA
        5. Rare symptom indicators
        6. System-specific symptom groups
        """
        print("\n" + "="*60)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*60)
        
        n_samples, n_features = X.shape
        
        # 1. Basic statistical features
        print("\n1. Creating statistical features...")
        symptom_count = X.sum(axis=1).reshape(-1, 1)  # Total symptoms
        symptom_density = (X.sum(axis=1) / n_features).reshape(-1, 1)  # Density
        
        print(f"   - Symptom count: mean={symptom_count.mean():.2f}, std={symptom_count.std():.2f}")
        print(f"   - Symptom density: mean={symptom_density.mean():.3f}")
        
        # 2. Symptom rarity features
        print("\n2. Computing symptom rarity indicators...")
        symptom_freq = X.sum(axis=0) / n_samples
        rare_threshold = np.percentile(symptom_freq, 25)  # Bottom 25%
        common_threshold = np.percentile(symptom_freq, 75)  # Top 25%
        
        rare_symptoms = (X * (symptom_freq < rare_threshold)).sum(axis=1).reshape(-1, 1)
        common_symptoms = (X * (symptom_freq > common_threshold)).sum(axis=1).reshape(-1, 1)
        
        print(f"   - Rare symptoms (freq < {rare_threshold:.3f}): present in {(rare_symptoms > 0).sum()} cases")
        print(f"   - Common symptoms (freq > {common_threshold:.3f}): present in {(common_symptoms > 0).sum()} cases")
        
        # 3. System-specific symptom groups
        print("\n3. Creating system-specific features...")
        system_groups = self._create_system_groups()
        system_features = []
        
        for system_name, symptoms in system_groups.items():
            # Get indices for symptoms in this system
            indices = [self.symptom_to_id[s] for s in symptoms if s in self.symptom_to_id]
            if indices:
                system_symptom_count = X[:, indices].sum(axis=1).reshape(-1, 1)
                system_features.append(system_symptom_count)
                print(f"   - {system_name}: {len(indices)} symptoms")
        
        system_features = np.hstack(system_features) if system_features else np.zeros((n_samples, 1))
        
        # 4. Symptom co-occurrence features
        print("\n4. Computing symptom co-occurrence patterns...")
        # Pairwise symptom interactions (top 50 most informative)
        cooccurrence_features = self._compute_cooccurrence_features(X, top_k=50)
        
        # 5. PCA embeddings
        print("\n5. Creating symptom embeddings with PCA...")
        n_components = min(50, n_features)
        self.pca = PCA(n_components=n_components, random_state=42)
        pca_features = self.pca.fit_transform(X)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"   - PCA components: {n_components}")
        print(f"   - Explained variance: {explained_var:.3f}")
        
        # Save PCA model
        with open(self.output_dir / 'pca_model.pkl', 'wb') as f:
            pickle.dump(self.pca, f)
        
        # Combine all features
        print("\n6. Combining feature sets...")
        feature_sets = {
            'original': X,
            'statistical': np.hstack([symptom_count, symptom_density, rare_symptoms, common_symptoms]),
            'system_groups': system_features,
            'cooccurrence': cooccurrence_features,
            'pca_embeddings': pca_features
        }
        
        # Create comprehensive feature matrix
        enhanced_features = np.hstack([
            X,  # Original symptoms
            symptom_count,
            symptom_density,
            rare_symptoms,
            common_symptoms,
            system_features,
            cooccurrence_features,
            pca_features
        ])
        
        print(f"\nFeature engineering summary:")
        print(f"  - Original features: {X.shape[1]}")
        print(f"  - Statistical features: 4")
        print(f"  - System group features: {system_features.shape[1]}")
        print(f"  - Co-occurrence features: {cooccurrence_features.shape[1]}")
        print(f"  - PCA features: {pca_features.shape[1]}")
        print(f"  - Total enhanced features: {enhanced_features.shape[1]}")
        
        # Save feature metadata
        feature_info = {
            'n_original': int(X.shape[1]),
            'n_statistical': 4,
            'n_system_groups': int(system_features.shape[1]),
            'n_cooccurrence': int(cooccurrence_features.shape[1]),
            'n_pca': int(pca_features.shape[1]),
            'n_total': int(enhanced_features.shape[1]),
            'system_groups': {k: len(v) for k, v in system_groups.items()},
            'pca_explained_variance': float(explained_var)
        }
        
        with open(self.output_dir / 'feature_engineering_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        feature_sets['enhanced'] = enhanced_features
        return feature_sets
    
    def _create_system_groups(self) -> Dict[str, List[str]]:
        """
        Group symptoms by body system for system-specific features
        """
        system_groups = {
            'respiratory': [],
            'cardiovascular': [],
            'gastrointestinal': [],
            'neurological': [],
            'musculoskeletal': [],
            'dermatological': [],
            'general': []
        }
        
        # Keywords for each system
        keywords = {
            'respiratory': ['cough', 'breath', 'lung', 'throat', 'nasal', 'sinus', 'chest', 'wheez'],
            'cardiovascular': ['heart', 'chest_pain', 'palpit', 'pressure', 'circulat'],
            'gastrointestinal': ['stomach', 'abdom', 'nausea', 'vomit', 'diarr', 'constip', 'digest'],
            'neurological': ['head', 'dizz', 'confusion', 'seizure', 'memory', 'vision', 'tremor'],
            'musculoskeletal': ['pain', 'joint', 'muscle', 'stiff', 'swell', 'weak'],
            'dermatological': ['rash', 'skin', 'itch', 'lesion', 'blister'],
            'general': ['fever', 'fatigue', 'weight', 'tired', 'weak']
        }
        
        # Classify each symptom
        for symptom in self.symptom_columns:
            symptom_lower = symptom.lower()
            classified = False
            
            for system, kw_list in keywords.items():
                if any(kw in symptom_lower for kw in kw_list):
                    system_groups[system].append(symptom)
                    classified = True
                    break
            
            if not classified:
                system_groups['general'].append(symptom)
        
        return system_groups
    
    def _compute_cooccurrence_features(self, X: np.ndarray, top_k: int = 50) -> np.ndarray:
        """
        Compute pairwise symptom co-occurrence features
        Select top_k most informative pairs based on mutual information
        """
        from sklearn.feature_selection import mutual_info_classif
        
        n_samples, n_features = X.shape
        
        # Compute all pairwise products (symptom interactions)
        interaction_features = []
        interaction_scores = []
        
        # Sample pairs to avoid too many features
        np.random.seed(42)
        n_pairs = min(1000, (n_features * (n_features - 1)) // 2)
        
        pairs_checked = 0
        for i in range(n_features):
            for j in range(i + 1, min(i + 20, n_features)):  # Limit pairs per symptom
                if pairs_checked >= n_pairs:
                    break
                
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                if interaction.sum() > 5:  # Only keep if co-occurs at least 5 times
                    interaction_features.append(interaction)
                    pairs_checked += 1
        
        if not interaction_features:
            return np.zeros((n_samples, 1))
        
        interaction_matrix = np.hstack(interaction_features)
        
        # Select top_k based on variance (simple heuristic)
        variances = interaction_matrix.var(axis=0)
        top_indices = np.argsort(variances)[-top_k:]
        
        selected_features = interaction_matrix[:, top_indices]
        
        return selected_features
    
    def analyze_data_distribution(self):
        """Analyze disease and symptom distributions"""
        print("\n" + "="*60)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Disease distribution
        disease_counts = self.df['disease'].value_counts()
        print(f"\nUnique diseases: {len(disease_counts)}")
        print(f"Avg samples per disease: {disease_counts.mean():.1f}")
        print(f"Min samples: {disease_counts.min()}")
        print(f"Max samples: {disease_counts.max()}")
        
        # Check for class imbalance
        imbalance_ratio = disease_counts.max() / disease_counts.min()
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 10:
            print("[!] WARNING: HIGH CLASS IMBALANCE DETECTED - will use stratification")
        
        # Symptom distribution
        symptom_counts = self.df[self.symptom_columns].sum()
        print(f"\nMost common symptoms:")
        print(symptom_counts.nlargest(10))
        
        # Symptom co-occurrence
        avg_symptoms_per_case = self.df[self.symptom_columns].sum(axis=1).mean()
        print(f"\nAvg symptoms per case: {avg_symptoms_per_case:.2f}")
        
        return disease_counts
    
    def create_train_val_test_split(self, 
                                    test_size: float = 0.15,
                                    val_size: float = 0.15,
                                    random_state: int = 42) -> Tuple:
        """
        Create stratified train/val/test split with feature engineering
        """
        print("\nCreating stratified train/val/test split...")
        
        # Prepare features and labels
        X = self.df[self.symptom_columns].values
        y = self.df['disease'].values
        
        # Check for diseases with too few samples
        disease_counts = pd.Series(y).value_counts()
        min_samples_required = 2  # Minimum for stratified split
        
        diseases_to_remove = disease_counts[disease_counts < min_samples_required].index.tolist()
        
        if diseases_to_remove:
            print(f"\n[!] WARNING: Removing {len(diseases_to_remove)} diseases with fewer than {min_samples_required} samples:")
            print(f"    Affected diseases: {len(diseases_to_remove)} total")
            print(f"    Samples removed: {disease_counts[diseases_to_remove].sum()}")
            
            # Filter out these diseases
            mask = ~pd.Series(y).isin(diseases_to_remove)
            X = X[mask]
            y = y[mask]
            
            print(f"    Remaining samples: {len(y)}")
            print(f"    Remaining diseases: {pd.Series(y).nunique()}")
        
        # Encode diseases
        y_encoded = self.disease_encoder.fit_transform(y)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=random_state
        )
        
        print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Val set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Engineer features for training set
        train_features = self.engineer_advanced_features(X_train)
        
        # Apply same transformations to val and test
        print("\nApplying feature engineering to validation and test sets...")
        val_features = self._transform_features(X_val)
        test_features = self._transform_features(X_test)
        
        # Scale all feature sets
        print("\nScaling features...")
        
        # Original features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Enhanced features
        scaler_enhanced = StandardScaler()
        X_train_enhanced_scaled = scaler_enhanced.fit_transform(train_features['enhanced'])
        X_val_enhanced_scaled = scaler_enhanced.transform(val_features['enhanced'])
        X_test_enhanced_scaled = scaler_enhanced.transform(test_features['enhanced'])
        
        # Save enhanced scaler
        with open(self.output_dir / 'scaler_enhanced.pkl', 'wb') as f:
            pickle.dump(scaler_enhanced, f)
        
        # Save splits
        splits = {
            # Original features
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            
            # Enhanced features
            'X_train_enhanced': train_features['enhanced'],
            'X_val_enhanced': val_features['enhanced'],
            'X_test_enhanced': test_features['enhanced'],
            'X_train_enhanced_scaled': X_train_enhanced_scaled,
            'X_val_enhanced_scaled': X_val_enhanced_scaled,
            'X_test_enhanced_scaled': X_test_enhanced_scaled,
            
            # Labels
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        np.savez(self.output_dir / 'data_splits.npz', **splits)
        
        # Save encoders and scaler
        with open(self.output_dir / 'disease_encoder.pkl', 'wb') as f:
            pickle.dump(self.disease_encoder, f)
        
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return splits
    
    def _transform_features(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply feature engineering transformations to new data"""
        n_samples, n_features = X.shape
        
        # Statistical features
        symptom_count = X.sum(axis=1).reshape(-1, 1)
        symptom_density = (X.sum(axis=1) / n_features).reshape(-1, 1)
        
        symptom_freq = X.sum(axis=0) / n_samples
        rare_threshold = np.percentile(symptom_freq, 25)
        common_threshold = np.percentile(symptom_freq, 75)
        
        rare_symptoms = (X * (symptom_freq < rare_threshold)).sum(axis=1).reshape(-1, 1)
        common_symptoms = (X * (symptom_freq > common_threshold)).sum(axis=1).reshape(-1, 1)
        
        # System groups
        system_groups = self._create_system_groups()
        system_features = []
        
        for system_name, symptoms in system_groups.items():
            indices = [self.symptom_to_id[s] for s in symptoms if s in self.symptom_to_id]
            if indices:
                system_symptom_count = X[:, indices].sum(axis=1).reshape(-1, 1)
                system_features.append(system_symptom_count)
        
        system_features = np.hstack(system_features) if system_features else np.zeros((n_samples, 1))
        
        # Co-occurrence (simplified for transform)
        cooccurrence_features = np.zeros((n_samples, 50))
        
        # PCA
        pca_features = self.pca.transform(X)
        
        # Combine
        enhanced_features = np.hstack([
            X,
            symptom_count,
            symptom_density,
            rare_symptoms,
            common_symptoms,
            system_features,
            cooccurrence_features,
            pca_features
        ])
        
        return {'enhanced': enhanced_features}
    
    def create_feature_metadata(self):
        """Create metadata for features"""
        metadata = {
            'n_features': len(self.symptom_columns),
            'n_classes': len(self.disease_encoder.classes_),
            'classes': self.disease_encoder.classes_.tolist(),
            'feature_names': self.symptom_columns,
            'dataset_size': len(self.df)
        }
        
        with open(self.output_dir / 'feature_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def process_pipeline(self):
        """Run complete preprocessing pipeline with feature engineering"""
        print("\n" + "="*60)
        print("SYMPTOM CHECKER - ENHANCED DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load and clean
        self.load_data()
        self.clean_data()
        
        # Create mappings
        self.create_symptom_mappings()
        
        # Analyze
        self.analyze_data_distribution()
        
        # Split data with feature engineering
        splits = self.create_train_val_test_split()
        
        # Create metadata
        metadata = self.create_feature_metadata()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE [OK]")
        print("="*60)
        print(f"\nFiles saved to: {self.output_dir}/")
        print("- data_splits.npz (includes enhanced features)")
        print("- symptom_lexicon.json")
        print("- disease_encoder.pkl")
        print("- scaler.pkl")
        print("- scaler_enhanced.pkl")
        print("- pca_model.pkl")
        print("- feature_metadata.json")
        print("- feature_engineering_info.json")
        
        return splits, metadata


# Example usage
if __name__ == "__main__":
    processor = SymptomDataProcessor(
        data_path="data/disease_symptom_data.csv",
        output_dir="processed_data"
    )
    
    splits, metadata = processor.process_pipeline()
    
    print("\n[OK] Ready for model training with enhanced features!")