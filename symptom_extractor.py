"""
Transformer-based Symptom Extraction from Natural Language
Uses sentence-transformers for semantic matching of symptoms
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import re


class SymptomExtractor:
    """
    Extract symptoms from natural language using semantic similarity.
    Uses sentence-transformers to match user descriptions to known symptoms.
    """
    
    def __init__(self, 
                 processed_data_dir: str = 'processed_data',
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.4):
        """
        Initialize symptom extractor
        
        Args:
            processed_data_dir: Directory containing symptom lexicon
            model_name: Sentence transformer model name
            similarity_threshold: Minimum cosine similarity for matching (0-1)
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.similarity_threshold = similarity_threshold
        
        print("Loading Symptom Extractor...")
        
        # Load sentence transformer model
        print(f"  Loading transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("  ✓ Model loaded")
        
        # Load symptom lexicon
        self.load_symptom_lexicon()
        
        # Create symptom embeddings
        self.create_symptom_embeddings()
        
        # Build synonym dictionary
        self.build_symptom_synonyms()
        
        print("Symptom Extractor ready!\n")
    
    def load_symptom_lexicon(self):
        """Load symptom names from lexicon"""
        lexicon_path = self.processed_data_dir / 'symptom_lexicon.json'
        
        if not lexicon_path.exists():
            raise FileNotFoundError(
                f"Symptom lexicon not found: {lexicon_path}\n"
                f"Please run training first to generate the lexicon."
            )
        
        with open(lexicon_path, 'r') as f:
            lexicon = json.load(f)
            self.symptoms = lexicon['symptoms']
        
        print(f"  ✓ Loaded {len(self.symptoms)} symptoms")
    
    def create_symptom_embeddings(self):
        """Create embeddings for all symptoms"""
        print("  Creating symptom embeddings...")
        
        # Convert symptom names to more natural descriptions
        symptom_descriptions = [
            self._symptom_to_description(s) for s in self.symptoms
        ]
        
        # Encode all symptoms
        self.symptom_embeddings = self.model.encode(
            symptom_descriptions,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        print(f"  ✓ Created embeddings ({self.symptom_embeddings.shape})")
    
    def _symptom_to_description(self, symptom: str) -> str:
        """
        Convert symptom code to natural description
        
        Example: 'stomach_pain' -> 'stomach pain'
        """
        return symptom.replace('_', ' ').strip()
    
    def build_symptom_synonyms(self):
        """
        Build dictionary of common symptom synonyms
        """
        self.synonyms = {
            # Pain synonyms
            'hurt': 'pain',
            'hurts': 'pain',
            'ache': 'pain',
            'aching': 'pain',
            'sore': 'pain',
            'painful': 'pain',
            
            # Fever synonyms
            'hot': 'fever',
            'burning up': 'fever',
            'high temperature': 'fever',
            
            # Fatigue synonyms
            'tired': 'fatigue',
            'exhausted': 'fatigue',
            'weak': 'fatigue',
            'weakness': 'fatigue',
            'no energy': 'fatigue',
            
            # Cough synonyms
            'coughing': 'cough',
            
            # Nausea synonyms
            'sick': 'nausea',
            'feel sick': 'nausea',
            'queasy': 'nausea',
            
            # Dizzy synonyms
            'dizziness': 'dizzy',
            'lightheaded': 'dizzy',
            'vertigo': 'dizzy',
            
            # Headache synonyms
            'head hurts': 'headache',
            'head pain': 'headache',
            'migraine': 'headache',
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess user input text
        
        Args:
            text: Raw user input
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Apply synonyms
        for synonym, replacement in self.synonyms.items():
            if synonym in text:
                text = text.replace(synonym, replacement)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def detect_negations(self, text: str) -> List[str]:
        """
        Detect negated symptoms
        
        Args:
            text: User input text
            
        Returns:
            List of negated symptom phrases
        """
        negation_patterns = [
            r"no\s+(\w+(?:\s+\w+)?)",
            r"not\s+(\w+(?:\s+\w+)?)",
            r"don't\s+have\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
            r"doesn't\s+hurt",
            r"never\s+(\w+)",
        ]
        
        negated = []
        for pattern in negation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    negated.append(match.group(1).lower())
        
        return negated
    
    def extract_symptoms(self, 
                        text: str,
                        top_k: int = 10,
                        return_scores: bool = True) -> List[Tuple[str, float]]:
        """
        Extract symptoms from natural language text
        
        Args:
            text: User's symptom description
            top_k: Maximum number of symptoms to extract
            return_scores: Whether to return confidence scores
            
        Returns:
            List of (symptom_name, confidence_score) tuples if return_scores=True,
            otherwise just list of symptom names
        """
        if not text or not text.strip():
            return []
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Detect negations
        negated_terms = self.detect_negations(text)
        
        # Encode user text
        text_embedding = self.model.encode(
            processed_text,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Compute similarities
        similarities = np.dot(self.symptom_embeddings, text_embedding)
        similarities = similarities / (
            np.linalg.norm(self.symptom_embeddings, axis=1) * 
            np.linalg.norm(text_embedding)
        )
        
        # Get top matches above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            
            if similarity < self.similarity_threshold:
                break
            
            symptom = self.symptoms[idx]
            symptom_desc = self._symptom_to_description(symptom)
            
            # Filter out negated symptoms
            is_negated = any(neg in symptom_desc for neg in negated_terms)
            if is_negated:
                continue
            
            if return_scores:
                results.append((symptom, similarity))
            else:
                results.append(symptom)
        
        return results
    
    def extract_symptoms_interactive(self, text: str) -> List[str]:
        """
        Extract symptoms with user-friendly output
        
        Args:
            text: User input
            
        Returns:
            List of extracted symptom names
        """
        results = self.extract_symptoms(text, return_scores=True)
        
        if not results:
            return []
        
        symptoms = [symptom for symptom, score in results]
        return symptoms
    
    def batch_extract(self, texts: List[str]) -> List[List[str]]:
        """
        Extract symptoms from multiple texts
        
        Args:
            texts: List of symptom descriptions
            
        Returns:
            List of symptom lists
        """
        all_symptoms = []
        for text in texts:
            symptoms = self.extract_symptoms_interactive(text)
            all_symptoms.append(symptoms)
        
        return all_symptoms


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = SymptomExtractor()
    
    # Test examples
    test_cases = [
        "I have a really bad headache and I feel dizzy",
        "My stomach hurts and I feel nauseous",
        "I'm running a fever and have a cough",
        "Feeling very tired and weak, also have muscle pain",
        "I have chest pain but no fever",
        "Headache but not dizzy"
    ]
    
    print("=" * 60)
    print("SYMPTOM EXTRACTION EXAMPLES")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\nInput: \"{text}\"")
        symptoms = extractor.extract_symptoms(text, return_scores=True)
        
        if symptoms:
            print("Extracted symptoms:")
            for symptom, score in symptoms:
                print(f"  • {symptom:30s} (confidence: {score:.2f})")
        else:
            print("  No symptoms detected")
