import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from typing import Dict

class FakeReviewDetector:
    """Class for detecting potential fake reviews using anomaly detection."""
    
    def __init__(self, config: Dict):
        """Initialize the fake review detector.
        
        Args:
            config (Dict): Configuration for fake review detection
        """
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.isolation_forest = IsolationForest(
            contamination=config.get('contamination', 0.1),
            random_state=config.get('random_seed', 42)
        )
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for fake review detection.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            np.ndarray: Feature matrix
        """
        # Text features
        text_features = self.vectorizer.fit_transform(df['processed_text'])
        
        # Metadata features
        metadata_features = np.column_stack([
            df['verified_purchase'].map({'Y': 1, 'N': 0}),
            df['helpful_votes'].values,
            df['star_rating'].values
        ])
        
        # Combine features
        return np.hstack([
            text_features.toarray(),
            metadata_features
        ])
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect potential fake reviews.
        
        Args:
            df (pd.DataFrame): Input DataFrame with review data
            
        Returns:
            pd.DataFrame: DataFrame with fake detection probabilities
        """
        # Create copy of DataFrame
        detected_df = df.copy()
        
        # Extract features
        features = self._extract_features(detected_df)
        
        # Fit and predict
        predictions = self.isolation_forest.fit_predict(features)
        
        # Convert predictions to probabilities
        # -1 indicates anomaly (fake), 1 indicates normal
        detected_df['fake_probability'] = np.where(
            predictions == -1,
            0.8 + np.random.uniform(0, 0.2, size=len(predictions)),  # High probability for anomalies
            np.random.uniform(0, 0.2, size=len(predictions))  # Low probability for normal reviews
        )
        
        return detected_df 