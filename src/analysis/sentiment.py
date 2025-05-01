import pandas as pd
from transformers import pipeline
from typing import Dict

class SentimentAnalyzer:
    """Class for analyzing sentiment in review text."""
    
    def __init__(self, config: Dict):
        """Initialize the sentiment analyzer.
        
        Args:
            config (Dict): Configuration for sentiment analysis
        """
        self.config = config
        self.model_name = config.get('model_name', 'distilbert-base-uncased-finetuned-sst-2-english')
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=-1  # Use CPU
        )
    
    def _analyze_text(self, text: str) -> float:
        """Analyze sentiment of a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Sentiment score between 0 and 1
        """
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Convert label and score to a single sentiment score
            if result['label'] == 'POSITIVE':
                return 0.5 + (result['score'] / 2)
            else:
                return 0.5 - (result['score'] / 2)
        except:
            return 0.5  # Neutral sentiment for errors
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for all reviews in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with review data
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment scores
        """
        # Create copy of DataFrame
        analyzed_df = df.copy()
        
        # Calculate sentiment scores
        analyzed_df['sentiment_score'] = (
            analyzed_df['processed_text']
            .fillna('')
            .apply(self._analyze_text)
        )
        
        return analyzed_df 