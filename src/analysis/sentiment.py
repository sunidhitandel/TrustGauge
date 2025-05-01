from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import logging
from typing import List, Dict
import numpy as np

class SentimentAnalyzer:
    """Sentiment analysis using RoBERTa model."""
    
    def __init__(self, config: Dict):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary containing NLP settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_model()
        
    def _setup_model(self):
        """Load and setup the RoBERTa model."""
        try:
            self.logger.info("Loading RoBERTa model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['roberta']['model_name']
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['roberta']['model_name']
            )
            
            # Move model to GPU if available
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Error loading RoBERTa model: {str(e)}")
            raise
    
    def _batch_texts(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Split texts into batches."""
        return [
            texts[i:i + batch_size] 
            for i in range(0, len(texts), batch_size)
        ]
    
    def _predict_batch(self, batch: List[str]) -> List[float]:
        """
        Predict sentiment scores for a batch of texts.
        
        Args:
            batch: List of text strings
            
        Returns:
            List of sentiment scores between 0 and 1
        """
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config['roberta']['max_length'],
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                
            # Convert to sentiment scores (0 to 1)
            return scores[:, 1].cpu().numpy().tolist()
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            return [0.5] * len(batch)  # Neutral sentiment as fallback
    
    def predict(self, texts: List[str]) -> List[float]:
        """
        Predict sentiment scores for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment scores between 0 and 1
        """
        all_scores = []
        batches = self._batch_texts(
            texts, 
            self.config['roberta']['batch_size']
        )
        
        for batch in batches:
            scores = self._predict_batch(batch)
            all_scores.extend(scores)
            
        return all_scores
    
    def analyze(self, df):
        """
        Add sentiment scores to the Spark DataFrame.
        
        Args:
            df: Spark DataFrame containing review text
            
        Returns:
            DataFrame with added sentiment scores
        """
        # Register UDF for sentiment prediction
        @udf(FloatType())
        def sentiment_udf(text):
            if not text:
                return 0.5
            return float(self.predict([text])[0])
        
        # Add sentiment scores
        return df.withColumn(
            "sentiment_score",
            sentiment_udf("review_body")
        )
    
    def analyze_batch(self, df, batch_size=1000):
        """
        Batch process sentiment analysis for large datasets.
        
        Args:
            df: Spark DataFrame
            batch_size: Number of reviews to process at once
            
        Returns:
            DataFrame with sentiment scores
        """
        # Collect review texts in batches
        reviews = df.select("review_body").rdd.flatMap(lambda x: x).collect()
        
        # Process batches
        all_scores = []
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            scores = self.predict(batch)
            all_scores.extend(scores)
        
        # Create sentiment score column
        sentiment_df = df.sparkSession.createDataFrame(
            [(float(score),) for score in all_scores],
            ["sentiment_score"]
        )
        
        # Join with original DataFrame
        return df.join(sentiment_df) 