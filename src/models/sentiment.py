import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DoubleType

class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load BERT model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_single_text(self, text):
        """Process a single text for sentiment analysis"""
        if not text or pd.isna(text):
            return 3.0  # Neutral sentiment
        
        try:
            # Truncate long texts to avoid memory issues
            if len(text) > 1000:
                text = text[:1000]
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                weighted_avg = (scores * torch.arange(1, 6).to(self.device)).sum(dim=1)
                return float(weighted_avg.cpu().numpy()[0])
        except Exception as e:
            print(f"Error processing text: {e}")
            return 3.0  # Neutral sentiment on error

    def create_sentiment_udf(self):
        """Create a pandas UDF for sentiment analysis"""
        @pandas_udf(DoubleType())
        def sentiment_udf(review_series: pd.Series) -> pd.Series:
            return review_series.apply(lambda text: self.process_single_text(text) if text else 3.0)
        
        return sentiment_udf

    def process_batch(self, df, batch_size=50):
        """Process dataframe in batches to avoid memory issues"""
        sentiment_udf = self.create_sentiment_udf()
        return df.withColumn("bert_sentiment_score", 
                           sentiment_udf(col("review_body").cast("string"))) 