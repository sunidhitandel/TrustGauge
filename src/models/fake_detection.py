from pyspark.sql.functions import (
    col, length, regexp_replace, count, 
    when, abs, stddev, avg, sum as spark_sum
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
import logging
from typing import Dict, List

class FakeReviewDetector:
    """Detect potentially fake reviews using ML techniques."""
    
    def __init__(self, config: Dict):
        """
        Initialize the fake review detector.
        
        Args:
            config: Configuration dictionary containing fake detection settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.threshold = config['threshold']
        self.features = config['features']
    
    def _extract_text_features(self, df):
        """Extract text-based features from reviews."""
        # Calculate text length
        df = df.withColumn(
            "review_length",
            length(col("review_body"))
        )
        
        # Calculate capital letter ratio
        df = df.withColumn(
            "capital_ratio",
            length(regexp_replace(col("review_body"), "[^A-Z]", "")) /
            length(col("review_body"))
        )
        
        # Calculate punctuation ratio
        df = df.withColumn(
            "punctuation_ratio",
            length(regexp_replace(col("review_body"), "[^\\p{Punct}]", "")) /
            length(col("review_body"))
        )
        
        return df
    
    def _calculate_behavioral_features(self, df):
        """Calculate behavioral features for reviews."""
        # Define windows for user and product level aggregations
        user_window = Window.partitionBy("customer_id")
        product_window = Window.partitionBy("product_id")
        
        # User behavior features
        df = df.withColumn(
            "user_review_count",
            count("*").over(user_window)
        ).withColumn(
            "user_avg_rating",
            avg("star_rating").over(user_window)
        ).withColumn(
            "rating_deviation",
            abs(col("star_rating") - col("user_avg_rating"))
        )
        
        # Product features
        df = df.withColumn(
            "product_review_count",
            count("*").over(product_window)
        ).withColumn(
            "product_avg_rating",
            avg("star_rating").over(product_window)
        ).withColumn(
            "product_rating_stddev",
            stddev("star_rating").over(product_window)
        )
        
        return df
    
    def _calculate_sentiment_extremity(self, df):
        """Calculate how extreme the sentiment is compared to norm."""
        return df.withColumn(
            "sentiment_extremity",
            abs(col("sentiment_score") - 0.5) * 2
        )
    
    def _prepare_features(self, df):
        """Prepare feature vector for ML model."""
        # Extract all required features
        df = self._extract_text_features(df)
        df = self._calculate_behavioral_features(df)
        df = self._calculate_sentiment_extremity(df)
        
        # Select features for vectorization
        feature_cols = [
            col(feature).cast("double") 
            for feature in self.features
        ]
        
        # Assemble feature vector
        assembler = VectorAssembler(
            inputCols=self.features,
            outputCol="features"
        )
        
        return assembler.transform(df)
    
    def _train_anomaly_detector(self, df):
        """Train K-means clustering for anomaly detection."""
        # Train K-means model
        kmeans = KMeans(
            k=2,
            featuresCol="features",
            predictionCol="cluster"
        )
        model = kmeans.fit(df)
        
        # Get cluster centers and distances
        df_with_clusters = model.transform(df)
        
        # Calculate distance to cluster center
        centers = model.clusterCenters()
        
        return df_with_clusters, centers
    
    def detect(self, df):
        """
        Detect potentially fake reviews.
        
        Args:
            df: Spark DataFrame with review data
            
        Returns:
            DataFrame with fake detection scores
        """
        try:
            self.logger.info("Preparing features for fake detection...")
            df_features = self._prepare_features(df)
            
            self.logger.info("Training anomaly detector...")
            df_clusters, centers = self._train_anomaly_detector(df_features)
            
            # Calculate suspicion score based on cluster assignment and distance
            df_with_scores = df_clusters.withColumn(
                "fake_review_score",
                when(
                    (col("cluster") == 1) &
                    (col("sentiment_extremity") > 0.8) &
                    (col("verified_purchase") != "Y"),
                    1.0
                ).otherwise(
                    when(
                        (col("cluster") == 1) |
                        (col("sentiment_extremity") > 0.8),
                        0.7
                    ).otherwise(0.3)
                )
            )
            
            # Flag potentially fake reviews
            result = df_with_scores.withColumn(
                "is_fake",
                col("fake_review_score") > self.threshold
            )
            
            # Add summary statistics
            window = Window.partitionBy("product_id")
            result = result.withColumn(
                "fake_review_ratio",
                spark_sum(when(col("is_fake"), 1).otherwise(0)).over(window) /
                count("*").over(window)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fake review detection: {str(e)}")
            raise
    
    def get_suspicious_patterns(self, df):
        """
        Get summary of suspicious patterns in reviews.
        
        Args:
            df: DataFrame with fake detection results
            
        Returns:
            DataFrame with suspicious pattern summaries
        """
        return df.filter(col("is_fake")).groupBy(
            "product_id",
            "product_title"
        ).agg(
            count("*").alias("fake_review_count"),
            avg("fake_review_score").alias("avg_suspicion_score"),
            avg("sentiment_extremity").alias("avg_sentiment_extremity"),
            spark_sum(when(col("verified_purchase") == "Y", 1))
            .alias("verified_fake_reviews")
        ) 