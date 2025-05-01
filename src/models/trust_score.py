from pyspark.sql.functions import (
    col, when, datediff, current_timestamp, 
    exp, sum as spark_sum, count, avg
)
from pyspark.sql.window import Window
import logging
from typing import Dict

class TrustScoreCalculator:
    """Calculate trust scores for products and sellers based on reviews."""
    
    def __init__(self, config: Dict):
        """
        Initialize the trust score calculator.
        
        Args:
            config: Configuration dictionary containing trust score settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def _calculate_time_decay(self, df):
        """Calculate time decay factor for reviews based on age."""
        half_life = self.config['time_decay']['half_life_days']
        
        return df.withColumn(
            "time_decay",
            exp(
                -0.693 * 
                datediff(current_timestamp(), col("review_date")) / 
                half_life
            )
        )
    
    def _calculate_helpfulness_ratio(self, df):
        """Calculate helpfulness ratio for reviews."""
        return df.withColumn(
            "helpful_ratio",
            when(col("total_votes") > 0,
                 col("helpful_votes") / col("total_votes")
            ).otherwise(0.0)
        )
    
    def _normalize_star_rating(self, df):
        """Normalize star ratings to 0-1 scale."""
        return df.withColumn(
            "normalized_rating",
            (col("star_rating") - 1) / 4.0
        )
    
    def calculate_review_trust(self, df):
        """
        Calculate trust score for individual reviews.
        
        Args:
            df: Spark DataFrame with review data
            
        Returns:
            DataFrame with added review trust scores
        """
        weights = self.config['weights']
        
        # Add time decay factor
        df = self._calculate_time_decay(df)
        
        # Add helpfulness ratio
        df = self._calculate_helpfulness_ratio(df)
        
        # Normalize star rating
        df = self._normalize_star_rating(df)
        
        # Calculate review trust score
        return df.withColumn(
            "review_trust_score",
            (weights['sentiment'] * col("sentiment_score") +
             weights['rating'] * col("normalized_rating") +
             weights['verified'] * when(col("verified_purchase") == "Y", 1.0).otherwise(0.0) +
             weights['helpfulness'] * col("helpful_ratio") +
             weights['recency'] * col("time_decay"))
        )
    
    def calculate_product_trust(self, df):
        """
        Calculate aggregate trust score for products.
        
        Args:
            df: Spark DataFrame with review trust scores
            
        Returns:
            DataFrame with product trust scores
        """
        # Define window for product-level aggregation
        product_window = Window.partitionBy("product_id")
        
        # Calculate product trust metrics
        product_trust = df.select(
            "product_id",
            "product_title",
            "review_trust_score",
            "sentiment_score",
            "normalized_rating",
            "verified_purchase",
            "helpful_ratio",
            "time_decay"
        ).groupBy("product_id", "product_title").agg(
            avg("review_trust_score").alias("product_trust_score"),
            avg("sentiment_score").alias("avg_sentiment"),
            avg("normalized_rating").alias("avg_rating"),
            (spark_sum(when(col("verified_purchase") == "Y", 1))
             / count("*")).alias("verified_ratio"),
            avg("helpful_ratio").alias("avg_helpful_ratio"),
            avg("time_decay").alias("recency_score"),
            count("*").alias("review_count")
        )
        
        return product_trust
    
    def calculate_seller_trust(self, df):
        """
        Calculate aggregate trust score for sellers/brands.
        
        Args:
            df: Spark DataFrame with product trust scores
            
        Returns:
            DataFrame with seller trust scores
        """
        # Define window for seller-level aggregation
        seller_window = Window.partitionBy("seller_id")
        
        # Calculate seller trust metrics
        seller_trust = df.select(
            "seller_id",
            "seller_name",
            "product_trust_score",
            "avg_sentiment",
            "avg_rating",
            "verified_ratio",
            "review_count"
        ).groupBy("seller_id", "seller_name").agg(
            avg("product_trust_score").alias("seller_trust_score"),
            avg("avg_sentiment").alias("seller_avg_sentiment"),
            avg("avg_rating").alias("seller_avg_rating"),
            avg("verified_ratio").alias("seller_verified_ratio"),
            spark_sum("review_count").alias("total_reviews")
        )
        
        return seller_trust
    
    def calculate(self, df):
        """
        Calculate all trust scores.
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            DataFrame with all trust scores
        """
        try:
            self.logger.info("Calculating review trust scores...")
            df_with_review_trust = self.calculate_review_trust(df)
            
            self.logger.info("Calculating product trust scores...")
            product_trust = self.calculate_product_trust(df_with_review_trust)
            
            self.logger.info("Calculating seller trust scores...")
            seller_trust = self.calculate_seller_trust(product_trust)
            
            # Join all trust scores back to original DataFrame
            result = df_with_review_trust.join(
                product_trust,
                on=["product_id"],
                how="left"
            )
            
            if "seller_id" in df.columns:
                result = result.join(
                    seller_trust,
                    on=["seller_id"],
                    how="left"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating trust scores: {str(e)}")
            raise 