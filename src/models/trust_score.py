from pyspark.sql.functions import col, mean, round

class TrustScoreCalculator:
    def __init__(self, rating_weight=0.7, sentiment_weight=0.3):
        self.rating_weight = rating_weight
        self.sentiment_weight = sentiment_weight

    def calculate_product_scores(self, df):
        """Calculate trust scores for individual products"""
        # Group and compute averages
        product_scores = df.groupBy("product_id").agg(
            mean("average_product_rating").alias("average_product_rating"),
            mean("bert_sentiment_score").alias("bert_sentiment_score")
        )
        
        # Compute final product score
        product_scores = product_scores.withColumn(
            "final_product_score",
            self.rating_weight * col("average_product_rating") + 
            self.sentiment_weight * col("bert_sentiment_score")
        )
        
        return product_scores

    def calculate_company_scores(self, df):
        """Calculate trust scores for companies (product_parent)"""
        # Group and compute averages
        company_scores = df.groupBy("product_parent").agg(
            mean("average_product_rating").alias("average_product_rating"),
            mean("bert_sentiment_score").alias("bert_sentiment_score")
        )
        
        # Compute final company score
        company_scores = company_scores.withColumn(
            "final_company_score",
            self.rating_weight * col("average_product_rating") + 
            self.sentiment_weight * col("bert_sentiment_score")
        )
        
        return company_scores

    def create_final_dataset(self, df, product_scores, company_scores):
        """Create final dataset with all scores"""
        # Join all scores
        final_df = df \
            .join(product_scores, on="product_id", how="left") \
            .join(company_scores, on="product_parent", how="left")
        
        # Select and rename columns
        final_df = final_df.select(
            "product_id",
            "product_parent",
            "product_title",
            "product_category",
            "average_product_rating",
            "bert_sentiment_score",
            "final_product_score",
            "final_company_score"
        )
        
        return final_df 