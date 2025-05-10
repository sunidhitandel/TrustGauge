import os
from pyspark.sql import SparkSession
from data.processor import create_spark_session, standardize_data, apply_data_quality_checks
from models.sentiment import SentimentAnalyzer
from models.trust_score import TrustScoreCalculator
from visualization.plots import TrustGaugeVisualizer

def main():
    # Initialize Spark session
    spark = create_spark_session()
    
    try:
        # 1. Data Loading
        print("Loading data from S3...")
        raw_df = spark.read.option("header", True) \
            .option("delimiter", "\t") \
            .csv("s3://trustgauge-bucket/datasets/*.tsv")
        
        initial_count = raw_df.count()
        print(f"Loaded {initial_count:,} records")
        
        # 2. Data Standardization
        print("\nStandardizing data...")
        standardized_df = standardize_data(raw_df)
        standardized_count = standardized_df.count()
        
        # 3. Data Quality Checks
        print("\nApplying data quality checks...")
        quality_df = apply_data_quality_checks(standardized_df)
        final_count = quality_df.count()
        
        # 4. Sentiment Analysis
        print("\nPerforming sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer()
        df_with_sentiment = sentiment_analyzer.process_batch(quality_df)
        
        # 5. Trust Score Calculation
        print("\nCalculating trust scores...")
        trust_calculator = TrustScoreCalculator()
        product_scores = trust_calculator.calculate_product_scores(df_with_sentiment)
        company_scores = trust_calculator.calculate_company_scores(df_with_sentiment)
        final_df = trust_calculator.create_final_dataset(
            df_with_sentiment, product_scores, company_scores
        )
        
        # 6. Save Results
        print("\nSaving results...")
        final_df.write.format("delta").mode("overwrite") \
            .saveAsTable("trustgauge.trust_scores.master_trust_score")
        
        # 7. Generate Visualizations
        print("\nGenerating visualizations...")
        visualizer = TrustGaugeVisualizer()
        
        # Save visualizations
        plots = {
            "top_categories": visualizer.plot_top_categories(final_df),
            "rating_distribution": visualizer.plot_rating_distribution(final_df),
            "trust_score_distribution": visualizer.plot_trust_score_distribution(final_df),
            "category_trust_scores": visualizer.plot_category_trust_scores(final_df),
            "pipeline_metrics": visualizer.plot_pipeline_metrics(
                initial_count, standardized_count, final_count
            )
        }
        
        # Save plots to HTML files
        os.makedirs("reports", exist_ok=True)
        for name, fig in plots.items():
            fig.write_html(f"reports/{name}.html")
        
        print("\nPipeline completed successfully!")
        print(f"Final dataset contains {final_df.count():,} records")
        print("Visualizations saved to reports/ directory")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 