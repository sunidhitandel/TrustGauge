from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lit, length, expr, year, to_date
from pyspark.sql.types import StringType, IntegerType, DateType, FloatType

def create_spark_session():
    """Create and configure Spark session"""
    return SparkSession.builder \
        .appName("TrustGauge-Data-Processing") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

def standardize_data(df):
    """Apply standardization steps to the raw dataset"""
    print("Starting data standardization...")
    
    # Create a copy to avoid modifying the cached raw data
    std_df = df
    
    # Marketplace: Ensure all values are 2-letter uppercase strings
    std_df = std_df.withColumn("marketplace",
                              when(col("marketplace") == "US", "US")
                              .otherwise("OTHER"))
    
    # Customer_id: Convert to string
    std_df = std_df.withColumn("customer_id", col("customer_id").cast(StringType()))
    
    # Review_id: Cast to string
    std_df = std_df.withColumn("review_id", col("review_id").cast(StringType()))
    
    # Product_id: Cast to string, drop if null
    std_df = std_df.withColumn("product_id", col("product_id").cast(StringType()))
    std_df = std_df.filter(col("product_id").isNotNull())
    
    # Product_parent: Cast to string
    std_df = std_df.withColumn("product_parent", col("product_parent").cast(StringType()))
    
    # Product_title: Trim
    std_df = std_df.withColumn("product_title", trim(col("product_title")))
    
    # Product_category: Trim, fill missing with "unknown"
    std_df = std_df.withColumn("product_category",
                              when(col("product_category").isNull(), "unknown")
                              .otherwise(trim(col("product_category"))))
    
    # Star_rating: Cast to IntegerType, constrain to 1-5
    std_df = std_df.withColumn("star_rating",
                              when(col("star_rating") > 5, 5)
                              .when(col("star_rating") < 1, 1)
                              .otherwise(col("star_rating"))
                              .cast(IntegerType()))
    
    # Helpful_votes: Cast to integer, fill missing with 0
    std_df = std_df.withColumn("helpful_votes",
                              when(col("helpful_votes").isNull(), 0)
                              .otherwise(col("helpful_votes").cast(IntegerType())))
    
    # Total_votes: Cast to integer, fill missing with 0
    std_df = std_df.withColumn("total_votes",
                              when(col("total_votes").isNull(), 0)
                              .otherwise(col("total_votes").cast(IntegerType())))
    
    # Convert review_date to proper date format
    std_df = std_df.withColumn("review_date", to_date(col("review_date"), "yyyy-MM-dd"))
    
    print("Data standardization completed")
    return std_df

def apply_data_quality_checks(df):
    """Apply data quality checks to the standardized dataset"""
    print("Applying data quality checks...")
    
    # 1. Keep only US marketplace records
    dq_df = df.filter(col("marketplace") == "US")
    
    # 2. Handle outliers in helpful_votes and total_votes using IQR
    # First calculate quantiles for helpful_votes
    helpful_votes_stats = dq_df.select("helpful_votes").summary("25%", "75%").collect()
    helpful_q1 = float(helpful_votes_stats[0]["helpful_votes"])
    helpful_q3 = float(helpful_votes_stats[1]["helpful_votes"])
    helpful_iqr = helpful_q3 - helpful_q1
    helpful_upper = helpful_q3 + 1.5 * helpful_iqr
    
    # Calculate quantiles for total_votes
    total_votes_stats = dq_df.select("total_votes").summary("25%", "75%").collect()
    total_q1 = float(total_votes_stats[0]["total_votes"])
    total_q3 = float(total_votes_stats[1]["total_votes"])
    total_iqr = total_q3 - total_q1
    total_upper = total_q3 + 1.5 * total_iqr
    
    # Apply outlier clipping
    dq_df = dq_df.withColumn("helpful_votes",
                            when(col("helpful_votes") > helpful_upper, helpful_upper)
                            .otherwise(col("helpful_votes")))
    
    dq_df = dq_df.withColumn("total_votes",
                           when(col("total_votes") > total_upper, total_upper)
                           .otherwise(col("total_votes")))
    
    # 3. Drop invalid review dates
    dq_df = dq_df.filter(col("review_date").isNotNull())
    
    # 4. Handle star_rating of 0
    dq_df = dq_df.filter(col("star_rating") > 0)
    
    print("Data quality checks completed")
    return dq_df 