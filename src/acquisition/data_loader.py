from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

class AmazonReviewLoader:
    """Data loader for Amazon review datasets using PySpark."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize the data loader with a SparkSession."""
        self.spark = spark or self._create_spark_session()
        self._setup_logging()
        self._df = None
        
    def _create_spark_session(self) -> SparkSession:
        """Create and configure a new Spark session."""
        return (SparkSession.builder
                .appName("TrustGauge")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .getOrCreate())
    
    def _setup_logging(self):
        """Configure logging for the data loader."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def get_schema() -> StructType:
        """Define the schema for Amazon review datasets."""
        return StructType([
            StructField("marketplace", StringType(), True),
            StructField("customer_id", StringType(), True),
            StructField("review_id", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("product_parent", StringType(), True),
            StructField("product_title", StringType(), True),
            StructField("product_category", StringType(), True),
            StructField("star_rating", IntegerType(), True),
            StructField("helpful_votes", IntegerType(), True),
            StructField("total_votes", IntegerType(), True),
            StructField("vine", StringType(), True),
            StructField("verified_purchase", StringType(), True),
            StructField("review_headline", StringType(), True),
            StructField("review_body", StringType(), True),
            StructField("review_date", TimestampType(), True)
        ])
    
    def load_dataset(self, file_path: str, apply_quality_checks: bool = True) -> Optional[SparkSession]:
        """
        Load an Amazon review dataset from TSV file.
        
        Args:
            file_path: Path to the TSV file
            apply_quality_checks: Whether to apply data quality checks
            
        Returns:
            Spark DataFrame with loaded and cleaned data
        """
        try:
            self.logger.info(f"Loading dataset from {file_path}")
            
            # Read TSV file with schema
            df = (self.spark.read.format("csv")
                 .option("header", "true")
                 .option("delimiter", "\t")
                 .option("quote", "\"")
                 .option("escape", "\"")
                 .schema(self.get_schema())
                 .load(file_path))
            
            if apply_quality_checks:
                df = self._apply_quality_checks(df)
            
            self.logger.info(f"Successfully loaded dataset with {df.count()} rows")
            self._df = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            return None
    
    def _apply_quality_checks(self, df):
        """Apply data quality checks and cleaning."""
        # Remove rows with null values in critical columns
        critical_columns = ["review_id", "product_id", "star_rating", "review_body"]
        for col_name in critical_columns:
            df = df.filter(col(col_name).isNotNull())
        
        # Clean review text
        df = df.withColumn(
            "review_body",
            regexp_replace(col("review_body"), "[^\\w\\s]", " ")
        )
        
        # Convert ratings to integers and validate range
        df = df.filter(
            (col("star_rating") >= 1) & (col("star_rating") <= 5)
        )
        
        # Add data quality metrics
        df = df.withColumn(
            "review_length",
            length(col("review_body"))
        )
        
        return df
    
    def load_multiple_datasets(self, file_paths: List[str]) -> Optional[SparkSession]:
        """Load and union multiple Amazon review datasets."""
        dfs = []
        for file_path in file_paths:
            df = self.load_dataset(file_path)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            self.logger.error("No datasets were successfully loaded")
            return None
        
        return reduce(DataFrame.unionAll, dfs)
    
    def sample_dataset(
        self,
        df: pd.DataFrame,
        fraction: float = 0.1,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Sample a fraction of the dataset for testing purposes.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            fraction (float): Fraction of data to sample (default: 0.1)
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Sampled dataset
        """
        if seed is not None:
            np.random.seed(seed)
        
        return df.sample(frac=fraction)
    
    def close(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            self.logger.info("Spark session stopped")
            self._df = None 