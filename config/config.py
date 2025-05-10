# AWS Configuration
AWS_CONFIG = {
    'bucket_name': 'trustgauge-bucket',
    'region': 'us-east-1',
    'glue_job_name': 'TrustGauge-ETL',
    'glue_version': '3.0',
    'worker_type': 'G.1X',
    'number_of_workers': 5
}

# Data Processing Configuration
DATA_CONFIG = {
    'input_path': 's3://trustgauge-bucket/datasets/',
    'output_path': 's3://trustgauge-bucket/processed/',
    'temp_path': 's3://trustgauge-bucket/temp/',
    'batch_size': 50,
    'max_review_length': 1000
}

# Model Configuration
MODEL_CONFIG = {
    'bert_model': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'rating_weight': 0.7,
    'sentiment_weight': 0.3,
    'device': 'cuda'  # or 'cpu'
}

# Visualization Configuration
VIZ_CONFIG = {
    'color_scale': 'RdYlGn',
    'template': 'plotly_white',
    'top_n_categories': 10,
    'output_dir': 'reports/'
}

# Spark Configuration
SPARK_CONFIG = {
    'driver_memory': '8g',
    'executor_memory': '8g',
    'max_result_size': '4g',
    'shuffle_partitions': 200
}

# Database Configuration
DB_CONFIG = {
    'catalog': 'trustgauge',
    'schema': 'trust_scores',
    'tables': {
        'master': 'master_trust_score',
        'reviews': 'review_master_table'
    }
} 