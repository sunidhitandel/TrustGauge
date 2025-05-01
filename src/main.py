import argparse
import yaml
import logging
from pathlib import Path
from acquisition.data_loader import AmazonReviewLoader
from preprocessing.text_processor import TextProcessor
from models.trust_score import TrustScoreCalculator
from models.fake_detection import FakeReviewDetector
from analysis.sentiment import SentimentAnalyzer
from visualization.dashboard import TrustGaugeDashboard

def setup_logging(config):
    """Configure logging based on config settings."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        filename=config['logging']['file']
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TrustGauge: Amazon Review Analysis System")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to Amazon review dataset (TSV file)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use only a sample of the dataset for testing"
    )
    return parser.parse_args()

def main():
    """Main execution pipeline."""
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)
    logger = setup_logging(config)
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        loader = AmazonReviewLoader()
        
        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}")
        df = loader.load_dataset(args.dataset)
        
        if args.sample:
            logger.info("Using sample dataset for testing")
            df = loader.sample_dataset(
                df, 
                fraction=config['data']['sample_fraction'],
                seed=config['data']['random_seed']
            )
        
        # Initialize processors and models
        text_processor = TextProcessor(config['nlp'])
        sentiment_analyzer = SentimentAnalyzer(config['nlp'])
        trust_calculator = TrustScoreCalculator(config['trust_score'])
        fake_detector = FakeReviewDetector(config['fake_detection'])
        
        # Process text and compute sentiment
        logger.info("Processing review text and computing sentiment...")
        df = text_processor.process(df)
        df = sentiment_analyzer.analyze(df)
        
        # Detect fake reviews
        logger.info("Detecting potential fake reviews...")
        df = fake_detector.detect(df)
        
        # Calculate trust scores
        logger.info("Calculating trust scores...")
        df = trust_calculator.calculate(df)
        
        # Save processed data
        output_path = Path(config['data']['processed_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        df.write.parquet(
            str(output_path / "processed_reviews.parquet"),
            mode="overwrite"
        )
        
        # Launch dashboard
        logger.info("Launching visualization dashboard...")
        dashboard = TrustGaugeDashboard(config['visualization'])
        dashboard.launch(df)
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}")
        raise
    
    finally:
        loader.close()

if __name__ == "__main__":
    main() 