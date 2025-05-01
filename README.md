# TrustGauge

TrustGauge is an advanced Amazon review analysis system that helps assess the trustworthiness of product reviews. It combines sentiment analysis, fake review detection, and trust scoring to provide comprehensive insights into review quality.

## Features

- Sentiment analysis using DistilBERT
- Fake review detection using anomaly detection
- Trust score calculation based on multiple factors
- Interactive visualization dashboard
- Support for large-scale review analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sunidhitandel/TrustGauge.git
cd TrustGauge
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your Amazon review dataset in TSV format with the following columns:
   - review_id
   - product_id
   - star_rating
   - helpful_votes
   - review_body
   - verified_purchase

2. Run the analysis:
```bash
python src/main.py --dataset path/to/your/reviews.tsv [--sample]
```

Options:
- `--dataset`: Path to your Amazon review dataset (TSV format)
- `--sample`: Use only a sample of the dataset for testing
- `--config`: Path to custom configuration file (default: config/config.yaml)

3. Access the dashboard:
   - Open your web browser
   - Navigate to http://localhost:7860
   - Enter product IDs to view detailed analysis

## Configuration

The system can be configured through `config/config.yaml`. Key configuration options:

- Data sampling and processing settings
- NLP model parameters
- Trust score weights
- Visualization settings
- Logging configuration

## Directory Structure

```
TrustGauge/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── acquisition/
│   ├── analysis/
│   ├── models/
│   ├── preprocessing/
│   ├── visualization/
│   └── main.py
├── logs/
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 