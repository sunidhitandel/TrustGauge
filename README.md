# TrustGauge: Advanced Amazon Review Analysis System

## Overview
TrustGauge is a comprehensive big data analysis system that processes Amazon product reviews to generate trust scores, detect fake reviews, and provide deep insights using advanced NLP techniques and distributed computing.

## Architecture
![Architecture Diagram](docs/architecture.png)

## Key Features
1. Company-Wide Trust Index
2. Fake Review Detection
3. Aspect-Based Sentiment Analysis (ABSA)
4. Trust-Based Product Leaderboard
5. Intent Detection
6. Review Summarization
7. Review-Product Graph Analytics
8. Temporal Trust Evolution Analysis
9. Hybrid Trust Score System
10. Smart Product Recommendations

## Tech Stack
- **Core Technologies**:
  - PySpark & Spark ML for distributed computing
  - Python 3.9+
  - NLTK & spaCy for NLP
  - RoBERTa & LoRA for advanced language models
  - OpenAI API integration
  - Gradio for interactive UI

## Project Structure
```
TrustGauge/
├── data/                      # Data storage and preprocessing
├── src/
│   ├── acquisition/          # Data ingestion modules
│   ├── preprocessing/        # Data cleaning and preparation
│   ├── models/              # ML models and algorithms
│   ├── analysis/            # Core analysis modules
│   ├── visualization/       # Dashboard and plotting
│   └── utils/              # Helper functions
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── docs/                   # Documentation
└── ui/                     # Gradio interface
```

## Setup Instructions

### Prerequisites
1. Python 3.9+
2. Apache Spark 3.4+
3. Java 8/11
4. Virtual environment tool (conda/venv)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/TrustGauge.git
cd TrustGauge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running on Cloud Platforms

#### Google Colab
1. Open the notebooks in `notebooks/colab/`
2. Follow the setup instructions in each notebook
3. Make sure to mount your Google Drive for data storage

#### Databricks
1. Import the notebooks from `notebooks/databricks/`
2. Configure cluster with required dependencies
3. Follow the execution instructions in each notebook

## Data Pipeline

### 1. Data Ingestion
- Supports multiple Amazon review datasets
- Handles TSV format with 555 columns
- Implements data quality checks

### 2. Preprocessing
- Text cleaning and normalization
- Feature extraction
- Sentiment analysis using RoBERTa

### 3. Analysis
- Trust score computation
- Fake review detection
- Aspect-based sentiment analysis

### 4. Visualization
- Interactive dashboards
- Temporal analysis plots
- Product comparison views

## Usage
1. Configure your environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other credentials
```

2. Run the data pipeline:
```bash
python src/main.py --dataset "amazon_reviews_us_Electronics_v1_00.tsv"
```

3. Launch the UI:
```bash
python src/ui/app.py
```

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 