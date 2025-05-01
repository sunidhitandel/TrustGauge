import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

class TextProcessor:
    """Class for preprocessing review text data."""
    
    def __init__(self, config: dict):
        """Initialize the text processor.
        
        Args:
            config (dict): Configuration dictionary for NLP settings
        """
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean raw text by removing special characters and extra whitespace.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Processed text with lemmatized tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(tokens)
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the review text in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with review text
            
        Returns:
            pd.DataFrame: DataFrame with processed text
        """
        # Create copy of DataFrame
        processed_df = df.copy()
        
        # Apply text processing to review text column
        processed_df['processed_text'] = (
            processed_df['review_body']
            .fillna('')
            .apply(self.clean_text)
            .apply(self.tokenize_and_lemmatize)
        )
        
        return processed_df 