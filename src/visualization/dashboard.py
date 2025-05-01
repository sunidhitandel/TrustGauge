import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

class TrustGaugeDashboard:
    """Class for creating an interactive dashboard to visualize trust scores."""
    
    def __init__(self, config: Dict):
        """Initialize the dashboard.
        
        Args:
            config (Dict): Configuration for visualization
        """
        self.config = config
        self.theme = config.get('theme', 'default')
        self.port = config.get('port', 7860)
    
    def _create_trust_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create trust score distribution plot.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.histogram(
            df,
            x='trust_score',
            nbins=50,
            title='Distribution of Trust Scores',
            labels={'trust_score': 'Trust Score', 'count': 'Number of Reviews'}
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def _create_sentiment_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Create sentiment vs trust score scatter plot.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.scatter(
            df,
            x='sentiment_score',
            y='trust_score',
            color='verified_purchase',
            title='Sentiment vs Trust Score',
            labels={
                'sentiment_score': 'Sentiment Score',
                'trust_score': 'Trust Score',
                'verified_purchase': 'Verified Purchase'
            }
        )
        return fig
    
    def _create_fake_probability_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create fake probability visualization.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.box(
            df,
            x='verified_purchase',
            y='fake_probability',
            title='Fake Probability by Verified Status',
            labels={
                'verified_purchase': 'Verified Purchase',
                'fake_probability': 'Fake Probability'
            }
        )
        return fig
    
    def _review_details(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """Get detailed review information for a product.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            product_id (str): Product ID to filter by
            
        Returns:
            pd.DataFrame: Filtered and formatted review details
        """
        product_reviews = df[df['product_id'] == product_id].copy()
        product_reviews = product_reviews[[
            'review_id',
            'star_rating',
            'review_body',
            'verified_purchase',
            'trust_score',
            'sentiment_score',
            'fake_probability'
        ]].sort_values('trust_score', ascending=False)
        
        return product_reviews
    
    def launch(self, df: pd.DataFrame):
        """Launch the interactive dashboard.
        
        Args:
            df (pd.DataFrame): Input DataFrame with review data
        """
        # Create interface
        with gr.Blocks(theme=self.theme) as interface:
            gr.Markdown("# TrustGauge Dashboard")
            
            with gr.Row():
                with gr.Column():
                    gr.Plot(self._create_trust_distribution(df))
                with gr.Column():
                    gr.Plot(self._create_sentiment_scatter(df))
            
            with gr.Row():
                with gr.Column():
                    gr.Plot(self._create_fake_probability_plot(df))
                with gr.Column():
                    product_input = gr.Textbox(
                        label="Enter Product ID",
                        placeholder="Type a product ID to see reviews..."
                    )
                    review_output = gr.DataFrame(
                        headers=[
                            "Review ID",
                            "Rating",
                            "Review Text",
                            "Verified",
                            "Trust Score",
                            "Sentiment",
                            "Fake Prob."
                        ]
                    )
            
            product_input.change(
                fn=lambda x: self._review_details(df, x),
                inputs=product_input,
                outputs=review_output
            )
        
        # Launch interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=self.port,
            share=True
        ) 